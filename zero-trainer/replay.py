"""Replay buffer and sliding window for the continuous trainer (plan §3.3).

Samples are stored as raw v2 bytes grouped per game (`Chunk`) and decoded only
when a batch is assembled, so the buffer costs ~700 B/sample.  `sample()`
returns exactly the 12-tuple that `train.compute_batch_loss` unpacks.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

from dataset import (
    BoardGeometry,
    MOVES_LEFT_SCALE,
    RawSample,
    _apply_board_sym,
    decode_aux_policy,
    decode_sample,
)


def replay_window(total: int, base: int = 250_000, alpha: float = 0.75,
                  beta: float = 0.4, cap: int = 4_000_000) -> int:
    """KataGo-style growing window: everything up to `base`, then sublinear.

    250k -> 250k, 1M -> ~530k, 4M -> ~1.18M, 10M -> ~2.24M (defaults).
    """
    if total <= base:
        size = int(total)
    else:
        size = int(base * (1 + beta * ((total / base) ** alpha - 1) / alpha))
    return min(size, cap)


@dataclass
class Chunk:
    """One complete game."""

    raw: list[bytes]
    flags: np.ndarray            # uint8 per sample
    plies_left: np.ndarray       # int32 per sample: len - 1 - i
    result_known: np.ndarray     # bool per sample (same for the whole game)
    stm: np.ndarray              # uint8 per sample
    value: np.ndarray            # int8 per sample (z from the stm perspective)
    source: str = ""             # file the game came from (window_start bookkeeping)
    offset: int = 0              # byte offset of the game's first sample in that file
    nbytes: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.nbytes:
            self.nbytes = sum(len(r) for r in self.raw)

    def __len__(self) -> int:
        return len(self.raw)

    @classmethod
    def from_samples(cls, samples: list[RawSample], source: str = "", offset: int = 0) -> "Chunk":
        n = len(samples)
        return cls(
            raw=[s.raw for s in samples],
            flags=np.array([s.flags for s in samples], dtype=np.uint8),
            plies_left=np.arange(n - 1, -1, -1, dtype=np.int32),
            result_known=np.array([s.result_known for s in samples], dtype=bool),
            stm=np.array([s.stm for s in samples], dtype=np.uint8),
            value=np.array([s.value for s in samples], dtype=np.int8),
            source=source,
            offset=offset,
        )


class ReplayBuffer:
    """Game chunks in arrival order; uniform sampling over the last `window`
    positions.

    Not thread-safe by itself.  `continuous.py` holds a lock only around the
    short structural calls (`add`, `draw_indices` + `locate`) and runs the
    expensive `collate` outside it — chunks are immutable once added, so a
    located (chunk, i) pair stays valid even if the chunk is evicted meanwhile."""

    def __init__(self, geom: BoardGeometry, z_lambda: float = 0.5, augment: bool = True,
                 value_gamma: float = 1.0, max_bytes: int = 0, decode_threads: int = 1) -> None:
        self._geom = geom
        self._z_lambda = z_lambda
        self._augment = augment
        self._value_gamma = value_gamma
        self._max_bytes = max_bytes
        self._chunks: deque[Chunk] = deque()
        self._total = 0
        self._bytes = 0
        self._cum: np.ndarray | None = None  # cumulative sample counts, rebuilt lazily
        self.bytes_capped = False            # set when max_bytes forced extra eviction
        self._pool = ThreadPoolExecutor(decode_threads) if decode_threads > 1 else None

    # ─── bookkeeping ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._total

    @property
    def bytes_used(self) -> int:
        return self._bytes

    @property
    def games(self) -> int:
        return len(self._chunks)

    def _pop_oldest(self) -> None:
        old = self._chunks.popleft()
        self._total -= len(old)
        self._bytes -= old.nbytes
        self._cum = None

    def add(self, chunk: Chunk, window: int) -> None:
        """Append a game, then evict the oldest games while the buffer would
        still hold at least `window` samples without them (never evicts the
        chunk just added).  `max_bytes` > 0 additionally evicts to fit memory
        and records `bytes_capped` for the caller to warn about."""
        if len(chunk) == 0:
            return
        self._chunks.append(chunk)
        self._total += len(chunk)
        self._bytes += chunk.nbytes
        self._cum = None
        while len(self._chunks) > 1 and self._total - len(self._chunks[0]) >= window:
            self._pop_oldest()
        if self._max_bytes > 0:
            while len(self._chunks) > 1 and self._bytes > self._max_bytes:
                self._pop_oldest()
                self.bytes_capped = True

    def window_starts(self) -> dict[str, int]:
        """Per source file: byte offset of the oldest game still held (plan §3.5)."""
        starts: dict[str, int] = {}
        for c in self._chunks:
            if c.source and (c.source not in starts or c.offset < starts[c.source]):
                starts[c.source] = c.offset
        return starts

    def _cumulative(self) -> np.ndarray:
        if self._cum is None:
            self._cum = np.cumsum([len(c) for c in self._chunks], dtype=np.int64)
        return self._cum

    def locate(self, flat: int) -> tuple[Chunk, int]:
        """Map a flat sample index in [0, len) to (chunk, index within chunk)."""
        cum = self._cumulative()
        ci = int(np.searchsorted(cum, flat, side="right"))
        prev = int(cum[ci - 1]) if ci > 0 else 0
        return self._chunks[ci], flat - prev

    def draw_indices(self, batch_size: int, window: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform flat indices over the last `window` samples."""
        if self._total == 0:
            raise ValueError("replay buffer is empty")
        start = max(0, self._total - max(1, window))
        return rng.integers(start, self._total, size=batch_size)

    # ─── decoding ───────────────────────────────────────────────────────────

    def decode_one(self, chunk: Chunk, i: int, sym: int):
        """Decode sample i of a chunk with D4 symmetry `sym` into numpy pieces."""
        geom = self._geom
        d = decode_sample(chunk.raw[i], geom, self._z_lambda, False,
                          self._value_gamma, int(chunk.plies_left[i]))

        # Opponent-reply target: the next sample of the same game (chunks are
        # whole games, so i+1 is always the opponent's position).
        aux_valid = False
        if i + 1 < len(chunk):
            aux_policy, aux_legal, aux_valid = decode_aux_policy(chunk.raw[i + 1], geom)
        else:
            aux_policy = np.zeros(geom.policy_size, dtype=np.float32)
            aux_legal = np.zeros(geom.policy_size, dtype=np.float32)

        planes = d.planes
        legal = d.legal_mask
        policy = d.policy
        corner = d.corner_class
        if sym > 0:
            planes = _apply_board_sym(planes, sym)
            perm = geom.policy_perms[sym].numpy()
            policy = policy[perm]
            legal = legal[perm]
            aux_policy = aux_policy[perm]
            aux_legal = aux_legal[perm]
            corner = geom.corner_perms[sym][corner]

        known = bool(chunk.result_known[i])
        return (
            planes, legal, policy, np.float32(d.value_target),
            np.float32(1.0 if d.policy_valid else 0.0),
            aux_policy, aux_legal.astype(bool), np.float32(1.0 if aux_valid else 0.0),
            np.int64(corner), np.float32(d.root_q),
            np.float32(chunk.plies_left[i] / MOVES_LEFT_SCALE),
            np.float32(1.0 if known else 0.0),
        )

    def sample(self, batch_size: int, window: int, rng: np.random.Generator):
        """A training batch in `compute_batch_loss` layout:
        (planes, legal_mask, pi_target, value_target, policy_valid, aux_policy,
         aux_legal, aux_valid, corner, root_q, moves_left_target, moves_left_valid).
        moves_left_valid doubles as the result-known indicator per sample."""
        idx = self.draw_indices(batch_size, window, rng)
        located = [self.locate(int(k)) for k in idx]
        syms = rng.integers(0, 8, size=batch_size) if self._augment else np.zeros(batch_size, dtype=np.int64)
        return self.collate(located, syms)

    def collate(self, located: list[tuple[Chunk, int]], syms) -> tuple[torch.Tensor, ...]:
        jobs = [(c, i, int(s)) for (c, i), s in zip(located, syms)]
        if self._pool is not None:
            rows = list(self._pool.map(lambda j: self.decode_one(*j), jobs))
        else:
            rows = [self.decode_one(*j) for j in jobs]
        cols = list(zip(*rows))
        planes = torch.from_numpy(np.stack(cols[0]))
        legal = torch.from_numpy(np.stack(cols[1]))
        policy = torch.from_numpy(np.stack(cols[2]))
        value = torch.from_numpy(np.asarray(cols[3], dtype=np.float32))
        policy_valid = torch.from_numpy(np.asarray(cols[4], dtype=np.float32))
        aux_policy = torch.from_numpy(np.stack(cols[5]))
        aux_legal = torch.from_numpy(np.stack(cols[6]))
        aux_valid = torch.from_numpy(np.asarray(cols[7], dtype=np.float32))
        corner = torch.from_numpy(np.asarray(cols[8], dtype=np.int64))
        root_q = torch.from_numpy(np.asarray(cols[9], dtype=np.float32))
        ml_target = torch.from_numpy(np.asarray(cols[10], dtype=np.float32))
        ml_valid = torch.from_numpy(np.asarray(cols[11], dtype=np.float32))
        return (planes, legal, policy, value, policy_valid, aux_policy, aux_legal,
                aux_valid, corner, root_q, ml_target, ml_valid)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
