"""Replay buffer and sliding window for the continuous trainer (plan §3.3).

Samples are stored as raw v2 bytes grouped per game (`Chunk`) and decoded only
when a batch is assembled, so the buffer costs ~700 B/sample (plus 144 B/sample of
side-file records when the data has them).  `sample()` returns exactly the 12-tuple
that `train.compute_batch_loss` unpacks; the input planes follow the buffer's plane
list (`planes`, dataset.parse_planes), extra planes stacked before the D4 symmetry.
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
    parse_planes,
    stack_planes,
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


# Self-play-only bytes of the side-file record (src/search/features.rs): the policy
# surprise of the sample's search and the FLAGS2 bits. Old files have 0 there.
FEAT_SURPRISE = 78
FEAT_FLAGS2 = 79
FEAT2_REDUCED = 1 << 0
SURPRISE_SCALE = 63.75
VALUE_SURPRISE_FLOOR = 0.01


def td_targets(root_q: np.ndarray, z_last: float, known: bool, horizons, unknown_as_draw: bool) -> np.ndarray:
    """Short-term value targets of one game (docs/plan-T9.md §3), (n, H) float32.

    For horizon h, lambda = 1/h, from the side to move of each sample (it alternates
    every ply):  v[last] = lambda q[last] + (1 - lambda) z[last];
                 v[i]    = lambda q[i]    + (1 - lambda) (-v[i+1]).
    A game with an unknown result is NaN (masked) unless `unknown_as_draw`, then z = 0.
    """
    n, hs = len(root_q), list(horizons)
    out = np.full((n, len(hs)), np.nan, dtype=np.float32)
    if n == 0 or not hs or (not known and not unknown_as_draw):
        return out
    z = float(z_last) if known else 0.0
    q = np.asarray(root_q, dtype=np.float64)
    # Unrolled: with c = -(1 - lambda), v[i] = sum_{k=i}^{n-2} c^(k-i) lambda q[k] + c^(n-1-i) v[last].
    i = np.arange(n)
    gap = i[None, :] - i[:, None]                    # k - i
    for j, h in enumerate(hs):
        lam = 1.0 / float(h)
        c = -(1.0 - lam)
        v_last = lam * q[-1] + (1.0 - lam) * z
        pw = np.where(gap >= 0, c ** np.maximum(gap, 0), 0.0)[:, : n - 1]   # k in [i, n-2]
        out[:, j] = pw @ (lam * q[: n - 1]) + c ** (n - 1 - i) * v_last
    return out


def sample_weights(surprise: np.ndarray, root_q: np.ndarray, z: np.ndarray, known: bool,
                   reduced: np.ndarray, a: float, b: float, reduced_weight: float,
                   unknown_as_draw: bool, full: np.ndarray | None = None) -> np.ndarray:
    """Sampling weight per sample of one game (docs/plan-T9.md §6), mean 1 over the game.

    w = (1 - a - b) + a * ps / mean(ps) + b * vs / mean(vs), times `reduced_weight` on
    reduced samples; a term whose game mean is 0 contributes its share uniformly.
    ps is the policy surprise in nats, vs = min(1, (z - q)^2 / 4) the value surprise
    (z = 0 for an unknown result when `unknown_as_draw`, else no value term); when the
    game's mean vs is below VALUE_SURPRISE_FLOOR the value share b shrinks in proportion.

    Only full searches have a policy surprise (`full`; all samples when None): its mean
    is over them and the other samples get the uniform share `a`, so the policy term
    reorders weight among full searches without moving weight from cheap to full ones
    (KataGo writes no cheap rows, so its mean is over policy rows too).
    """
    n = len(root_q)
    w = np.full(n, 1.0 - a - b, dtype=np.float64)
    ps = np.asarray(surprise, dtype=np.float64)
    full = np.ones(n, dtype=bool) if full is None else np.asarray(full, dtype=bool)
    mean_ps = ps[full].mean() if full.any() else 0.0
    if a > 0 and mean_ps > 0:
        w += np.where(full, a * ps / mean_ps, a)
    else:
        w += a
    if b > 0 and (known or unknown_as_draw):
        zz = np.asarray(z, dtype=np.float64) if known else np.zeros(n)
        vs = np.minimum(1.0, (zz - np.asarray(root_q, dtype=np.float64)) ** 2 / 4.0)
        # KataGo's guard (play.cpp, valueSurpriseDataWeight): a game with almost no value
        # surprise (a draw where every root value sits near 0) would hand its whole value
        # share to one barely different position; below a mean of VALUE_SURPRISE_FLOOR the
        # share shrinks in proportion and the rest is spread uniformly.
        m = vs.mean()
        b_eff = b * min(1.0, m / VALUE_SURPRISE_FLOOR)
        w += (b_eff * vs / m if m > 0 else 0.0) + (b - b_eff)
    else:
        w += b
    w = np.where(np.asarray(reduced, dtype=bool), w * reduced_weight, w)
    total = w.sum()
    return (w * n / total) if total > 0 else np.ones(n)


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
    # Side-file records of the game's samples, (len, 144) uint8; None when the data
    # has no side file (only the base planes can be trained on it).
    feat: np.ndarray | None = None
    first_sample: int = -1       # sample index of the game's first sample in its file
    root_q: np.ndarray | None = None   # int8 per sample (search value x 127, stm view)
    # Filled by ReplayBuffer.add when the buffer uses them (T9):
    td: np.ndarray | None = None       # (len, H) short-term value targets, NaN = masked
    wcum: np.ndarray | None = None     # cumulative sampling weights, float64 (len,)

    def __post_init__(self) -> None:
        if not self.nbytes:
            self.nbytes = sum(len(r) for r in self.raw)
            if self.feat is not None:
                self.nbytes += self.feat.nbytes

    @property
    def nbytes_raw(self) -> int:
        """Bytes the game takes in its sample file."""
        return sum(len(r) for r in self.raw)

    def attach_feat(self, records: np.ndarray) -> None:
        assert records.shape[0] == len(self.raw), "one record per sample"
        self.feat = records
        self.nbytes += records.nbytes

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
            root_q=np.array([s.root_q for s in samples], dtype=np.int8),
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
                 value_gamma: float = 1.0, max_bytes: int = 0, decode_threads: int = 1,
                 planes: str = "base", td_horizons=(), unknown_as_draw: bool = False,
                 surprise_policy: float = 0.0, surprise_value: float = 0.0,
                 reduced_weight: float = 1.0, extended_batch: bool = False) -> None:
        """T9 options (docs/plan-T9.md), all off by default: `td_horizons` fills the
        short-term value targets, `unknown_as_draw` counts ply-limit games as draws
        for them (WDL runs), `surprise_*` / `reduced_weight` make sampling weighted,
        `extended_batch` adds z and the td targets to every batch (14 elements)."""
        self._geom = geom
        self._spec = parse_planes(planes)
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
        self._td_horizons = tuple(int(h) for h in td_horizons)
        self._unknown_as_draw = bool(unknown_as_draw)
        self._surprise_policy = float(surprise_policy)
        self._surprise_value = float(surprise_value)
        self._reduced_weight = float(reduced_weight)
        self._weighted = self._surprise_policy > 0 or self._surprise_value > 0 or self._reduced_weight != 1.0
        self._extended = bool(extended_batch)
        self._wtot: np.ndarray | None = None  # cumulative chunk weight sums, rebuilt lazily
        self._reset_weight_stats()

    def _reset_weight_stats(self) -> None:
        self._st = {"n": 0, "full": 0, "ps": 0.0, "reduced": 0, "w_max": 0.0}

    def take_weight_stats(self) -> dict:
        """Since the last call, over the games added: mean policy surprise of full
        searches (nats), share of reduced samples, largest sampling weight."""
        st = self._st
        out = {   # None when no game (or no full search) arrived since the last call
            "surprise": st["ps"] / st["full"] if st["full"] else None,
            "reduced": st["reduced"] / st["n"] if st["n"] else None,
            "w_max": st["w_max"] if st["n"] else None,
        }
        self._reset_weight_stats()
        return out

    def _prepare(self, chunk: Chunk) -> None:
        """T9 per-game arrays: td targets and sampling weights."""
        n = len(chunk)
        if chunk.feat is not None:
            surprise = chunk.feat[:, FEAT_SURPRISE].astype(np.float64) / SURPRISE_SCALE
            reduced = (chunk.feat[:, FEAT_FLAGS2] & FEAT2_REDUCED) != 0
        else:
            surprise = np.zeros(n)
            reduced = np.zeros(n, dtype=bool)
        q = (chunk.root_q.astype(np.float64) / 127.0) if chunk.root_q is not None else np.zeros(n)
        known = bool(chunk.result_known[0]) if n else True
        if self._td_horizons:
            chunk.td = td_targets(q, float(chunk.value[-1]), known, self._td_horizons, self._unknown_as_draw)
        w = None
        if self._weighted:
            w = sample_weights(surprise, q, chunk.value.astype(np.float64), known, reduced,
                               self._surprise_policy, self._surprise_value, self._reduced_weight,
                               self._unknown_as_draw, full=(chunk.flags & 1) != 0)
            chunk.wcum = np.cumsum(w)
        # the arrays above live as long as the chunk: count them against max_bytes
        chunk.nbytes += sum(x.nbytes for x in (chunk.td, chunk.wcum) if x is not None)
        full = (chunk.flags & 1) != 0
        st = self._st
        st["n"] += n
        st["full"] += int(full.sum())
        st["ps"] += float(surprise[full].sum())
        st["reduced"] += int(reduced.sum())
        if w is not None and n:
            st["w_max"] = max(st["w_max"], float(w.max()))

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
        self._wtot = None

    def add(self, chunk: Chunk, window: int) -> None:
        """Append a game, then evict the oldest games while the buffer would
        still hold at least `window` samples without them (never evicts the
        chunk just added).  `max_bytes` > 0 additionally evicts to fit memory
        and records `bytes_capped` for the caller to warn about."""
        if len(chunk) == 0:
            return
        if self._spec.needs_features and chunk.feat is None:
            raise ValueError(
                f"planes {self._spec.canonical!r} need the feature side file, but the game at "
                f"{chunk.source}:{chunk.offset} has none (data from an engine without .feat files)")
        if self._td_horizons or self._weighted or self._extended or chunk.feat is not None:
            self._prepare(chunk)
        self._chunks.append(chunk)
        self._total += len(chunk)
        self._bytes += chunk.nbytes
        self._cum = None
        self._wtot = None
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

    def window_start_samples(self) -> dict[str, int]:
        """Per source file: sample index of the oldest game still held (side-file
        record index on resume); files whose games carry no index are left out."""
        starts: dict[str, int] = {}
        for c in self._chunks:
            if c.source and c.first_sample >= 0 and (
                    c.source not in starts or c.first_sample < starts[c.source]):
                starts[c.source] = c.first_sample
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
        """Flat indices over the last `window` samples: uniform, or proportional to the
        per-sample weights when the buffer is weighted (T9)."""
        if self._total == 0:
            raise ValueError("replay buffer is empty")
        start = max(0, self._total - max(1, window))
        if not self._weighted:
            return rng.integers(start, self._total, size=batch_size)
        return self._draw_weighted(batch_size, start, rng)

    def _draw_weighted(self, batch_size: int, start: int, rng: np.random.Generator) -> np.ndarray:
        cum = self._cumulative()
        if self._wtot is None:
            self._wtot = np.cumsum([c.wcum[-1] for c in self._chunks], dtype=np.float64)
        wtot = self._wtot
        # The window starts inside chunk c0 at sample i0: only its weight from i0 on counts.
        c0 = int(np.searchsorted(cum, start, side="right"))
        i0 = start - (int(cum[c0 - 1]) if c0 > 0 else 0)
        first = self._chunks[c0].wcum
        skip = float(first[i0 - 1]) if i0 > 0 else 0.0
        base = (float(wtot[c0 - 1]) if c0 > 0 else 0.0) + skip
        u = base + rng.random(batch_size) * (float(wtot[-1]) - base)
        out = np.empty(batch_size, dtype=np.int64)
        cis = np.minimum(np.searchsorted(wtot, u, side="right"), len(self._chunks) - 1)
        for k, (ci, uk) in enumerate(zip(cis, u)):
            ci = int(ci)
            within = uk - (float(wtot[ci - 1]) if ci > 0 else 0.0)
            wc = self._chunks[ci].wcum
            i = min(int(np.searchsorted(wc, within, side="right")), len(wc) - 1)
            if ci == c0:
                i = max(i, i0)
            out[k] = (int(cum[ci - 1]) if ci > 0 else 0) + i
        return out

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
        if not self._spec.is_base:
            record = chunk.feat[i] if chunk.feat is not None else None
            planes = stack_planes(planes, self._spec, record, geom)
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
        row = (
            planes, legal, policy, np.float32(d.value_target),
            np.float32(1.0 if d.policy_valid else 0.0),
            aux_policy, aux_legal.astype(bool), np.float32(1.0 if aux_valid else 0.0),
            np.int64(corner), np.float32(d.root_q),
            np.float32(chunk.plies_left[i] / MOVES_LEFT_SCALE),
            np.float32(1.0 if known else 0.0),
        )
        if not self._extended:
            return row
        z = np.float32(chunk.value[i] if known else 0.0)
        td = chunk.td[i] if chunk.td is not None else np.zeros(0, dtype=np.float32)
        return row + (z, td)

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
        out = (planes, legal, policy, value, policy_valid, aux_policy, aux_legal,
               aux_valid, corner, root_q, ml_target, ml_valid)
        if not self._extended:
            return out
        z = torch.from_numpy(np.asarray(cols[12], dtype=np.float32))
        td = torch.from_numpy(np.stack(cols[13]).astype(np.float32))
        return out + (z, td)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
