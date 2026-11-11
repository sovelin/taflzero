"""Training batches prepared ahead in worker processes.

Decoding a batch is Python per sample (record, the opponent-reply target, the input
planes, the D4 symmetry): about 150 ms for 1024 samples on one core, while a small net's
step on the GPU takes about 35 ms, so the GPU mostly waited. Threads do not help (the
GIL; decode_threads made it slower). Here the main process still draws the samples
(the same rng calls in the same order as ReplayBuffer.sample / ContinuousRun.sample_batch)
and hands their payloads to a pool of processes; each decodes a whole batch into a slot
of shared memory, `workers` batches at once, while the GPU trains on an earlier one.

    pf = BatchPrefetcher(buffer, lock, rng, batch=1024, workers=6)
    batch = pf.next(window)   # tensors over shared memory, valid until the next call
    pf.close()

The tensors are views of a slot that is reused once next() is called again: train on a
batch before asking for the next one (the trainer moves it to the GPU in train_step).
"""
from __future__ import annotations

import multiprocessing as mp
import threading
from collections import deque
from multiprocessing import shared_memory

import numpy as np
import torch

from dataset import geometry, parse_planes
from replay import DecodeSettings, ReplayBuffer, decode_record, sample_payload

ALIGN = 64

# ─── worker side ────────────────────────────────────────────────────────────

_W: dict = {}


def _views(buf, layout: list[tuple[tuple[int, ...], str, int]]) -> list[np.ndarray]:
    return [np.ndarray(shape, dtype=np.dtype(dt), buffer=buf, offset=off) for shape, dt, off in layout]


def _init_worker(settings: DecodeSettings, layout, slot_names: list[str]) -> None:
    _W["geom"] = geometry(settings.board_size)
    _W["spec"] = parse_planes(settings.planes)
    _W["settings"] = settings
    _W["shms"] = [shared_memory.SharedMemory(name=n) for n in slot_names]
    _W["views"] = [_views(s.buf, layout) for s in _W["shms"]]


def _fill(slot: int, payloads: list[tuple], syms: np.ndarray) -> int:
    """Decode a batch into slot `slot` (row k of every column = sample k)."""
    geom, spec, s = _W["geom"], _W["spec"], _W["settings"]
    cols = _W["views"][slot]
    for k, (pl, sym) in enumerate(zip(payloads, syms)):
        row = decode_record(geom, spec, s, pl, int(sym))
        for c, v in enumerate(row):
            cols[c][k] = v
    return slot


# ─── main side ──────────────────────────────────────────────────────────────


def _layout_of(row: tuple, batch: int) -> tuple[list[tuple[tuple[int, ...], str, int]], int]:
    """Column shapes/dtypes of a batch as ReplayBuffer.collate builds it, and the slot size."""
    layout, off = [], 0
    for c, v in enumerate(row):
        a = np.asarray(v)
        dt = np.dtype(np.float32) if c == 13 else a.dtype  # collate casts the td rows to float32
        shape = (batch,) + a.shape
        layout.append((shape, dt.str, off))
        off += int(np.prod(shape, dtype=np.int64)) * dt.itemsize
        off = (off + ALIGN - 1) // ALIGN * ALIGN
    return layout, max(off, 1)


class BatchPrefetcher:
    def __init__(self, buffer: ReplayBuffer, lock: threading.Lock | None, rng: np.random.Generator,
                 batch: int, workers: int, depth: int | None = None) -> None:
        if workers < 1:
            raise ValueError("BatchPrefetcher needs at least one worker")
        self.buffer, self.lock, self.rng, self.batch = buffer, lock or threading.Lock(), rng, batch
        settings = buffer.decode_settings()
        with self.lock:
            if len(buffer) == 0:
                raise ValueError("BatchPrefetcher needs a non-empty buffer (the batch layout comes from a sample)")
            chunk, i = buffer.locate(0)
            row = buffer.decode_one(chunk, i, 0)
        self.layout, size = _layout_of(row, batch)
        self.depth = depth or workers + 2
        self.shms = [shared_memory.SharedMemory(create=True, size=size) for _ in range(self.depth)]
        self.views = [_views(s.buf, self.layout) for s in self.shms]
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(workers, initializer=_init_worker,
                             initargs=(settings, self.layout, [s.name for s in self.shms]))
        self.free: deque[int] = deque(range(self.depth))
        self.pending: deque = deque()
        self.lent: int | None = None  # the slot behind the batch last returned

    def _draw(self, window: int) -> tuple[list[tuple], np.ndarray]:
        # the same rng calls, in the same order, as ContinuousRun.sample_batch
        with self.lock:
            idx = self.buffer.draw_indices(self.batch, window, self.rng)
            located = [self.buffer.locate(int(k)) for k in idx]
            syms = self.rng.integers(0, 8, size=self.batch)
            payloads = [sample_payload(c, i) for c, i in located]
        return payloads, syms

    def next(self, window: int) -> tuple[torch.Tensor, ...]:
        """The next batch, drawn from `window` (batches already in flight were drawn from
        the window of their time). Its tensors live until the following call."""
        if self.lent is not None:
            self.free.append(self.lent)
            self.lent = None
        while self.free:
            slot = self.free.popleft()
            payloads, syms = self._draw(window)
            self.pending.append(self.pool.apply_async(_fill, (slot, payloads, syms)))
        slot = self.pending.popleft().get()
        self.lent = slot
        return tuple(torch.from_numpy(v) for v in self.views[slot])

    def decode(self, located: list[tuple], syms) -> tuple[torch.Tensor, ...]:
        """Decode given samples through a worker (tests: compare with buffer.collate).
        Uses a free slot; copies the result out."""
        if not self.free:
            raise RuntimeError("no free slot: call decode() before next()")
        slot = self.free[0]
        with self.lock:
            payloads = [sample_payload(c, i) for c, i in located]
        n = len(payloads)
        if n != self.batch:
            raise ValueError(f"decode() takes exactly a batch ({self.batch}) of samples, got {n}")
        self.pool.apply(_fill, (slot, payloads, np.asarray(syms)))
        return tuple(torch.from_numpy(v.copy()) for v in self.views[slot])

    def close(self) -> None:
        self.pool.terminate()
        self.pool.join()
        self.views = []
        for s in self.shms:
            s.close()
            s.unlink()
        self.shms = []
