"""Batches decoded in worker processes (prefetch.py) are the batches the trainer built
in-process: bit for bit, and in the same order for the same rng."""
from __future__ import annotations

import threading

import numpy as np
import torch

from prefetch import BatchPrefetcher
from replay import ReplayBuffer
from test_continuous import GEOM, _chunk


def _buffer() -> ReplayBuffer:
    buf = ReplayBuffer(GEOM, augment=True, td_horizons=(4, 12), extended_batch=True)
    for g in range(6):
        buf.add(_chunk(9 + g, offset=1000 * g, winner="atk" if g % 2 else "def"), window=10_000)
    return buf


def _same(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> None:
    assert len(a) == len(b)
    for k, (x, y) in enumerate(zip(a, b)):
        assert x.dtype == y.dtype and x.shape == y.shape, (k, x.dtype, y.dtype, x.shape, y.shape)
        assert torch.equal(x, y), k


def test_a_worker_decodes_the_same_batch_as_collate() -> None:
    buf = _buffer()
    rng = np.random.default_rng(7)
    n = 16
    located = [buf.locate(int(k)) for k in rng.integers(0, len(buf), size=n)]
    syms = rng.integers(0, 8, size=n)  # every symmetry, not only the identity
    pf = BatchPrefetcher(buf, None, np.random.default_rng(0), batch=n, workers=1)
    try:
        _same(pf.decode(located, syms), buf.collate(located, syms))
    finally:
        pf.close()


def test_next_gives_the_batches_of_sample_batch_for_the_same_rng() -> None:
    buf = _buffer()
    window, n = len(buf), 8

    def sample_batch(rng: np.random.Generator):  # ContinuousRun.sample_batch
        idx = buf.draw_indices(n, window, rng)
        located = [buf.locate(int(k)) for k in idx]
        syms = rng.integers(0, 8, size=n)
        return buf.collate(located, syms)

    ref_rng = np.random.default_rng(42)
    expected = [sample_batch(ref_rng) for _ in range(5)]
    pf = BatchPrefetcher(buf, threading.Lock(), np.random.default_rng(42), batch=n, workers=2, depth=3)
    try:
        for want in expected:
            got = tuple(t.clone() for t in pf.next(window))  # the slot is reused after the next call
            _same(got, want)
    finally:
        pf.close()
