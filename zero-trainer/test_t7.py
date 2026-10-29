"""Input planes in the continuous loop and the ONNX export (docs/plan-T7-planes.md,
items 9-12). Data: the two-game fixture zero-trainer/testdata/features.{bin,feat}
(3 + 9 samples), which is what a worker writes as worker<N>.bin / worker<N>.feat."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from az_micro_net import TaflAlphaZeroNet
from dataset import FEAT_HEADER_BYTES, FEAT_RECORD_BYTES, decode_sample, geometry, parse_planes, stack_planes
from ingest import Ingester, samples_before
from replay import ReplayBuffer

GEOM = geometry(11)
HERE = Path(__file__).resolve().parent
FIX_BIN = HERE / "testdata" / "features.bin"
FIX_FEAT = HERE / "testdata" / "features.feat"
FULL = "base,last2,rep_dst,king12,ply,cap,mob"


def _records() -> np.ndarray:
    raw = FIX_FEAT.read_bytes()[FEAT_HEADER_BYTES:]
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, FEAT_RECORD_BYTES)


def _data_dir(tmp_path: Path, feat_records: int | None) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(FIX_BIN, data / "worker0.bin")
    if feat_records is not None:
        raw = FIX_FEAT.read_bytes()
        (data / "worker0.feat").write_bytes(raw[:FEAT_HEADER_BYTES + feat_records * FEAT_RECORD_BYTES])
    return data


# ─── ingest ──────────────────────────────────────────────────────────────────


def test_ingest_waits_for_the_side_file_and_attaches_records(tmp_path: Path) -> None:
    data = _data_dir(tmp_path, feat_records=3 + 4)  # game 2's records half written
    ing = Ingester(data, GEOM)
    first = ing.poll()
    assert [len(c) for c in first.chunks] == [3]
    assert first.chunks[0].first_sample == 0
    assert np.array_equal(first.chunks[0].feat, _records()[:3])
    # The .feat catches up: game 2 comes in with its records.
    shutil.copy(FIX_FEAT, data / "worker0.feat")
    second = ing.poll()
    assert [len(c) for c in second.chunks] == [9]
    assert second.chunks[0].first_sample == 3
    assert np.array_equal(second.chunks[0].feat, _records()[3:])
    assert ing.sample_offsets == {"worker0.bin": 12}
    assert ing.offsets["worker0.bin"] == FIX_BIN.stat().st_size
    assert ing.poll().chunks == []


def test_ingest_without_side_file_is_base_only(tmp_path: Path) -> None:
    data = _data_dir(tmp_path, feat_records=None)
    res = Ingester(data, GEOM).poll()
    assert [len(c) for c in res.chunks] == [3, 9]
    assert all(c.feat is None for c in res.chunks)
    buf = ReplayBuffer(GEOM, augment=False, planes=FULL)
    with pytest.raises(ValueError, match="side file"):
        buf.add(res.chunks[0], 100)
    base = ReplayBuffer(GEOM, augment=False)
    for c in res.chunks:
        base.add(c, 100)
    assert len(base) == 12


def test_resumed_state_without_sample_counts_recounts(tmp_path: Path) -> None:
    data = _data_dir(tmp_path, feat_records=12)
    full = Ingester(data, GEOM).poll()
    game2_offset = full.chunks[1].offset
    assert samples_before(data / "worker0.bin", GEOM, game2_offset) == 3
    # A state from before sample counting: byte offset only.
    ing = Ingester(data, GEOM, offsets={"worker0.bin": game2_offset})
    res = ing.poll()
    assert [c.first_sample for c in res.chunks] == [3]
    assert np.array_equal(res.chunks[0].feat, _records()[3:])


# ─── replay ──────────────────────────────────────────────────────────────────


def test_replay_batches_stack_the_planes_before_the_symmetry(tmp_path: Path) -> None:
    data = _data_dir(tmp_path, feat_records=12)
    chunks = Ingester(data, GEOM).poll().chunks
    spec = parse_planes(FULL)
    buf = ReplayBuffer(GEOM, augment=False, planes=FULL)
    for c in chunks:
        buf.add(c, 100)
    located = [(chunks[1], i) for i in range(9)]
    batch = buf.collate(located, [0] * 9)
    assert batch[0].shape == (9, spec.num_planes, 11, 11)
    for i in range(9):
        want = stack_planes(decode_sample(chunks[1].raw[i], GEOM).planes, spec, chunks[1].feat[i], GEOM)
        assert np.array_equal(batch[0][i].numpy(), want)
    # Symmetry 3 rotates the whole stack: the last-move planes move with the pieces.
    rot = buf.collate(located, [3] * 9)
    from dataset import _apply_board_sym
    assert np.array_equal(rot[0][4].numpy(), _apply_board_sym(batch[0][4].numpy(), 3))
    assert buf.window_start_samples() == {"worker0.bin": 0}


def test_cheap_value_weight_reaches_the_loss() -> None:
    from train import compute_batch_loss

    torch.manual_seed(0)
    model = TaflAlphaZeroNet(trunk_channels=8, num_blocks=1)
    model.eval()
    buf = ReplayBuffer(GEOM, augment=False)
    from ingest import group_games
    from dataset import parse_samples
    samples, _ = parse_samples(FIX_BIN.read_bytes(), GEOM)
    chunks, _ = group_games(samples)
    for c in chunks:
        buf.add(c, 100)
    batch = list(buf.collate([(chunks[1], i) for i in range(9)], [0] * 9))
    batch[4] = torch.tensor([1.0, 0.0] * 4 + [1.0])
    dev = torch.device("cpu")
    with torch.no_grad():
        zero = compute_batch_loss(model, batch, dev, 1.0, 0.0, 0.0, False, cheap_value_weight=0.0)[2]
        full_only = compute_batch_loss(model, [t[batch[4] > 0.5] for t in batch], dev, 1.0, 0.0, 0.0,
                                       False)[2]
    assert torch.allclose(zero, full_only)


# ─── export ──────────────────────────────────────────────────────────────────


def test_export_writes_planes_metadata_and_fp16_keeps_it(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from export_onnx import export_model_to_onnx, get_planes_metadata

    spec = parse_planes(FULL)
    model = TaflAlphaZeroNet(in_channels=spec.num_planes, planes=FULL, trunk_channels=8, num_blocks=1)
    out = tmp_path / "net.onnx"
    export_model_to_onnx(model, out)  # fp16 + .fp32.onnx
    for path in (out, out.with_suffix(".fp32.onnx")):
        m = onnx.load(str(path))
        assert get_planes_metadata(m) == spec.canonical
        assert m.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 24
    # An old checkpoint without a plane list exports as base; a wider one cannot.
    old = TaflAlphaZeroNet(trunk_channels=8, num_blocks=1)
    export_model_to_onnx(old, tmp_path / "old.onnx", fp16=False)
    assert get_planes_metadata(onnx.load(str(tmp_path / "old.onnx"))) == "base"
    wide = TaflAlphaZeroNet(in_channels=13, trunk_channels=8, num_blocks=1)
    with pytest.raises(ValueError):
        export_model_to_onnx(wide, tmp_path / "wide.onnx", fp16=False)
    with pytest.raises(ValueError):
        TaflAlphaZeroNet(in_channels=11, planes=FULL)


def test_qnxx_round_trip_keeps_the_plane_list(tmp_path: Path) -> None:
    from qnxx_io import load_qnxx, save_qnxx

    model = TaflAlphaZeroNet(in_channels=13, planes="base,rep_dst", trunk_channels=8, num_blocks=1)
    save_qnxx(model, tmp_path / "m.qnxx")
    again = load_qnxx(tmp_path / "m.qnxx")
    assert again.model_kwargs["planes"] == "base,rep_dst"


# ─── continuous config ───────────────────────────────────────────────────────


def test_config_defaults_and_old_runs_resume(tmp_path: Path) -> None:
    from continuous import parse_config

    args = parse_config(["--run-dir", str(tmp_path / "r"), "--workers", "0"])
    assert args.planes == "base" and args.cheap_value_weight == 1.0
    args = parse_config(["--run-dir", str(tmp_path / "r"), "--workers", "0",
                         "--planes", "last2,base,-throne"])
    assert args.planes == "base,-throne,last2"
    # An old run's config.json: no planes / cheap_value_weight, and record-only keys.
    old = tmp_path / "old"
    old.mkdir()
    (old / "config.json").write_text(json.dumps({
        "workers": 0, "batch": 64, "engine_commit": "2793e76",
        "engine_sha256": "ab" * 32, "engine_version": "taflzero 0.2.0 abc",
    }), encoding="utf-8")
    args = parse_config(["--resume", str(old)])
    assert args.planes == "base" and args.cheap_value_weight == 1.0 and args.batch == 64


def test_engine_identity_of_a_file(tmp_path: Path) -> None:
    import hashlib
    from continuous import engine_identity

    fake = tmp_path / "engine.bin"
    fake.write_bytes(b"not an engine")
    ident = engine_identity(fake)
    assert ident["engine_sha256"] == hashlib.sha256(b"not an engine").hexdigest()
    assert ident["engine_version"] == "unknown"
    assert engine_identity(None) == {"engine_sha256": None, "engine_version": None}
