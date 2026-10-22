"""Plane lists and the feature side file (docs/probe-input-planes.md, part 2).

Run from the repo root:  python -m pytest zero-trainer -q
The fixture zero-trainer/testdata/features.{bin,feat} is written by the Rust test
`feature_dump::tests::python_fixture_matches_the_tool`: two Copenhagen games from the
start position (a three-ply undecided one, then a nine-ply back-and-forth that ends in
a threefold repetition), and the side file `taflzero --features` makes from them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from az_micro_net import TaflAlphaZeroNet
from dataset import (
    BASE_PLANES,
    FEAT_FLAG_LAST_PLY,
    FEAT_FLAG_THREEFOLD,
    FEAT_FLAGS,
    FEAT_HIST,
    FEAT_PLIES_TO_END,
    FEAT_PLY,
    FEAT_REP2,
    FEAT_REP3,
    NUM_PLANES,
    SelfPlayDataset,
    _SYM_POS,
    _apply_board_sym,
    decode_sample,
    extra_planes,
    geometry,
    open_features,
    parse_planes,
    parse_samples,
)

GEOM = geometry(11)
HERE = Path(__file__).resolve().parent
FIX_BIN = HERE / "testdata" / "features.bin"
FIX_FEAT = HERE / "testdata" / "features.feat"


def _samples():
    return parse_samples(FIX_BIN.read_bytes(), GEOM)[0]


def _squares(packed: bytes) -> set[int]:
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")[: GEOM.sqs]
    return set(np.flatnonzero(bits).tolist())


def _move_between(a: bytes, b: bytes) -> tuple[int, int]:
    """The move from sample a to sample b, read off the piece planes alone."""
    stm = a[48]
    occ = lambda s: _squares(s[0:16]) | _squares(s[16:32]) | _squares(s[32:48])
    mover = (lambda s: _squares(s[0:16])) if stm == 0 else (lambda s: _squares(s[16:32]) | _squares(s[32:48]))
    (frm,) = mover(a) - occ(b)
    (to,) = mover(b) - occ(a)
    return frm, to


# ─── plane lists ─────────────────────────────────────────────────────────────


def test_plane_list_parsing() -> None:
    assert parse_planes("base").num_planes == NUM_PLANES == len(BASE_PLANES)
    assert parse_planes("base").is_base
    spec = parse_planes("base,-throne,-corners,-edges")
    assert spec.num_planes == 8 and not spec.is_base
    assert [BASE_PLANES[i] for i in spec.base] == [
        "attackers", "defenders", "king", "stm", "group_bfs", "king_bfs", "rep1", "rep2"]
    assert parse_planes("base,-rep").num_planes == 9
    assert parse_planes("-throne,base").base == parse_planes("base,-throne").base
    assert parse_planes("last2").num_planes == 4 and parse_planes("last2").base == ()
    assert parse_planes("base,ones").num_planes == 12 and not parse_planes("base,ones").needs_features
    assert parse_planes("base,-stm").index_of("stm") is None
    assert parse_planes("base,-throne").index_of("stm") == 3
    # The probe's variants and their plane counts (spec table).
    counts = {
        "base": 11, "base,-throne,-corners,-edges": 8, "base,last2": 15, "base,rep_dst": 13,
        "base,king12": 13, "base,last2,rep_dst,king12,ply": 20,
        "base,-throne,-corners,-edges,last2,rep_dst,king12,ply": 17, "base,cap,mob": 15,
        "base,last2,rep_dst,king12,ply,cap,mob": 24, "base,last4": 19,
    }
    for planes, n in counts.items():
        assert parse_planes(planes).num_planes == n, planes


@pytest.mark.parametrize("bad", [
    "base,base", "base,last2,last2", "base,-throne,-throne", "base,last1,last2",
    "base,history", "base,-king", "-throne", "", "base,", "base,-rep,-rep",
])
def test_plane_list_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_planes(bad)


def test_feature_planes_need_the_side_file() -> None:
    with pytest.raises(ValueError, match="side file"):
        SelfPlayDataset(FIX_BIN, planes="base,last2")
    # `ones` comes from nowhere: fine without one.
    ds = SelfPlayDataset(FIX_BIN, planes="base,ones", augment=False)
    planes = ds[0][0]
    assert planes.shape == (12, 11, 11) and torch.all(planes[11] == 1)


def test_side_file_must_match_the_sample_count(tmp_path: Path) -> None:
    short = tmp_path / "short.bin"
    raw = FIX_BIN.read_bytes()
    first_game = sum(len(s.raw) for s in _samples()[:3])
    short.write_bytes(raw[:first_game])
    with pytest.raises(ValueError, match="12 records"):
        SelfPlayDataset(short, planes="base,last2", features=FIX_FEAT)
    # A data file that grew past the side file is read up to the side file's end.
    longer = tmp_path / "longer.bin"
    longer.write_bytes(raw + raw[:first_game])
    with pytest.raises(ValueError):
        SelfPlayDataset(longer, planes="base,last2", features=FIX_FEAT)
    ds = SelfPlayDataset(longer, planes="base,last2", features=FIX_FEAT, limit=12)
    assert len(ds) == 12


# ─── base is unchanged ───────────────────────────────────────────────────────


def test_base_planes_bit_for_bit_unchanged() -> None:
    samples = _samples()
    plain = SelfPlayDataset(FIX_BIN, augment=False)
    listed = SelfPlayDataset(FIX_BIN, augment=False, planes="base", features=FIX_FEAT)
    for i, s in enumerate(samples):
        ref = decode_sample(s.raw, GEOM).planes
        a, b = plain[i], listed[i]
        assert a[0].dtype == torch.float32 and a[0].shape == (NUM_PLANES, 11, 11)
        assert np.array_equal(a[0].numpy(), ref)
        for x, y in zip(a, b):
            assert torch.equal(x, y)
    # Same with augmentation: the symmetry draw is the only randomness.
    plain_aug = SelfPlayDataset(FIX_BIN, augment=True)
    listed_aug = SelfPlayDataset(FIX_BIN, augment=True, planes="base", features=FIX_FEAT)
    for i in range(len(samples)):
        np.random.seed(i)
        a = plain_aug[i]
        np.random.seed(i)
        b = listed_aug[i]
        for x, y in zip(a, b):
            assert torch.equal(x, y)


# ─── the side file's planes ──────────────────────────────────────────────────


def test_fixture_records_describe_the_games() -> None:
    feat = open_features(FIX_FEAT)
    samples = _samples()
    assert feat.shape == (12, 144) and len(samples) == 12
    plies = [int(r[FEAT_PLY]) | int(r[FEAT_PLY + 1]) << 8 for r in feat]
    to_end = [int(r[FEAT_PLIES_TO_END]) | int(r[FEAT_PLIES_TO_END + 1]) << 8 for r in feat]
    assert plies == [0, 1, 2] + list(range(9))
    assert to_end == [3, 2, 1] + list(range(9, 0, -1))
    assert [bool(r[FEAT_FLAGS] & FEAT_FLAG_LAST_PLY) for r in feat] == [
        False, False, True] + [False] * 8 + [True]
    assert [bool(r[FEAT_FLAGS] & FEAT_FLAG_THREEFOLD) for r in feat] == [False] * 3 + [True] * 9
    # hist[0] of every non-first sample is the move between it and the previous sample.
    for i in range(1, 12):
        if plies[i] == 0:
            assert tuple(feat[i][FEAT_HIST:FEAT_HIST + 2]) == (255, 255)
        else:
            assert tuple(feat[i][FEAT_HIST:FEAT_HIST + 2]) == _move_between(samples[i - 1].raw, samples[i].raw)


def test_last2_planes_mark_the_last_two_moves() -> None:
    samples = _samples()
    ds = SelfPlayDataset(FIX_BIN, augment=False, planes="base,last2", features=FIX_FEAT)
    assert ds.num_planes == 15
    # Sample 2 of the first game: two plies were played before it.
    planes = ds[2][0].numpy()
    assert np.array_equal(planes[:11], decode_sample(samples[2].raw, GEOM).planes)
    opp = _move_between(samples[1].raw, samples[2].raw)
    ours = _move_between(samples[0].raw, samples[1].raw)
    for k, sq in enumerate([opp[0], opp[1], ours[0], ours[1]]):
        plane = planes[11 + k]
        assert plane.sum() == 1.0, k
        assert plane[sq // 11, sq % 11] == 1.0, k
    # Sample 1 has only the opponent's move; sample 0 none.
    p1 = ds[1][0].numpy()
    assert p1[11].sum() == 1 and p1[12].sum() == 1 and p1[13].sum() == 0 and p1[14].sum() == 0
    assert ds[0][0].numpy()[11:].sum() == 0


def test_rep_dst_planes_match_the_bitmasks() -> None:
    feat = open_features(FIX_FEAT)
    ds = SelfPlayDataset(FIX_BIN, augment=False, planes="base,rep_dst", features=FIX_FEAT)
    seen_rep2 = seen_rep3 = 0
    for i in range(len(ds)):
        planes = ds[i][0].numpy()
        rep3 = _squares(bytes(feat[i][FEAT_REP3:FEAT_REP3 + 16]))
        rep2 = _squares(bytes(feat[i][FEAT_REP2:FEAT_REP2 + 16]))
        assert set(np.flatnonzero(planes[11].ravel()).tolist()) == rep3
        assert set(np.flatnonzero(planes[12].ravel()).tolist()) == rep2
        seen_rep2 += len(rep2)
        seen_rep3 += len(rep3)
    # The back-and-forth game: four samples with one rep2 square, then one rep3 square.
    assert (seen_rep2, seen_rep3) == (4, 1)


def test_symmetry_moves_last1_to_the_rotated_square() -> None:
    samples = _samples()
    ds = SelfPlayDataset(FIX_BIN, augment=False, planes="base,last1", features=FIX_FEAT)
    planes = ds[1][0].numpy()
    frm, to = _move_between(samples[0].raw, samples[1].raw)
    rotated = _apply_board_sym(planes, 3)
    last = GEOM.board_size - 1
    for k, sq in enumerate([frm, to]):
        r, c = _SYM_POS[3](sq // 11, sq % 11, last)
        assert rotated[11 + k].sum() == 1.0
        assert rotated[11 + k][r, c] == 1.0
    # The base planes rotate the same way: the moved attacker's destination is in both.
    r, c = _SYM_POS[3](to // 11, to % 11, last)
    assert rotated[0][r, c] == 1.0


def test_ply_plane_is_a_scaled_constant() -> None:
    feat = open_features(FIX_FEAT)
    p = extra_planes(feat[5], ["ply"], GEOM)
    assert p.shape == (1, 11, 11) and np.allclose(p, 2 / 400)


# ─── model and trainer ───────────────────────────────────────────────────────


def test_in_channels_follow_the_plane_count() -> None:
    for planes in ["base", "base,-throne,-corners,-edges", "base,last2,rep_dst,king12,ply,cap,mob"]:
        ds = SelfPlayDataset(FIX_BIN, augment=True, planes=planes, features=FIX_FEAT)
        model = TaflAlphaZeroNet(in_channels=ds.num_planes, trunk_channels=8, num_blocks=1)
        assert model.model_kwargs["in_channels"] == parse_planes(planes).num_planes
        x = torch.stack([ds[i][0] for i in range(4)])
        assert x.shape[1] == ds.num_planes
        policy, value = model(x)
        assert policy.shape == (4, 4840) and value.shape == (4, 1)


def test_base_onnx_export_still_works(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from export_onnx import export_model_to_onnx

    ds = SelfPlayDataset(FIX_BIN, planes="base")
    model = TaflAlphaZeroNet(in_channels=ds.num_planes, trunk_channels=8, num_blocks=1)
    out = tmp_path / "base.onnx"
    export_model_to_onnx(model, out, fp16=False)
    graph = onnx.load(str(out))
    onnx.checker.check_model(graph)
    dims = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
    assert dims[1:] == [11, 11, 11]


def test_cheap_value_weight_zero_drops_cheap_samples_from_value_loss() -> None:
    from train import compute_batch_loss

    torch.manual_seed(0)
    model = TaflAlphaZeroNet(trunk_channels=8, num_blocks=1)
    model.eval()
    ds = SelfPlayDataset(FIX_BIN, augment=False, need_moves_left=True)
    batch = [torch.stack([ds[i][k] for i in range(4, 12)]) for k in range(12)]
    # Mark half the samples as cheap searches.
    batch[4] = torch.tensor([1.0, 0.0] * 4)
    only_full = [t[::2] for t in batch]
    dev = torch.device("cpu")
    with torch.no_grad():
        mixed = compute_batch_loss(model, batch, dev, 1.0, 0.0, 0.0, False, cheap_value_weight=0.0)[2]
        full = compute_batch_loss(model, only_full, dev, 1.0, 0.0, 0.0, False)[2]
        default = compute_batch_loss(model, batch, dev, 1.0, 0.0, 0.0, False)[2]
        explicit = compute_batch_loss(model, batch, dev, 1.0, 0.0, 0.0, False, cheap_value_weight=1.0)[2]
    assert torch.allclose(mixed, full)
    assert torch.equal(default, explicit)


def test_train_cli_with_a_plane_list(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    out = tmp_path / "probe.onnx"
    cmd = [sys.executable, str(HERE / "train.py"), "--data", str(FIX_BIN), "--features", str(FIX_FEAT),
           "--planes", "base,-throne,last2,rep_dst", "--out", str(out), "--steps", "2", "--batch", "4",
           "--channels", "8", "--blocks", "1", "--cheap-value-weight", "0", "--save-checkpoint",
           str(tmp_path / "probe.qnxx")]
    res = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "'in_channels': 16" in res.stdout
    assert out.exists()
    bad = subprocess.run(cmd[:4] + ["--planes", "base,last2", "--out", str(out)], cwd=HERE,
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "needs --features" in bad.stderr
