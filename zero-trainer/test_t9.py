"""T9 trainer changes (docs/plan-T9.md): short-term value targets, WDL value head,
soft policy, surprise / soft-resignation sampling weights, config plumbing.

    python -m pytest test_t9.py -q
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from az_micro_net import TaflAlphaZeroNet
from continuous import ContinuousRun, build_model, parse_config, parse_td_horizons, worker_command
from qnxx_io import load_qnxx, save_qnxx
from replay import FEAT2_REDUCED, FEAT_FLAGS2, FEAT_SURPRISE, ReplayBuffer, sample_weights, td_targets
from test_continuous import GEOM, _chunk, _tiny_args, make_game, write_games
from train import WDL_DRAW, WDL_LOSS, WDL_WIN, compute_batch_loss, soft_policy_target, wdl_classes


# ─── short-term value targets ───────────────────────────────────────────────


def test_td_targets_follow_the_recursion_with_alternating_sides() -> None:
    q = np.array([0.2, -0.4, 0.6, -0.8])
    # Horizon 1: lambda = 1, the target is the sample's own root value.
    t = td_targets(q, 1.0, True, [1], False)
    assert np.allclose(t[:, 0], q)
    # Horizon 2: lambda = 0.5, built backwards from the result of the last sample.
    t = td_targets(q, 1.0, True, [2], False)[:, 0]
    v3 = 0.5 * -0.8 + 0.5 * 1.0
    v2 = 0.5 * 0.6 + 0.5 * -v3
    v1 = 0.5 * -0.4 + 0.5 * -v2
    v0 = 0.5 * 0.2 + 0.5 * -v1
    assert np.allclose(t, [v0, v1, v2, v3])
    # A very long horizon tends to the result itself, alternating with the side.
    t = td_targets(q, -1.0, True, [10_000], False)[:, 0]
    assert np.allclose(t, [1, -1, 1, -1], atol=1e-3)
    # Several horizons give one column each.
    assert td_targets(q, 1.0, True, [40, 12, 4], False).shape == (4, 3)


def test_td_targets_of_an_unknown_result_are_masked_unless_draws() -> None:
    q = np.array([0.5, -0.5, 0.5])
    assert np.isnan(td_targets(q, 0.0, False, [4], False)).all()
    t = td_targets(q, 0.0, False, [1_000_000], True)[:, 0]
    assert np.isfinite(t).all() and np.allclose(t, 0.0, atol=1e-5)


# ─── sampling weights ───────────────────────────────────────────────────────


def test_sample_weights_uniform_when_off_and_mean_one_when_on() -> None:
    n = 6
    ps = np.array([0.0, 0.5, 2.0, 0.0, 1.0, 0.5])
    q = np.array([0.1, -0.2, 0.9, -0.9, 0.0, 0.3])
    z = np.array([1, -1, 1, -1, 1, -1], dtype=float)
    red = np.array([False, False, False, False, True, True])
    w = sample_weights(ps, q, z, True, np.zeros(n, bool), 0.0, 0.0, 1.0, False)
    assert np.allclose(w, 1.0)
    w = sample_weights(ps, q, z, True, red, 0.5, 0.1, 0.1, False)
    assert np.isclose(w.mean(), 1.0)
    # more surprise -> more weight (same value term, no reduction)
    w2 = sample_weights(ps, np.zeros(n), np.zeros(n), True, np.zeros(n, bool), 0.5, 0.0, 1.0, False)
    assert w2[2] > w2[1] > w2[0] and np.isclose(w2[0], w2[3])
    # reduced samples weigh reduced_weight times a sample with the same surprise
    w3 = sample_weights(np.zeros(n), np.zeros(n), np.zeros(n), True, red, 0.0, 0.0, 0.1, False)
    assert np.isclose(w3[4] / w3[0], 0.1) and np.isclose(w3.mean(), 1.0)
    # an old side file (all zero surprise) gives uniform weights
    w4 = sample_weights(np.zeros(n), q * 0, z * 0, True, np.zeros(n, bool), 0.5, 0.0, 1.0, False)
    assert np.allclose(w4, 1.0)


def _chunk_with_feat(length: int, surprise: list[int] | None = None, reduced: list[bool] | None = None,
                     winner: str = "atk"):
    c = _chunk(length, winner=winner)
    feat = np.zeros((length, 144), dtype=np.uint8)
    if surprise is not None:
        feat[:, FEAT_SURPRISE] = surprise
    if reduced is not None:
        feat[:, FEAT_FLAGS2] = [FEAT2_REDUCED if r else 0 for r in reduced]
    c.attach_feat(feat)
    return c


def test_weighted_draw_is_proportional_and_stays_in_the_window() -> None:
    # Two games of 10; in the second, sample 5 has all the policy surprise.
    buf = ReplayBuffer(GEOM, augment=False, surprise_policy=1.0)
    buf.add(_chunk_with_feat(10, surprise=[0] * 10), window=1000)
    s = [0] * 10
    s[5] = 255
    buf.add(_chunk_with_feat(10, surprise=s), window=1000)
    rng = np.random.default_rng(0)
    idx = buf.draw_indices(20_000, window=1000, rng=rng)
    counts = np.bincount(idx, minlength=20)
    # game 1: uniform weight 1 each; game 2: weight 10 on sample 15, 0 elsewhere
    assert counts[15] / 20_000 == pytest.approx(10 / 20, abs=0.02)
    assert counts[10:15].sum() == 0 and counts[16:20].sum() == 0
    assert counts[:10].sum() / 20_000 == pytest.approx(0.5, abs=0.02)
    # A window that starts inside the first game never draws before it.
    idx = buf.draw_indices(5000, window=13, rng=rng)
    assert idx.min() >= 7 and idx.max() < 20


def test_weighted_draw_with_all_weights_one_is_uniform() -> None:
    buf = ReplayBuffer(GEOM, augment=False, reduced_weight=0.5)   # weighted path, no reduced samples
    for _ in range(4):
        buf.add(_chunk_with_feat(10), window=1000)
    rng = np.random.default_rng(3)
    idx = buf.draw_indices(40_000, window=25, rng=rng)
    assert idx.min() >= 15 and idx.max() < 40
    counts = np.bincount(idx, minlength=40)[15:]
    assert counts.min() > 0.8 * 40_000 / 25 and counts.max() < 1.2 * 40_000 / 25


def test_buffer_reports_surprise_and_reduced_share() -> None:
    buf = ReplayBuffer(GEOM, augment=False, surprise_policy=0.5, reduced_weight=0.1)
    buf.add(_chunk_with_feat(4, surprise=[64, 64, 0, 0], reduced=[False, False, True, True]), window=100)
    st = buf.take_weight_stats()
    # 4 full-search samples (make_game marks all as full), surprise 1.0 nat on two of them
    assert st["surprise"] == pytest.approx(0.5, abs=0.01)
    assert st["reduced"] == pytest.approx(0.5)
    assert st["w_max"] > 1.0
    assert buf.take_weight_stats()["reduced"] is None   # reset after reading: nothing new


# ─── WDL, soft policy, losses ───────────────────────────────────────────────


def test_wdl_classes_and_soft_target() -> None:
    z = torch.tensor([1.0, -1.0, 0.0, 1.0])
    known = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert wdl_classes(z, known).tolist() == [WDL_WIN, WDL_LOSS, WDL_DRAW, WDL_DRAW]
    pi = torch.tensor([[0.9, 0.1, 0.0, 0.0]])
    legal = torch.tensor([[True, True, True, False]])
    t = soft_policy_target(pi, legal)
    assert torch.allclose(t.sum(1), torch.ones(1)) and t[0, 3] == 0
    assert t[0, 0] > t[0, 1] > t[0, 2] > 0 and t[0, 0] < 0.9   # flatter than pi, same order


def _t9_model(**over) -> TaflAlphaZeroNet:
    kw = dict(trunk_channels=8, num_blocks=1, aux_heads=True, moves_left=True,
              value_wdl=True, td_heads=3, soft_policy=True)
    kw.update(over)
    return TaflAlphaZeroNet(**kw)


def test_wdl_model_forward_is_a_scalar_expected_score() -> None:
    torch.manual_seed(0)
    m = _t9_model().eval()
    x = torch.randn(4, 11, 11, 11)
    out = m(x)
    assert len(out) == 3 and out[1].shape == (4, 1)   # policy, value, moves_left: unchanged
    o = m.forward_train(x)
    p = torch.softmax(o["value_logits"], dim=1)
    assert torch.allclose(out[1][:, 0], p[:, 0] - p[:, 2], atol=1e-6)
    assert o["td"].shape == (4, 3) and o["soft_policy"].shape == o["policy"].shape


def test_new_kwargs_are_recorded_only_when_on_and_checkpoints_roundtrip(tmp_path: Path) -> None:
    old = TaflAlphaZeroNet(trunk_channels=8, num_blocks=1)
    assert not {"value_wdl", "td_heads", "soft_policy"} & set(old.model_kwargs)
    m = _t9_model()
    save_qnxx(m, tmp_path / "m.qnxx")
    back = load_qnxx(tmp_path / "m.qnxx")
    assert back.value_wdl and back.model_kwargs["td_heads"] == 3 and "soft_policy_head" in back._modules
    x = torch.randn(2, 11, 11, 11)
    m.eval(), back.eval()
    assert torch.allclose(m(x)[1], back(x)[1])


def test_onnx_export_keeps_outputs_and_value_is_p_win_minus_p_loss(tmp_path: Path) -> None:
    import onnx
    from export_onnx import export_model_to_onnx
    torch.manual_seed(1)
    m = _t9_model().eval()
    path = tmp_path / "m.onnx"
    export_model_to_onnx(m, path, fp16=False)
    graph = onnx.load(str(path)).graph
    assert [o.name for o in graph.output] == ["policy", "value", "moves_left"]
    x = torch.randn(3, 11, 11, 11)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"input": x.numpy()})
    except ImportError:   # a broken onnxruntime install: onnx's reference interpreter
        from onnx.reference import ReferenceEvaluator
        got = ReferenceEvaluator(str(path)).run(None, {"input": x.numpy()})
    p = torch.softmax(m.forward_train(x)["value_logits"], dim=1)
    assert np.allclose(got[1][:, 0], (p[:, 0] - p[:, 2]).detach().numpy(), atol=1e-5)


def _batch(buf: ReplayBuffer, n: int = 16):
    return buf.sample(n, 10_000, np.random.default_rng(0))


def test_old_options_take_the_old_path_and_ignore_the_extended_batch() -> None:
    torch.manual_seed(2)
    m = TaflAlphaZeroNet(trunk_channels=8, num_blocks=1, aux_heads=True, moves_left=True)
    plain = ReplayBuffer(GEOM, augment=False)
    ext = ReplayBuffer(GEOM, augment=False, td_horizons=(4,), extended_batch=True)
    for winner in ("atk", "def", "undecided", "atk"):
        plain.add(_chunk(6, winner=winner), window=10_000)
        ext.add(_chunk(6, winner=winner), window=10_000)
    b12, b14 = _batch(plain), _batch(ext)
    assert len(b12) == 12 and len(b14) == 14
    dev = torch.device("cpu")
    a = compute_batch_loss(m, b12, dev, 1.0, 0.15, 0.1, True, moves_left_weight=0.1)
    b = compute_batch_loss(m, b14, dev, 1.0, 0.15, 0.1, True, moves_left_weight=0.1)
    for x, y in zip(a, b):
        assert torch.allclose(x, y)


def test_t9_loss_trains_every_head() -> None:
    torch.manual_seed(3)
    m = _t9_model()
    buf = ReplayBuffer(GEOM, augment=False, td_horizons=(40, 12, 4), unknown_as_draw=True,
                       surprise_policy=0.5, surprise_value=0.1, reduced_weight=0.1, extended_batch=True)
    for winner in ("atk", "def", "undecided", "atk", "def"):
        buf.add(_chunk_with_feat(8, surprise=[10] * 8, reduced=[False] * 6 + [True] * 2, winner=winner),
                window=10_000)
    batch = _batch(buf, 32)
    extra: dict = {}
    total, p, v, ap, c, ml = compute_batch_loss(
        m, batch, torch.device("cpu"), 1.0, 0.15, 0.1, True, moves_left_weight=0.1,
        value_weight=1.0, td_weight=0.5, soft_policy_weight=8.0, extra=extra)
    assert torch.isfinite(total)
    assert {"td", "soft_p", "draw_pred", "draw_frac"} <= set(extra)
    assert 0.0 < extra["draw_frac"] < 1.0       # the undecided game counts as draws
    total.backward()
    for name in ("td_mlp", "soft_policy_head", "value_mlp"):
        grads = [prm.grad for prm in getattr(m, name).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert any(g.abs().sum() > 0 for g in grads), name


def test_wdl_loss_includes_unknown_results_scalar_loss_masks_them() -> None:
    torch.manual_seed(4)
    buf = ReplayBuffer(GEOM, augment=False, unknown_as_draw=True, extended_batch=True)
    buf.add(_chunk(8, winner="undecided"), window=1000)
    batch = _batch(buf, 8)
    dev = torch.device("cpu")
    wdl = _t9_model(td_heads=0, soft_policy=False)
    _, _, v_wdl, *_ = compute_batch_loss(wdl, batch, dev, 1.0, 0.0, 0.0, False)
    assert v_wdl > 0     # draws are trained
    scalar = TaflAlphaZeroNet(trunk_channels=8, num_blocks=1)
    _, _, v_sc, *_ = compute_batch_loss(scalar, batch, dev, 1.0, 0.0, 0.0, False)
    assert v_sc == 0     # unknown results stay masked for the scalar head


# ─── config plumbing ────────────────────────────────────────────────────────


def test_parse_td_horizons() -> None:
    assert parse_td_horizons("") == () and parse_td_horizons(None) == ()
    assert parse_td_horizons("40, 12,4") == (40, 12, 4)
    with pytest.raises(ValueError):
        parse_td_horizons("40,x")
    with pytest.raises(ValueError):
        parse_td_horizons("0")


def test_defaults_build_the_old_model_and_worker_command(tmp_path: Path) -> None:
    args = parse_config(_tiny_args(tmp_path / "r"))
    m = build_model(args)
    assert not m.value_wdl and "td_head" not in m._modules and "soft_policy_head" not in m._modules
    cmd = worker_command(parse_config(_tiny_args(tmp_path / "r", **{"engine-bin": "x.exe"})),
                         tmp_path, 0, tmp_path / "n.onnx")
    assert "--soft-resign" not in cmd


def test_t9_config_builds_heads_and_passes_soft_resign(tmp_path: Path) -> None:
    over = {"value-wdl": True, "td-horizons": "40,12,4", "soft-policy-weight": 8.0,
            "surprise-policy": 0.5, "surprise-value": 0.1, "reduced-weight": 0.1,
            "soft-resign": 0.9, "soft-resign-plies": 3, "value-lambda": 1.0, "engine-bin": "x.exe"}
    args = parse_config(_tiny_args(tmp_path / "r", **over))
    m = build_model(args)
    assert m.value_wdl and m.model_kwargs["td_heads"] == 3 and "soft_policy_head" in m._modules
    cmd = worker_command(args, tmp_path, 0, tmp_path / "n.onnx")
    i = cmd.index("--soft-resign")
    assert cmd[i + 1] == "0.9" and cmd[cmd.index("--soft-resign-plies") + 1] == "3"
    with pytest.raises(SystemExit):
        parse_config(_tiny_args(tmp_path / "r", **{"surprise-policy": 0.8, "surprise-value": 0.5}))


def test_trainer_loop_with_every_t9_option_and_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    data = run_dir / "data"
    data.mkdir(parents=True)
    rng = np.random.default_rng(0)
    games = [make_game(int(rng.integers(4, 9)), winner=["atk", "def", "undecided"][i % 3], first_stm=i % 2,
                       policy_valid=(i % 4 != 3)) for i in range(12)]
    write_games(data / "worker0.bin", games)
    over = {"aux-heads": True, "moves-left": True, "value-wdl": True, "td-horizons": "40,12,4",
            "soft-policy-weight": 8.0, "surprise-value": 0.1, "reduced-weight": 0.1, "value-lambda": 1.0}
    run = ContinuousRun(parse_config(_tiny_args(run_dir, **over)))
    assert run.run() == 0
    with open(run_dir / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert all(np.isfinite(float(r["loss"])) for r in rows)
    assert rows[-1]["td"] and rows[-1]["soft_p"] and rows[-1]["draw_pred"]
    # weights are reported for the games added since the previous row: all arrive before the first
    assert rows[0]["w_max"] and all(r["w_max"] == "" for r in rows[1:])
    assert rows[-1]["surprise"] == "" and rows[-1]["reduced"] == ""   # those options are off
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["value_wdl"] and cfg["td_horizons"] == "40,12,4"
    (run_dir / "STOP").unlink()
    run2 = ContinuousRun(parse_config(["--resume", str(run_dir), "--max-steps", "8"]))
    assert run2.model.value_wdl and run2.run() == 0
    # A config that disagrees with the model's heads is refused on resume.
    (run_dir / "STOP").unlink()
    with pytest.raises(SystemExit):
        ContinuousRun(parse_config(["--resume", str(run_dir), "--max-steps", "10", "--td-horizons", "4"]))


# ─── review follow-ups ──────────────────────────────────────────────────────


def _td_reference(q, z, horizons):
    out = np.zeros((len(q), len(horizons)))
    for j, h in enumerate(horizons):
        lam = 1.0 / h
        v = lam * q[-1] + (1 - lam) * z
        out[-1, j] = v
        for i in range(len(q) - 2, -1, -1):
            v = lam * q[i] + (1 - lam) * (-v)
            out[i, j] = v
    return out


def test_td_targets_match_the_plain_recursion_on_random_games() -> None:
    rng = np.random.default_rng(5)
    for n in (1, 2, 7, 160, 400):
        q = rng.uniform(-1, 1, n)
        z = float(rng.choice([-1, 1]))
        got = td_targets(q, z, True, [40, 12, 4, 1], False)
        assert np.allclose(got, _td_reference(q, z, [40, 12, 4, 1]), atol=1e-5), n


def test_td_targets_of_a_buffer_chunk_use_its_stored_root_values() -> None:
    buf = ReplayBuffer(GEOM, augment=False, td_horizons=(4,), extended_batch=True)
    c = _chunk(6, winner="def")
    buf.add(c, window=100)
    q = c.root_q.astype(np.float64) / 127.0
    assert np.allclose(c.td, _td_reference(q, float(c.value[-1]), [4]), atol=1e-5)
    # each sample's z is from its own side to move: the last sample's sign decides the end
    row = buf.collate([(c, 5)], [0])
    assert row[12].item() == float(c.value[5]) and row[13].shape == (1, 1)


def test_policy_surprise_does_not_move_weight_from_cheap_to_full_searches() -> None:
    n = 160
    rng = np.random.default_rng(0)
    full = rng.random(n) < 0.25
    ps = np.where(full, 1.0, 0.0)          # every full search equally surprising
    w = sample_weights(ps, np.zeros(n), np.zeros(n), True, np.zeros(n, bool), 0.5, 0.0, 1.0, False, full=full)
    assert np.allclose(w, 1.0)              # nothing to reorder: uniform, as without the term
    ps = np.where(full, rng.uniform(0, 2, n), 0.0)
    w = sample_weights(ps, np.zeros(n), np.zeros(n), True, np.zeros(n, bool), 0.5, 0.0, 1.0, False, full=full)
    assert np.isclose(w[full].sum(), full.sum()) and np.isclose(w[~full].sum(), (~full).sum())


def test_reduced_share_is_reported_with_soft_resignation_alone() -> None:
    buf = ReplayBuffer(GEOM, augment=False)          # no T9 trainer option on
    buf.add(_chunk_with_feat(4, reduced=[False, True, True, True]), window=100)
    assert buf.take_weight_stats()["reduced"] == pytest.approx(0.75)
    assert len(buf.sample(4, 100, np.random.default_rng(0))) == 12   # batch layout unchanged


def test_wdl_refuses_a_bootstrapped_value_target(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_config(_tiny_args(tmp_path / "r", **{"value-wdl": True, "value-lambda": 0.5}))
    parse_config(_tiny_args(tmp_path / "r", **{"value-wdl": True, "value-lambda": 1.0}))


def test_fp16_onnx_value_is_p_win_minus_p_loss(tmp_path: Path) -> None:
    import onnx
    from onnx.reference import ReferenceEvaluator
    from export_onnx import export_model_to_onnx, to_fp16
    torch.manual_seed(6)
    m = _t9_model().eval()
    export_model_to_onnx(m, tmp_path / "m.onnx", fp16=False)
    onnx.save(to_fp16(onnx.load(str(tmp_path / "m.onnx"))), str(tmp_path / "m16.onnx"))
    x = torch.randn(3, 11, 11, 11)
    got = ReferenceEvaluator(str(tmp_path / "m16.onnx")).run(None, {"input": x.numpy()})
    p = torch.softmax(m.forward_train(x)["value_logits"], dim=1)
    assert np.allclose(got[1][:, 0].astype(np.float32), (p[:, 0] - p[:, 2]).detach().numpy(), atol=2e-2)
