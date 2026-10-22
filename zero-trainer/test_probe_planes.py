"""tools/probe_planes.py on the two-game fixture (zero-trainer/testdata, see test_planes.py):
the split is by game, the holdout subsets come from the samples and the side file, a
run writes its json and --report turns the jsons into a table relative to A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tools"))
import probe_planes  # noqa: E402

FIX_BIN = HERE / "testdata" / "features.bin"
FIX_FEAT = HERE / "testdata" / "features.feat"


def _args(out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data=FIX_BIN, features=FIX_FEAT, out=out, device="cpu", no_amp=True, channels=8, blocks=1,
        lr=1e-3, weight_decay=1e-4, z_lambda=0.5, epochs=2.0, steps=3, batch=2, eval_every=2,
        eval_batch=8, workers=0, curve_samples=0,
    )


def test_split_by_game_and_holdout_subsets() -> None:
    data = probe_planes.ProbeData(FIX_BIN, FIX_FEAT, samples=1000, holdout_frac=0.05,
                                  z_lambda=0.5, curve_samples=0)
    # Two games (3 + 9 samples): the second, later one is the holdout.
    assert (data.window, data.games, data.split, data.holdout_games) == (12, 2, 3, 1)
    sizes = {k: int(v.sum()) for k, v in data.subsets.items()}
    assert sizes == {"all": 9, "tactical": 6, "king_move": 0, "repeat_game": 9, "cheap": 0}
    assert data.hold_known.all() and (abs(data.hold_z) == 1).all()
    # A window that starts inside the first game moves to the next game boundary;
    # one game alone cannot be split.
    with pytest.raises(SystemExit, match="only 1 game"):
        probe_planes.ProbeData(FIX_BIN, FIX_FEAT, samples=10, holdout_frac=0.05,
                               z_lambda=0.5, curve_samples=0)
    with pytest.raises(SystemExit, match="inside the last game"):
        probe_planes.ProbeData(FIX_BIN, FIX_FEAT, samples=5, holdout_frac=0.05,
                               z_lambda=0.5, curve_samples=0)


def test_run_writes_json_and_report(tmp_path: Path) -> None:
    data = probe_planes.ProbeData(FIX_BIN, FIX_FEAT, samples=1000, holdout_frac=0.5,
                                  z_lambda=0.5, curve_samples=0)
    args = _args(tmp_path)
    for vid in ("A", "I"):
        r = probe_planes.run_variant(vid, 1, data, args, tmp_path / f"{vid}-s1.json")
        assert r["in_channels"] == {"A": 11, "I": 24}[vid]
    saved = json.loads((tmp_path / "I-s1.json").read_text())
    assert saved["final"]["all"]["n"] == 9 and saved["final"]["all"]["policy_ce"] is not None
    assert [p["step"] for p in saved["curve"]] == [2, 3]
    report = probe_planes.write_report(tmp_path).read_text(encoding="utf-8")
    assert "| A |" in report and "| I |" in report and "%" in report
    assert probe_planes.stage2_variants(tmp_path)[0] == "A"


def test_report_needs_a(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        probe_planes.write_report(tmp_path)
