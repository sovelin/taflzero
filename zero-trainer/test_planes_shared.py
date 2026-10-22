"""Fixtures shared with the Rust engine (docs/plan-T7-planes.md, items 1 and 4).

zero-trainer/testdata/planes.json lists plane specs with their canonical form and
channel order as `dataset.PlaneSpec` sees them, plus input tensors this side builds
from the two-game fixture (features.bin / features.feat). The Rust tests
`nn_common::tests::planes_fixture_*` parse the same specs and fill the same inputs; both
must agree bit for bit. TAFLZERO_UPDATE_FIXTURES=1 rewrites the files from Python.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from dataset import FEAT_TERMINAL, FEAT_TERMINAL_NAMES, SelfPlayDataset, parse_planes

HERE = Path(__file__).resolve().parent
TESTDATA = HERE / "testdata"
PLANES_JSON = TESTDATA / "planes.json"
FIX_BIN = TESTDATA / "features.bin"
FIX_FEAT = TESTDATA / "features.feat"

sys.path.insert(0, str(HERE / "tools"))
from probe_planes import VARIANTS  # noqa: E402

# Beyond the probe's variants: removals out of order, the other history lengths,
# extras without base, `ones`.
EXTRA_SPECS = [
    "base,last1", "base,last4", "base,last2,-rep,-throne", "base,-stm,-group_bfs,-king_bfs",
    "last1,ply", "ones", "base,ones,mob,cap", "rep_dst,base,-edges",
]
# Inputs filled on both sides from the fixture games.
INPUT_SPECS = [
    "base,last2,rep_dst,king12,ply,cap,mob",
    "base,-throne,-rep,last4,ones",
    "last1,ply,king12",
]


def _expected() -> dict:
    specs = []
    seen = set()
    for spec in [p for p, _ in VARIANTS.values()] + EXTRA_SPECS:
        if spec in seen:
            continue
        seen.add(spec)
        ps = parse_planes(spec)
        specs.append({"spec": spec, "canonical": ps.canonical, "channels": ps.channel_names()})
    inputs = [{"spec": s, "file": f"inputs_{i}.f32"} for i, s in enumerate(INPUT_SPECS)]
    return {"specs": specs, "inputs": inputs}


def _input_bytes(spec: str) -> bytes:
    ds = SelfPlayDataset(FIX_BIN, augment=False, planes=spec, features=FIX_FEAT)
    stack = np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype("<f4")
    return stack.tobytes()


def test_shared_plane_fixture_is_current() -> None:
    want = _expected()
    if os.environ.get("TAFLZERO_UPDATE_FIXTURES") == "1":
        PLANES_JSON.write_text(json.dumps(want, indent=1) + "\n", encoding="utf-8")
        for item in want["inputs"]:
            (TESTDATA / item["file"]).write_bytes(_input_bytes(item["spec"]))
    assert json.loads(PLANES_JSON.read_text(encoding="utf-8")) == want, \
        "rerun with TAFLZERO_UPDATE_FIXTURES=1"
    for item in want["inputs"]:
        assert (TESTDATA / item["file"]).read_bytes() == _input_bytes(item["spec"]), item


def test_canonical_form_round_trips() -> None:
    for item in _expected()["specs"]:
        ps = parse_planes(item["spec"])
        again = parse_planes(ps.canonical)
        assert (again.base, again.extras) == (ps.base, ps.extras)
        assert len(item["channels"]) == ps.num_planes
    assert parse_planes("base").canonical == "base"
    assert parse_planes("last2,base,-rep,-throne").canonical == "base,-throne,-rep,last2"


def test_terminal_byte_in_fixture() -> None:
    feat = FIX_FEAT.read_bytes()[16:]
    recs = [feat[i:i + 144] for i in range(0, len(feat), 144)]
    terms = [r[FEAT_TERMINAL] for r in recs]
    # Game 1 (3 samples) stopped undecided, game 2 (9) ended by threefold repetition.
    assert terms == [0, 0, 9] + [0] * 8 + [4]
    assert FEAT_TERMINAL_NAMES[9] == "max_plies" and FEAT_TERMINAL_NAMES[4] == "atk_threefold"
