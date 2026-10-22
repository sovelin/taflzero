"""Tests for the dashboard's game viewer: tools/match.py's game records, the self-play
game index and the /api/game decoder (docs/plan-game-viewer.md).

    python -m pytest zero-trainer/test_game_viewer.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_match_tool():
    # `match` is a soft keyword and tools/ is not a package: load the file directly.
    spec = importlib.util.spec_from_file_location("match_tool", HERE / "tools" / "match.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


match_tool = _load_match_tool()


# в”Ђв”Ђв”Ђ tools/match.py: moves / evals lines в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


def test_parse_record_lines() -> None:
    p = match_tool.parse_record_line
    assert p("moves 3 1 d1d3 e5e3 h1h3") == ("moves", 3, True, ["d1d3", "e5e3", "h1h3"])
    assert p("evals 3 0 0.1250 -0.5000") == ("evals", 3, False, [0.125, -0.5])
    # A game that ended inside its opening has no evals; one with no moves at all is legal too.
    assert p("evals 0 1") == ("evals", 0, True, [])
    assert p("moves 0 1 ") == ("moves", 0, True, [])
    assert p("game 0 1 attackers atk_capture 40") is None
    assert p("evals 0 1 nan?") is None
    assert p("done") is None


def test_match_stream_pairs_games_with_their_records() -> None:
    s = match_tool.MatchStream()
    lines = [
        "game 1 0 defenders def_corner 3",
        "moves 1 0 d1d3 e5e3 h1h3",
        "evals 1 0 0.1000 -0.2000",
        "game 0 1 attackers atk_capture 2",
        "moves 0 1 a4a3 e5e2",
        "evals 0 1 0.3000",
        "game 1 1 draw max_plies 1",
        "moves 1 1 d1d2",
        "evals 1 1",
        "game 0 0 defenders def_fort 2",
        "moves 0 0 a4a2 f4f2",
        "evals 0 0 0.0000 0.5000",
        "done",
    ]
    out = [pair for line in lines for pair in s.feed(line)]
    assert s.done
    assert [p for p, _ in out] == [1, 0]
    one = dict(out)[1]
    assert one[False] == {"winner": "defenders", "reason": "def_corner", "plies": 3,
                          "moves": ["d1d3", "e5e3", "h1h3"], "evals": [0.1, -0.2]}
    assert one[True]["moves"] == ["d1d2"] and one[True]["evals"] == []
    assert dict(out)[0][True]["evals"] == [0.3]


def test_match_stream_older_engine_without_records() -> None:
    # An engine without the record lines: a pair is complete at the next game line / done.
    s = match_tool.MatchStream()
    got = []
    for line in ["game 0 1 attackers atk_capture 20", "game 0 0 draw max_plies 30", "done"]:
        got += s.feed(line)
    assert len(got) == 1
    pair, both = got[0]
    assert pair == 0 and "moves" not in both[True] and both[False]["reason"] == "max_plies"


# в”Ђв”Ђв”Ђ dashboard: self-play game index and /api/game в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

import json  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402

sys.path.insert(0, str(HERE))
import dashboard  # noqa: E402

N = 11
LEGAL_BYTES = dashboard.legal_mask_bytes(N)


def pack(squares) -> bytes:
    out = bytearray(16)
    for sq in squares:
        out[sq // 8] |= 1 << (sq % 8)
    return bytes(out)


def sample(atk, dfn, king, stm, *, value=1, root_q=64, flags=dashboard.VIEWER_FLAG_POLICY_VALID,
           policy=((131, 30), (7, 10))) -> bytes:
    """One v2 sample (dataset.py layout); the legal mask is not read by the viewer."""
    out = bytearray(pack(atk) + pack(dfn) + pack([king]) + bytes([stm, 1]) + bytes(LEGAL_BYTES))
    out += struct.pack("<H", len(policy))
    for mi, v in policy:
        out += struct.pack("<HH", mi, v)
    out += struct.pack("<bbBB", value, root_q, flags, 255)
    return bytes(out)


def hand_game() -> list[bytes]:
    """Attackers d1-d3 (3 -> 25), defenders e5-... (50 -> 52); attackers win, so z is +1
    on attacker plies and -1 on defender plies. The last sample closes the game."""
    last = dashboard.VIEWER_FLAG_POLICY_VALID | dashboard.VIEWER_FLAG_LAST_OF_GAME
    return [sample([3, 9], [50], 60, 0, value=1),
            sample([25, 9], [50], 60, 1, value=-1, root_q=-20, flags=0, policy=()),
            sample([25, 9], [52], 60, 0, value=1, flags=last)]


def plain_game(length: int, winner: str = "def") -> list[bytes]:
    """A game of `length` samples (positions do not matter for the index)."""
    out = []
    for i in range(length):
        stm = i % 2
        flags = dashboard.VIEWER_FLAG_POLICY_VALID
        if i == length - 1:
            flags |= dashboard.VIEWER_FLAG_LAST_OF_GAME
        if winner == "draw":
            flags |= dashboard.VIEWER_FLAG_RESULT_UNKNOWN
            value = 0
        else:
            value = 1 if (winner == "atk") == (stm == 0) else -1
        out.append(sample([1 + i], [50], 60, stm, value=value, flags=flags))
    return out


def make_run(tmp_path: Path, games: list[list[bytes]]) -> tuple[Path, Path]:
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"board_size": 11, "variant": "copenhagen11x11"}))
    bin_path = run / "data" / "worker0.bin"
    bin_path.write_bytes(b"".join(b"".join(g) for g in games))
    dashboard.GAME_INDEXES.clear()
    return run, bin_path


def test_game_index_ignores_truncated_game_then_picks_it_up(tmp_path: Path) -> None:
    g1, g2, g3 = plain_game(3, "def"), plain_game(4, "atk"), plain_game(5, "draw")
    run, bin_path = make_run(tmp_path, [g1, g2, g3[:2]])
    res = dashboard.selfplay_games(run, 0, 50)
    assert "indexing" not in res
    assert [(g["start_sample"], g["plies"], g["result"]) for g in res["games"]] == [(3, 4, "atk"), (0, 3, "def")]
    idx = json.loads((run / "data" / "games0.idx.json").read_text())
    assert idx["bin_size"] == len(b"".join(g1 + g2)) and idx["samples"] == 7
    assert [g[:3] for g in idx["games"]] == [[0, 3, 0], [len(b"".join(g1)), 4, 3]]
    # The game completes: the next request extends the index from where it stopped.
    with open(bin_path, "ab") as f:
        f.write(b"".join(g3[2:]))
    res = dashboard.selfplay_games(run, 0, 50)
    assert [(g["start_sample"], g["plies"], g["result"]) for g in res["games"]] == [
        (7, 5, "draw"), (3, 4, "atk"), (0, 3, "def")]
    # An undecided game can only have ended on a limit; decided ones need the side file.
    assert [g["reason"] for g in res["games"]] == ["max_plies", "unknown", "unknown"]
    # A fresh dashboard (no in-memory index) reads the saved index back.
    dashboard.GAME_INDEXES.clear()
    assert len(dashboard.selfplay_games(run, 0, 2)["games"]) == 2


def test_game_index_first_build_runs_in_background(tmp_path: Path, monkeypatch) -> None:
    run, _ = make_run(tmp_path, [plain_game(3), plain_game(2)])
    monkeypatch.setattr(dashboard, "GAME_INDEX_SYNC_BYTES", 0)
    monkeypatch.setattr(dashboard, "GAME_INDEX_CHUNK", 1000)  # several chunks, each saved
    res = dashboard.selfplay_games(run, 0, 50)
    assert "indexing" in res  # the request does not wait for the scan
    idx = dashboard.game_index(run, 0)
    t = idx.thread
    if t is not None:
        t.join(10)
    res = dashboard.selfplay_games(run, 0, 50)
    assert "indexing" not in res and [g["plies"] for g in res["games"]] == [2, 3]


def test_reason_from_feature_side_file(tmp_path: Path) -> None:
    run, bin_path = make_run(tmp_path, [plain_game(3, "def"), plain_game(2, "atk")])
    rec = 144
    feat = bytearray(b"TZFEAT01" + struct.pack("<II", rec, 0) + bytes(rec * 4))  # covers samples 0..3 only
    feat[16 + 2 * rec + 77] = 5  # last sample of game 1: def_corner
    bin_path.with_suffix(".feat").write_bytes(bytes(feat))
    res = dashboard.selfplay_games(run, 0, 50)
    assert res["feat"]
    # Game 2 ends on sample 4, past the side file: unknown.
    assert [g["reason"] for g in res["games"]] == ["unknown", "def_corner"]


def test_api_game_reproduces_hand_made_game(tmp_path: Path) -> None:
    run, _ = make_run(tmp_path, [plain_game(2), hand_game()])
    games = dashboard.selfplay_games(run, 0, 50)["games"]
    assert games[0]["id"] == "0:2" and games[0]["result"] == "atk"
    d = dashboard.selfplay_game(run, 0, 2)
    assert d["result"] == "atk" and len(d["plies"]) == 3
    assert [p["move"] for p in d["plies"]] == [None, {"from": 3, "to": 25}, {"from": 50, "to": 52}]
    assert [p["stm"] for p in d["plies"]] == [0, 1, 0]
    assert [p["cheap"] for p in d["plies"]] == [False, True, False]
    assert d["plies"][1]["root_q"] == round(-20 / 127, 4)
    p0 = d["plies"][0]
    assert sorted((x["sq"], x["type"]) for x in p0["pieces"]) == [
        (3, "attacker"), (9, "attacker"), (50, "defender"), (60, "king")]
    # Policy index 131 = from 3, Down, distance 2 -> 25; visit shares of the target.
    assert p0["top"][0] == {"from": 3, "to": 25, "visits": 30, "share": 0.75}
    assert p0["top"][1]["share"] == 0.25
    assert dashboard.selfplay_game(run, 0, 1).get("error")


def test_policy_index_round_trip() -> None:
    # The inverse of move_to_policy_index for every direction on 11x11.
    n, md = 11, 10
    for frm, d, dist, to in [(60, 0, 3, 27), (60, 1, 5, 115), (60, 2, 5, 55), (60, 3, 1, 61)]:
        assert dashboard.policy_move(frm * 4 * md + d * md + dist - 1, n) == {"from": frm, "to": to}


def _engine() -> Path | None:
    exe = "taflzero.exe" if sys.platform == "win32" else "taflzero"
    for d in ("target-match/release", "target/release", "target/debug"):
        p = HERE.parent / d / exe
        if p.exists():
            return p
    return None


@pytest.mark.skipif(_engine() is None, reason="no taflzero binary built")
def test_match_game_replays_moves_with_engine(tmp_path: Path) -> None:
    run, _ = make_run(tmp_path, [plain_game(2)])
    rec = {"key": "step100|anchor", "time": "t", "nodes": 10, "games": [
        {"a_side": "def", "score": "w", "reason": "def_corner", "plies": 3},  # no record: not listed
        {"a_side": "atk", "score": "l", "reason": "def_corner", "plies": 3,
         "moves": "d1d3 e5e3 h1h3", "evals": [0.25, -0.5]}]}
    (run / "eval.games.jsonl").write_text(json.dumps(rec) + "\n")
    listed = dashboard.match_games(run, 50)["games"]
    assert [g["id"] for g in listed] == ["m:0:1"] and listed[0]["ckpt"] == "step100"
    d = dashboard.match_game(run, _engine(), "m:0:1")
    assert "error" not in d, d
    assert len(d["plies"]) == 4 and d["opening"] == 1 and d["result"] == "def"
    assert [p["move"] for p in d["plies"]] == [None, {"from": 3, "to": 25}, {"from": 48, "to": 26},
                                               {"from": 7, "to": 29}]
    # Ply 1 (defenders to move, the anchor's move as a is attacking) carries evals[0].
    assert [p["root_q"] for p in d["plies"]] == [None, 0.25, -0.5, None]
    assert [p["engine"] for p in d["plies"]] == [None, "anchor", "step100", None]
    assert [p["role"] for p in d["plies"]] == [None, "anchor", "ckpt", None]
    sq = {(x["sq"], x["type"]) for x in d["plies"][3]["pieces"]}
    assert (25, "attacker") in sq and (26, "defender") in sq and (29, "attacker") in sq
