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


# ─── tools/match.py: moves / evals lines ────────────────────────────────────


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
