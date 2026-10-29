"""Paired match between two nets at a fixed node budget.

    python zero-trainer/tools/match.py --exe taflzero.exe --a net_a.onnx --b net_b.onnx
        [--pairs 50] [--nodes 400] [--threads 0] [--opening-moves 16] [--max-plies 400]
        [--variant copenhagen11x11] [--json out.json] [--seed N]

Each pair: one random opening (`opening-moves` uniformly random legal moves from the
start), played twice with the nets swapping sides. Score and Elo are for `--a`
relative to `--b`. Games that hit `max-plies` count as draws. The games are played by
one engine process in match mode (`taflzero --match`, src/search/match_play.rs): every
game is a thread there and the leaves of all games on one net's move go to the GPU in
one call, so a match costs the GPU a fraction of what two UCI engines per pair did.
`--threads 0` plays every game of the match at once. The engine names how each game
ended (`def_fort`, `atk_capture`, ...), which the result and the progress file tally
from `--a`'s side ("win" / "loss" / "draw" -> reason -> count). With `--progress`,
every finished match is also appended to `eval.games.jsonl` next to that file (or
`--games-log`): one line per match with each game's reason and length, and, from an
engine that prints them, the game's `moves` (space-separated, opening included) and
`evals` (each searched move's root value from the mover's view) for the dashboard's
game viewer.

`make_opening` / `play_game` below drive a UCI engine and stay for the tools that
record games move by move (build_fort_bench.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uci_client import Engine  # noqa: E402


def elo_from_score(score: float) -> float:
    if score <= 0.0:
        return -999.0
    if score >= 1.0:
        return 999.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_ci(score: float, n_games: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval in Elo from the Wilson score interval (sane at 0% and 100%)."""
    if n_games == 0:
        return (-999.0, 999.0)
    n = n_games
    centre = (score + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo = min(max(centre - half, 0.001), 0.999)
    hi = min(max(centre + half, 0.001), 0.999)
    return (elo_from_score(lo), elo_from_score(hi))


def make_opening(eng: Engine, moves: int, rng: random.Random) -> list[str]:
    """Random legal moves from the start; stops early if a game-ending position shows up."""
    cur: list[str] = []
    for _ in range(moves):
        if eng.result(cur):
            break
        mv = eng.random_move(cur)
        if mv is None:
            break
        cur.append(mv)
    # The engine's randommove uses its own RNG; `rng` only decides whether to shorten
    # the opening a little so openings of one run are not all the same length.
    if cur and rng.random() < 0.25:
        cur = cur[: max(0, len(cur) - 1)]
    return cur


def play_game(atk: Engine, dfn: Engine, opening: list[str], nodes: int, max_plies: int) -> tuple[str, str, int]:
    """('attackers' / 'defenders' / 'draw', how it ended, plies). Attackers move first
    (Copenhagen)."""
    cur = list(opening)
    for ply in range(max_plies):
        who, reason = atk.result_reason(cur)
        if who:
            return who, reason or "unknown", len(cur)
        side_engine = atk if len(cur) % 2 == 0 else dfn
        best, _ = side_engine.search(cur, nodes)
        if best is None:
            return "draw", "no_bestmove", len(cur)
        cur.append(best)
    return "draw", "max_plies", len(cur)


def parse_game_line(line: str) -> tuple[int, bool, str, str, int] | None:
    """`game <pair> <a_attacks 0|1> <winner> <reason> <plies>` from the engine, else None."""
    parts = line.split()
    if len(parts) != 6 or parts[0] != "game":
        return None
    return int(parts[1]), parts[2] == "1", parts[3], parts[4], int(parts[5])


def parse_record_line(line: str) -> tuple[str, int, bool, list[str] | list[float]] | None:
    """`moves <pair> <a_attacks> <m1> …` / `evals <pair> <a_attacks> <v1> …` from the
    engine (printed right after that game's `game` line), else None. Either list may be
    empty: a game can end inside its opening, and the opening has no evals."""
    parts = line.split()
    if len(parts) < 3 or parts[0] not in ("moves", "evals"):
        return None
    try:
        pair, a_attacks = int(parts[1]), parts[2] == "1"
        rest: list[str] | list[float] = parts[3:] if parts[0] == "moves" else [float(v) for v in parts[3:]]
    except ValueError:
        return None
    return parts[0], pair, a_attacks, rest


class MatchStream:
    """Turns the engine's match output into finished pairs.

    Games of a pair finish in any order; a pair is handed out once both of its games
    are in. A game is in once its record is complete: the engine prints `game`, `moves`
    and `evals` of one game under one lock, so the `evals` line closes it, and so does
    the next `game` line or `done` (an older engine prints no record lines at all).
    """

    def __init__(self) -> None:
        self.games: dict[tuple[int, bool], dict] = {}
        self.open: tuple[int, bool] | None = None
        self.closed: dict[int, set[bool]] = {}
        self.done = False

    def _close(self) -> list[tuple[int, dict[bool, dict]]]:
        key, self.open = self.open, None
        if key is None:
            return []
        pair, a_attacks = key
        sides = self.closed.setdefault(pair, set())
        sides.add(a_attacks)
        if len(sides) < 2:
            return []
        del self.closed[pair]
        return [(pair, {s: self.games.pop((pair, s)) for s in (True, False)})]

    def feed(self, line: str) -> list[tuple[int, dict[bool, dict]]]:
        """Pairs completed by `line`: (pair, {a_attacks: {winner, reason, plies, moves?, evals?}})."""
        line = line.strip()
        if line == "done":
            self.done = True
            return self._close()
        g = parse_game_line(line)
        if g is not None:
            out = self._close()
            pair, a_attacks, winner, why, plies = g
            self.games[(pair, a_attacks)] = {"winner": winner, "reason": why, "plies": plies}
            self.open = (pair, a_attacks)
            return out
        rec = parse_record_line(line)
        if rec is not None and rec[1:3] == self.open:
            kind, _, _, values = rec
            self.games[self.open][kind] = values  # type: ignore[index]
            if kind == "evals":
                return self._close()
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--threads", type=int, default=0, help="game threads in the engine; 0 = every game at once")
    ap.add_argument("--search-batch", type=int, default=8,
                    help="leaves per net call inside one search; 8 is what UCI play uses, 1 is plain MCTS "
                         "(and an 8x longer chain of net calls per game, so a slower match)")
    ap.add_argument("--workers", type=int, default=None, help="ignored (older callers); see --threads")
    ap.add_argument("--opening-moves", type=int, default=16)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--variant", default="copenhagen11x11")
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--progress", default=None, help="JSON file rewritten after every pair (live score for a dashboard)")
    ap.add_argument("--games-log", default=None,
                    help="append one line per match with every game's ending (default: eval.games.jsonl next to --progress)")
    args = ap.parse_args()

    env = dict(os.environ, TAFLZERO_ORT_THREADS="1")
    totals = {"w": 0, "l": 0, "d": 0, "pairs": 0}
    t0 = time.time()

    err_path = (args.json + ".err") if args.json else None
    recent: list[str] = []
    reasons: dict[str, dict[str, int]] = {"win": {}, "loss": {}, "draw": {}}
    games: list[dict] = []

    def write_progress() -> None:
        """Atomic rewrite so a reader never sees a half file."""
        if not args.progress:
            return
        n = totals["w"] + totals["l"] + totals["d"]
        sc = (totals["w"] + 0.5 * totals["d"]) / n if n else 0.5
        lo, hi = elo_ci(sc, n)
        state = {
            "phase": "match", "a": os.path.basename(args.a), "b": os.path.basename(args.b),
            "pairs_done": totals["pairs"], "pairs": args.pairs, "games": n,
            "w": totals["w"], "l": totals["l"], "d": totals["d"],
            "score": round(sc, 4), "elo": round(elo_from_score(sc), 1) if n else None,
            "ci": [round(lo, 1), round(hi, 1)] if n else None,
            "recent": recent[-10:], "reasons": reasons, "started": t0, "updated": time.time(),
            "nodes": args.nodes,
        }
        tmp = args.progress + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, args.progress)

    def report_error(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, file=sys.stderr, flush=True)
        if err_path:
            with open(err_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    write_progress()

    cmd = [os.path.abspath(args.exe), "--match", "--a", os.path.abspath(args.a), "--b", os.path.abspath(args.b),
           "--pairs", str(args.pairs), "--nodes", str(args.nodes), "--opening-moves", str(args.opening_moves),
           "--max-plies", str(args.max_plies), "--variant", args.variant,
           "--threads", str(args.threads if args.threads > 0 else 2 * args.pairs),
           "--search-batch", str(args.search_batch)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

    stream = MatchStream()
    for line in proc.stdout:  # type: ignore[union-attr]
        for pair, both in stream.feed(line):
            r1, why1, len1 = both[True]["winner"], both[True]["reason"], both[True]["plies"]  # a attacks
            r2, why2, len2 = both[False]["winner"], both[False]["reason"], both[False]["plies"]  # b attacks
            s1 = {"attackers": "w", "defenders": "l", "draw": "d"}[r1]
            s2 = {"attackers": "l", "defenders": "w", "draw": "d"}[r2]
            totals[s1] += 1
            totals[s2] += 1
            totals["pairs"] += 1
            recent.append(f"{r1[:3]}/{r2[:3]}")
            for side, sc, why, n_plies, rec in (("atk", s1, why1, len1, both[True]), ("def", s2, why2, len2, both[False])):
                bucket = reasons[{"w": "win", "l": "loss", "d": "draw"}[sc]]
                bucket[why] = bucket.get(why, 0) + 1
                entry = {"a_side": side, "score": sc, "reason": why, "plies": n_plies}
                # The game's record for the dashboard's viewer (absent from older engines).
                if "moves" in rec:
                    entry["moves"] = " ".join(rec["moves"])
                if "evals" in rec:
                    entry["evals"] = rec["evals"]
                games.append(entry)
            write_progress()
            if not args.quiet:
                n = totals["w"] + totals["l"] + totals["d"]
                sc = (totals["w"] + 0.5 * totals["d"]) / n
                print(f"pair {totals['pairs']}/{args.pairs}: {r1}/{r2} | W{totals['w']} L{totals['l']} D{totals['d']} | elo {elo_from_score(sc):+.0f}", flush=True)
        if stream.done:
            break
    done = stream.done
    stderr = proc.stderr.read() if proc.stderr else ""  # type: ignore[union-attr]
    proc.wait()
    if not done or totals["pairs"] != args.pairs:
        report_error(f"engine ended after {totals['pairs']}/{args.pairs} pairs (exit {proc.returncode}): {stderr.strip()[-500:]}")
        sys.exit(1)

    n = totals["w"] + totals["l"] + totals["d"]
    score = (totals["w"] + 0.5 * totals["d"]) / n if n else 0.5
    lo, hi = elo_ci(score, n)
    result = {
        "a": os.path.basename(args.a), "b": os.path.basename(args.b),
        "pairs": totals["pairs"], "games": n, "w": totals["w"], "l": totals["l"], "d": totals["d"],
        "score": round(score, 4), "elo": round(elo_from_score(score), 1), "ci": [round(lo, 1), round(hi, 1)],
        "nodes": args.nodes, "seconds": round(time.time() - t0, 1), "reasons": reasons,
    }
    games_log = args.games_log or (os.path.join(os.path.dirname(os.path.abspath(args.progress)), "eval.games.jsonl")
                                   if args.progress else None)
    if games_log:
        stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731
        with open(games_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": f"{stem(args.a)}|{stem(args.b)}", "time": time.strftime("%Y-%m-%d %H:%M"),
                                "nodes": args.nodes, "reasons": reasons, "games": games}) + "\n")
    print(json.dumps(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
