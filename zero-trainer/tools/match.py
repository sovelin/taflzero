"""Paired match between two nets at a fixed node budget, over UCI only.

    python zero-trainer/tools/match.py --exe taflzero.exe --a net_a.onnx --b net_b.onnx
        [--pairs 50] [--nodes 400] [--workers 2] [--opening-moves 16] [--max-plies 400]
        [--variant copenhagen11x11] [--json out.json] [--seed N]

Each pair: one random opening (`opening-moves` uniformly random legal moves from the
start), played twice with the nets swapping sides. Score and Elo are for `--a`
relative to `--b`. Games that hit `max-plies` count as draws. The engine's own `result`
command decides the end of a game, so no second rules implementation is involved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
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


def play_game(atk: Engine, dfn: Engine, opening: list[str], nodes: int, max_plies: int) -> str:
    """'attackers' / 'defenders' / 'draw'. Attackers move first (Copenhagen)."""
    cur = list(opening)
    for ply in range(max_plies):
        who = atk.result(cur)
        if who:
            return who
        side_engine = atk if len(cur) % 2 == 0 else dfn
        best, _ = side_engine.search(cur, nodes)
        if best is None:
            return "draw"
        cur.append(best)
    return "draw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--opening-moves", type=int, default=16)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--variant", default="copenhagen11x11")
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    env = dict(os.environ, TAFLZERO_ORT_THREADS="1")
    lock = threading.Lock()
    totals = {"w": 0, "l": 0, "d": 0, "pairs": 0}
    next_pair = [0]
    t0 = time.time()

    err_path = (args.json + ".err") if args.json else None

    def report_error(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, file=sys.stderr, flush=True)
        if err_path:
            with open(err_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def worker() -> None:
        ea = eb = None
        failures = 0
        try:
            while True:
                with lock:
                    if next_pair[0] >= args.pairs:
                        return
                    next_pair[0] += 1
                try:
                    if ea is None:
                        ea = Engine(args.exe, args.a, args.variant, env)
                        eb = Engine(args.exe, args.b, args.variant, env)
                    opening = make_opening(ea, args.opening_moves, rng)
                    r1 = play_game(ea, eb, opening, args.nodes, args.max_plies)  # a attacks
                    r2 = play_game(eb, ea, opening, args.nodes, args.max_plies)  # b attacks
                except Exception as e:  # engine crash or hang: log, restart engines, replay the pair
                    failures += 1
                    report_error(f"pair failed ({type(e).__name__}: {e}); restarting engines (failure {failures})")
                    for eng in (ea, eb):
                        if eng is not None:
                            try:
                                eng.proc.kill()
                            except Exception:
                                pass
                    ea = eb = None
                    with lock:
                        next_pair[0] -= 1
                    if failures >= 5:
                        report_error("giving up on this worker after 5 failures")
                        return
                    continue
                s1 = {"attackers": "w", "defenders": "l", "draw": "d"}[r1]
                s2 = {"attackers": "l", "defenders": "w", "draw": "d"}[r2]
                with lock:
                    totals[s1] += 1
                    totals[s2] += 1
                    totals["pairs"] += 1
                    if not args.quiet:
                        n = totals["w"] + totals["l"] + totals["d"]
                        sc = (totals["w"] + 0.5 * totals["d"]) / n
                        print(f"pair {totals['pairs']}/{args.pairs}: {r1}/{r2} | W{totals['w']} L{totals['l']} D{totals['d']} | elo {elo_from_score(sc):+.0f}", flush=True)
        finally:
            for eng in (ea, eb):
                if eng is not None:
                    eng.close()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, args.workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    n = totals["w"] + totals["l"] + totals["d"]
    score = (totals["w"] + 0.5 * totals["d"]) / n if n else 0.5
    lo, hi = elo_ci(score, n)
    result = {
        "a": os.path.basename(args.a), "b": os.path.basename(args.b),
        "pairs": totals["pairs"], "games": n, "w": totals["w"], "l": totals["l"], "d": totals["d"],
        "score": round(score, 4), "elo": round(elo_from_score(score), 1), "ci": [round(lo, 1), round(hi, 1)],
        "nodes": args.nodes, "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
