"""Builds bench/fort.jsonl: positions from games a strong defender won by a fort.

    python zero-trainer/tools/build_fort_bench.py --exe taflzero.exe
        --attackers runs/T6/ckpt/step120000.onnx,runs/T6/ckpt/step138000.onnx
        --defender copenhagen11x11.onnx [--games 80] [--nodes 400] [--workers 3]

Each game starts from a random opening (as in match.py); an attacker net from
`--attackers` (round robin) attacks, `--defender` defends. Every finished game goes to
bench/fort_games.jsonl (moves, how it ended), so the bench can be rebuilt from it with
`--rebuild`. From every game that ended `def_fort`, the attacker-to-move positions
about k plies before the end (k in KS) go to bench/fort.jsonl, each with the
defender's own raw value there (`ref_wr`, attacker winrate in percent, `go nodes 1`).
Truth for all of them: the defenders won. tools/eval_fort.py scores a net on them.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import make_opening  # noqa: E402
from uci_client import Engine  # noqa: E402

BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench")
KS = (4, 8, 16, 24, 32, 48)


def play_recorded(atk: Engine, dfn: Engine, opening: list[str], nodes: int, max_plies: int) -> tuple[str, str, list[str]]:
    """play_game, but keeps the moves."""
    cur = list(opening)
    for _ in range(max_plies):
        who, reason = atk.result_reason(cur)
        if who:
            return who, reason or "unknown", cur
        best, _ = (atk if len(cur) % 2 == 0 else dfn).search(cur, nodes)
        if best is None:
            return "draw", "no_bestmove", cur
        cur.append(best)
    return "draw", "max_plies", cur


def fort_rows(game: dict) -> list[dict]:
    """Attacker-to-move prefixes (even length) about k plies before the end."""
    moves = game["moves"].split()
    rows = []
    for k in KS:
        n = len(moves) - k
        n -= n % 2
        if n < game.get("opening", 0):
            continue
        rows.append({"kind": "fort", "k": k, "moves": " ".join(moves[:n]), "game": game["id"],
                     "attacker": game["attacker"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--attackers", default="")
    ap.add_argument("--defender", default="")
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--opening-moves", type=int, default=16)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--variant", default="copenhagen11x11")
    ap.add_argument("--rebuild", action="store_true", help="only rebuild fort.jsonl from fort_games.jsonl")
    args = ap.parse_args()

    games_path = os.path.join(BENCH_DIR, "fort_games.jsonl")
    bench_path = os.path.join(BENCH_DIR, "fort.jsonl")
    env = dict(os.environ, TAFLZERO_ORT_THREADS="1")
    lock = threading.Lock()

    def rebuild(ref: Engine) -> None:
        """Called under `lock`; `ref` is a defender engine for the ref_wr values."""
        games = [json.loads(l) for l in open(games_path, encoding="utf-8") if l.strip()] if os.path.exists(games_path) else []
        old = {}
        if os.path.exists(bench_path):
            for l in open(bench_path, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    old[r["moves"]] = r.get("ref_wr")
        rows = [r for g in games if g["reason"] == "def_fort" for r in fort_rows(g)]
        for r in rows:
            r["ref_wr"] = old.get(r["moves"])
            if r["ref_wr"] is None:
                r["ref_wr"] = ref.search(r["moves"].split(), 1)[1]
        tmp = bench_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, bench_path)
        forts = sum(g["reason"] == "def_fort" for g in games)
        print(f"{len(games)} games, {forts} forts, {len(rows)} positions", flush=True)

    if args.rebuild:
        ref = Engine(args.exe, args.defender, args.variant, env)
        try:
            with lock:
                rebuild(ref)
        finally:
            ref.close()
        return

    attackers = [a.strip() for a in args.attackers.split(",") if a.strip()]
    next_game = [0]
    rng = random.Random()
    start_id = 0
    if os.path.exists(games_path):
        start_id = sum(1 for l in open(games_path, encoding="utf-8") if l.strip())

    def worker() -> None:
        engines: dict[str, Engine] = {}

        def engine(p: str) -> Engine:
            if p not in engines:
                engines[p] = Engine(args.exe, p, args.variant, env)
            return engines[p]

        try:
            while True:
                with lock:
                    if next_game[0] >= args.games:
                        return
                    gid = start_id + next_game[0]
                    next_game[0] += 1
                atk_path = attackers[gid % len(attackers)]
                atk, dfn = engine(atk_path), engine(args.defender)
                opening = make_opening(atk, args.opening_moves, rng)
                who, reason, moves = play_recorded(atk, dfn, opening, args.nodes, args.max_plies)
                game = {"id": gid, "attacker": os.path.splitext(os.path.basename(atk_path))[0],
                        "defender": os.path.splitext(os.path.basename(args.defender))[0], "nodes": args.nodes,
                        "opening": len(opening), "winner": who, "reason": reason, "plies": len(moves),
                        "moves": " ".join(moves)}
                with lock:
                    with open(games_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(game) + "\n")
                    print(f"game {gid}: {game['attacker']} attacks, {reason} after {len(moves)} plies", flush=True)
                    rebuild(dfn)
        finally:
            for e in engines.values():
                e.close()

    os.makedirs(BENCH_DIR, exist_ok=True)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, args.workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
