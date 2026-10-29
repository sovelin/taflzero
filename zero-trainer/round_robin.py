"""Round robin between chosen nets at a small node budget, with a live standings file.

    python zero-trainer/round_robin.py --run-dir runs/T6 --engine-bin taflzero.exe
        --nets runs/T6/ckpt/step100000.onnx,copenhagen11x11.onnx,... [--nodes 20] [--pairs 2]
        [--workers 1]

Started and stopped from the dashboard. Every pairing plays `--pairs` pairs (one random
opening, both colours); the schedule goes round by round, so the table is meaningful long
before the end. After every game `<run>/rr/state.json` is rewritten: standings (Elo from
all games so far, copenhagen11x11 = 0 when it plays), the cross table, the games being
played and the last results with how they ended. Games go to `<run>/rr/games.jsonl`.
A `<run>/rr/STOP` file ends it after the games in progress.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tools"))
from eval_ladder import fit_ratings  # noqa: E402
from match import make_opening, play_game  # noqa: E402
from uci_client import Engine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine-bin", required=True)
    ap.add_argument("--nets", required=True, help="comma-separated .onnx paths")
    ap.add_argument("--nodes", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--opening-moves", type=int, default=16)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--variant", default="copenhagen11x11")
    args = ap.parse_args()

    out_dir = Path(args.run_dir).resolve() / "rr"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "STOP").unlink(missing_ok=True)
    nets = {Path(n).stem: str(Path(n).resolve()) for n in args.nets.split(",") if n.strip()}
    names = list(nets)
    pairings = list(itertools.combinations(names, 2))
    rng = random.Random()
    schedule = []
    for _ in range(args.pairs):
        rnd = pairings[:]
        rng.shuffle(rnd)
        schedule += rnd
    total = 2 * len(schedule)
    env = dict(os.environ, TAFLZERO_ORT_THREADS="1")

    lock = threading.Lock()
    cross = {a: {b: [0, 0, 0] for b in names if b != a} for a in names}  # a vs b: [w, l, d] for a
    recent: list[dict] = []
    current: dict[int, dict] = {}
    done = [0]
    started = time.time()
    status = ["running"]
    reference = "copenhagen11x11" if "copenhagen11x11" in nets else names[0]

    def write_state() -> None:
        """Called under `lock`."""
        matches = []
        for a, b in pairings:
            w, l, d = cross[a][b]
            if w + l + d:
                matches.append({"a": a, "b": b, "w": w, "l": l, "d": d})
        fit = fit_ratings(matches, reference) if matches else {}
        table = []
        for n in names:
            w = sum(cross[n][o][0] for o in cross[n])
            l = sum(cross[n][o][1] for o in cross[n])
            d = sum(cross[n][o][2] for o in cross[n])
            g = w + l + d
            elo, se = fit.get(n, (None, None))
            table.append({"name": n, "w": w, "l": l, "d": d, "games": g,
                          "score": round((w + 0.5 * d) / g, 3) if g else None,
                          "elo": None if elo is None else round(elo, 1), "se": None if se is None else round(se, 1)})
        table.sort(key=lambda r: (r["elo"] is None, -(r["elo"] or 0)))
        state = {"status": status[0], "started": started, "updated": time.time(), "nodes": args.nodes,
                 "pairs": args.pairs, "nets": names, "reference": reference, "games_done": done[0],
                 "games_total": total, "table": table, "cross": cross,
                 "current": list(current.values()), "recent": recent[-12:], "pid": os.getpid()}
        tmp = out_dir / "state.json.tmp"
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, out_dir / "state.json")

    with lock:
        write_state()
    queue = list(reversed(schedule))
    games_log = open(out_dir / "games.jsonl", "a", encoding="utf-8")

    def worker(wid: int) -> None:
        engines: dict[str, Engine] = {}

        def engine(n: str) -> Engine:
            if n not in engines:
                engines[n] = Engine(args.engine_bin, nets[n], args.variant, env)
            return engines[n]

        try:
            while True:
                with lock:
                    if not queue or (out_dir / "STOP").exists():
                        return
                    a, b = queue.pop()
                ea, eb = engine(a), engine(b)
                opening = make_opening(ea, args.opening_moves, rng)
                for atk, dfn, an, dn in ((ea, eb, a, b), (eb, ea, b, a)):
                    with lock:
                        current[wid] = {"attackers": an, "defenders": dn, "since": time.time()}
                        write_state()
                    who, reason, plies = play_game(atk, dfn, opening, args.nodes, args.max_plies)
                    winner = an if who == "attackers" else dn if who == "defenders" else None
                    row = {"attackers": an, "defenders": dn, "winner": winner, "reason": reason, "plies": plies,
                           "nodes": args.nodes, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                    with lock:
                        if winner is None:
                            cross[an][dn][2] += 1
                            cross[dn][an][2] += 1
                        else:
                            loser = dn if winner == an else an
                            cross[winner][loser][0] += 1
                            cross[loser][winner][1] += 1
                        done[0] += 1
                        recent.append(row)
                        current.pop(wid, None)
                        games_log.write(json.dumps(row) + "\n")
                        games_log.flush()
                        write_state()
        finally:
            for e in engines.values():
                e.close()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(max(1, args.workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with lock:
        status[0] = "stopped" if (out_dir / "STOP").exists() else "done"
        write_state()
    games_log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
