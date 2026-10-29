"""Optional side-car: puts every checkpoint of a run on one rating scale with cheap matches.

    python zero-trainer/eval_ladder.py --run-dir runs/T6 --engine-bin taflzero.exe
        [--anchors copenhagen11x11.onnx,...] [--nodes 20] [--pairs 4] [--opponents 8]
        [--workers 1] [--once]

eval_anchor.py rates each checkpoint against a few fixed anchors at 400 nodes; this one
plays short matches at a small node budget between checkpoints themselves and fits all
results at once (Bradley-Terry / Elo), so the whole run is one curve. A full round robin
is quadratic in the checkpoints, so each checkpoint plays `--opponents` others: its
nearest rated neighbours, then checkpoints spread over the run, plus every anchor.

Newest unfinished checkpoint first, then the older ones are filled in. Writes
`ladder.jsonl` (one line per match), `ladder.ratings.json` (the current fit),
`ladder.progress.json` (live match) and `ladder.log`. It never touches the trainer:
start and stop it at any time. A `LADDER_STOP` file in the run dir (or a run stopped
for good) ends it between matches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from eval_anchor import NO_WINDOW, generated_at, step_of, stopped_for_good

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"


def read_matches(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def fit_ratings(matches: list[dict], reference: str, iters: int = 2000) -> dict[str, tuple[float, float]]:
    """Elo per player from all games (draws = half a point), `reference` pinned at 0.
    Each pairing gets half a virtual game, drawn, so a 100% score stays finite
    (it pulls a pairing of 8 games about 6% toward even). Returns
    {name: (elo, standard error)}; players not connected to `reference` float freely
    around the mean."""
    score: dict[tuple[str, str], list[float]] = {}  # (a, b) -> [points of a, games]
    for m in matches:
        key = (m["a"], m["b"])
        s = score.setdefault(key, [0.0, 0.0])
        s[0] += m["w"] + 0.5 * m["d"]
        s[1] += m["w"] + m["l"] + m["d"]
    for s in score.values():
        s[0] += 0.25
        s[1] += 0.5
    players = sorted({p for k in score for p in k})
    # Minorise-maximise for Bradley-Terry (Hunter 2004): gamma_i = W_i / sum_j n_ij / (gamma_i + gamma_j),
    # monotone in the likelihood, so it converges where a plain Newton step oscillates.
    wins = {p: 0.0 for p in players}
    for (a, b), (pts, n) in score.items():
        wins[a] += pts
        wins[b] += n - pts
    games_of: dict[str, list[tuple[str, float]]] = {p: [] for p in players}  # p -> [(opponent, games)]
    for (a, b), (_, n) in score.items():
        games_of[a].append((b, n))
        games_of[b].append((a, n))
    gamma = {p: 1.0 for p in players}
    for _ in range(iters):
        new = {}
        for p in players:
            den = sum(n / (gamma[p] + gamma[q]) for q, n in games_of[p])
            new[p] = wins[p] / den if den > 0 else gamma[p]
        norm = math.exp(sum(math.log(g) for g in new.values()) / len(new))
        new = {p: g / norm for p, g in new.items()}
        delta = max(abs(math.log(new[p] / gamma[p])) for p in players)
        gamma = new
        if delta < 1e-7:
            break
    r = {p: 400 * math.log10(gamma[p]) for p in players}
    c = math.log(10) / 400
    shift = r.get(reference, sum(r.values()) / max(1, len(r)))
    se = {}
    for p in players:
        info = sum(c * c * n / ((1 + 10 ** ((r[q] - r[p]) / 400)) * (1 + 10 ** ((r[p] - r[q]) / 400)))
                   for q, n in games_of[p])
        se[p] = 1 / math.sqrt(info) if info > 0 else 999.0
    return {p: (r[p] - shift, 0.0 if p == reference else se[p]) for p in players}


def pick_opponents(ckpt: str, ckpts: list[str], anchors: list[str], rated: set[str], want: int,
                   played: set[str]) -> list[str]:
    """Anchors, the two nearest rated checkpoints, then rated checkpoints spread evenly
    over the run, up to `want` opponents in all (anchors included). Already played
    opponents count toward `want` and are not returned."""
    pool = [c for c in ckpts if c != ckpt and c in rated]
    chosen: list[str] = list(anchors)
    by_dist = sorted(pool, key=lambda c: abs(step_of(Path(c)) - step_of(Path(ckpt))))
    chosen += [c for c in by_dist[:2] if c not in chosen]
    rest = [c for c in pool if c not in chosen]
    slots = max(0, want - len(chosen))
    if rest and slots:
        idx = sorted({round(i * (len(rest) - 1) / max(1, slots - 1)) for i in range(slots)}) if slots > 1 else [len(rest) // 2]
        chosen += [rest[i] for i in idx]
    return [c for c in chosen[:max(want, len(anchors))] if c not in played]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine-bin", required=True)
    ap.add_argument("--anchors", default="", help="comma-separated .onnx paths, also rated; the first is pinned at 0")
    ap.add_argument("--nodes", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--opponents", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--opening-moves", type=int, default=16)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--variant", default="copenhagen11x11")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_dir = run_dir / "ckpt"
    matches_path = run_dir / "ladder.jsonl"
    ratings_path = run_dir / "ladder.ratings.json"
    progress_path = run_dir / "ladder.progress.json"
    anchors = {Path(a.strip()).resolve().stem: Path(a.strip()).resolve() for a in args.anchors.split(",") if a.strip()}
    for p in anchors.values():
        if not p.exists():
            print(f"anchor missing: {p}", file=sys.stderr)
            return 2
    engine = str(Path(args.engine_bin).resolve())
    log = open(run_dir / "ladder.log", "a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def external() -> set[str]:
        """Opponents another tool plays against this run's checkpoints and appends to
        ladder.jsonl (an engine this side-car cannot run): rated as anchors, never
        scheduled here. Listed in ladder.external.json as {"anchors": [name, ...]}."""
        try:
            return set(json.loads((run_dir / "ladder.external.json").read_text(encoding="utf-8")).get("anchors", []))
        except (OSError, ValueError):
            return set()

    def write_ratings(matches: list[dict]) -> None:
        mine = [m for m in matches if m.get("nodes") == args.nodes]
        if not mine:
            return
        names = {m["a"] for m in mine} | {m["b"] for m in mine}
        reference = next((a for a in anchors if a in names), None) or min(names, key=lambda n: step_of(Path(n)))
        fit = fit_ratings(mine, reference)
        games = {n: sum(m["w"] + m["l"] + m["d"] for m in mine if n in (m["a"], m["b"])) for n in fit}
        ext = external()
        players = []
        for n, (elo, se) in fit.items():
            is_anchor = n in anchors or n in ext
            step = None if is_anchor else step_of(Path(n))
            players.append({"name": n, "anchor": is_anchor, "step": step,
                            "generated": None if is_anchor else generated_at(run_dir, step),
                            "elo": round(elo, 1), "se": round(se, 1), "games": games[n]})
        players.sort(key=lambda p: (p["anchor"], p["step"] or 0))
        tmp = ratings_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"updated": time.time(), "nodes": args.nodes, "reference": reference,
                                   "players": players}), encoding="utf-8")
        os.replace(tmp, ratings_path)

    say(f"ladder: nodes {args.nodes}, {args.pairs} pairs x {args.opponents} opponents, anchors {list(anchors) or '-'}")
    write_ratings(read_matches(matches_path))
    while True:
        if (run_dir / "LADDER_STOP").exists() or stopped_for_good(run_dir):
            say("stopping")
            break
        # Matches against external opponents neither count as played nor make anyone rated here.
        ext = external()
        matches = [m for m in read_matches(matches_path) if m.get("nodes") == args.nodes
                   and m["a"] not in ext and m["b"] not in ext]
        ckpts = [c.stem for c in sorted(ckpt_dir.glob("step*.onnx"), key=step_of)] if ckpt_dir.exists() else []
        rated = {m["a"] for m in matches} | {m["b"] for m in matches} | set(anchors)
        if not rated:
            rated = set(ckpts)  # no anchors and nothing played yet: any checkpoint may start
        job = None
        for ckpt in reversed(ckpts):
            played = {m["b"] for m in matches if m["a"] == ckpt} | {m["a"] for m in matches if m["b"] == ckpt}
            todo = pick_opponents(ckpt, ckpts, list(anchors), rated, args.opponents, played)
            if todo and len(played) < max(args.opponents, len(anchors)):
                job = (ckpt, todo[0])
                break
        if job is None:
            progress_path.write_text(json.dumps({"phase": "idle", "updated": time.time()}), encoding="utf-8")
            if args.once:
                break
            time.sleep(args.poll)
            continue
        a, b = job
        path_a = ckpt_dir / f"{a}.onnx"
        path_b = anchors.get(b, ckpt_dir / f"{b}.onnx")
        out = run_dir / "ladder.match.json"
        cmd = [sys.executable, str(TOOLS / "match.py"), "--exe", engine, "--a", str(path_a), "--b", str(path_b),
               "--pairs", str(args.pairs), "--nodes", str(args.nodes), "--workers", str(args.workers),
               "--opening-moves", str(args.opening_moves), "--max-plies", str(args.max_plies),
               "--variant", args.variant, "--json", str(out), "--quiet",
               "--progress", str(progress_path), "--games-log", str(run_dir / "ladder.games.jsonl")]
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
            r = json.loads(out.read_text(encoding="utf-8"))
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            say(f"match failed for {a} vs {b}: {e}")
            time.sleep(args.poll)
            continue
        out.unlink(missing_ok=True)
        row = {"a": a, "b": b, "w": r["w"], "l": r["l"], "d": r["d"], "pairs": r["pairs"], "nodes": args.nodes,
               "seconds": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%d %H:%M"),
               "reasons": r.get("reasons")}
        with open(matches_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        say(f"{a} vs {b}: {r['w']}-{r['l']}-{r['d']} ({row['seconds']:.0f} s)")
        write_ratings(read_matches(matches_path))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
