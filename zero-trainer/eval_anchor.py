"""Side-car evaluator for a continuous run: rates every checkpoint against fixed anchors.

    python zero-trainer/eval_anchor.py --run-dir runs/T5 --engine-bin taflzero.exe
        --anchors copenhagen11x11.onnx,weights/P-A/gen0010.onnx [--pairs 50] [--nodes 400]
        [--workers 2] [--blind] [--playout-every 5] [--once]

The newest `ckpt/stepNNNNNN.onnx` not yet in `elo.jsonl` gets one paired match per anchor
(tools/match.py) and, with --blind, the blind-spot metrics (tools/eval_blind.py →
`eval.jsonl`). Fixed anchors keep checkpoints comparable across the whole run and
across runs; a chain of "new vs previous" would drift. Runs until STOP appears in the
run dir (or forever without a run dir of a live trainer); --once evaluates what is
there and exits. Uses two engine threads at most so the trainer's workers keep their
cores.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Console children of a windowless parent would each open a console window on Windows.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
TOOLS = HERE / "tools"


def load_done(path: Path, key: str) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)[key])
                except Exception:
                    pass
    return done


def step_of(ckpt: Path) -> int:
    m = re.search(r"step(\d+)", ckpt.stem)
    return int(m.group(1)) if m else -1


def generated_at(run_dir: Path, step: int) -> int | None:
    """Positions generated when `step` was reached: the checkpoint's own json, else the
    nearest metrics.csv row."""
    meta = run_dir / "ckpt" / f"step{step:06d}.json"
    if meta.exists():
        try:
            return int(json.loads(meta.read_text(encoding="utf-8"))["generated"])
        except Exception:
            pass
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return None
    best = None
    with open(metrics, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        try:
            si, gi = header.index("step"), header.index("generated")
        except ValueError:
            return None
        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= max(si, gi):
                continue
            try:
                s, g = int(float(parts[si])), int(float(parts[gi]))
            except ValueError:
                continue
            if best is None or abs(s - step) < abs(best[0] - step):
                best = (s, g)
    return best[1] if best else None


STOP_GRACE_S = 600.0


def stopped_for_good(run_dir: Path) -> bool:
    """A STOP file that has been there for a while. A restart (stop, change config,
    resume) also writes STOP for a minute or two and then removes it; the side-car
    must survive that, so only a STOP older than STOP_GRACE_S counts."""
    stop = run_dir / "STOP"
    try:
        return time.time() - stop.stat().st_mtime >= STOP_GRACE_S
    except OSError:
        return False


def write_phase(run_dir: Path, **state) -> None:
    """`eval.progress.json`: what the side-car is doing right now, for the dashboard.
    During a match tools/match.py rewrites the same file after every pair."""
    path = run_dir / "eval.progress.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"updated": time.time(), **state}), encoding="utf-8")
    os.replace(tmp, path)


def run_match(args, ckpt: Path, anchor: Path) -> dict:
    out = ckpt.with_suffix(f".vs-{anchor.stem}.json")
    run_dir = Path(args.run_dir)
    cmd = [sys.executable, str(TOOLS / "match.py"), "--exe", args.engine_bin, "--a", str(ckpt), "--b", str(anchor),
           "--pairs", str(args.pairs), "--nodes", str(args.nodes), "--workers", str(args.workers),
           "--opening-moves", str(args.opening_moves), "--max-plies", str(args.max_plies),
           "--variant", args.variant, "--json", str(out), "--quiet",
           "--progress", str(run_dir / "eval.progress.json")]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    result = json.loads(out.read_text(encoding="utf-8"))
    out.unlink(missing_ok=True)
    return result


def run_blind(args, ckpt: Path, playout: bool, out_path: Path) -> None:
    write_phase(Path(args.run_dir), phase="blind", a=ckpt.name, started=time.time())
    cmd = [sys.executable, str(TOOLS / "eval_blind.py"), args.engine_bin, str(ckpt), "--out", str(out_path),
           "--label", ckpt.stem]
    if playout:
        cmd.append("--playout")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine-bin", required=True)
    ap.add_argument("--anchors", required=True, help="comma-separated .onnx paths")
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--opening-moves", type=int, default=16)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--variant", default="copenhagen11x11")
    ap.add_argument("--blind", action="store_true", help="also run the blind-spot metrics")
    ap.add_argument("--playout-every", type=int, default=5, help="every Nth checkpoint also plays out the conversion positions")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_dir = run_dir / "ckpt"
    elo_path = run_dir / "elo.jsonl"
    eval_path = run_dir / "eval.jsonl"
    anchors = [Path(a.strip()).resolve() for a in args.anchors.split(",") if a.strip()]
    for a in anchors:
        if not a.exists():
            print(f"anchor missing: {a}", file=sys.stderr)
            return 2
    args.engine_bin = str(Path(args.engine_bin).resolve())
    log = open(run_dir / "eval.log", "a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    n_seen = 0
    while True:
        done_elo = load_done(elo_path, "key")
        done_blind = load_done(eval_path, "net")
        ckpts = sorted(ckpt_dir.glob("step*.onnx"), key=step_of) if ckpt_dir.exists() else []
        todo = [c for c in ckpts if any(f"{c.stem}|{a.stem}" not in done_elo for a in anchors)
                or (args.blind and c.stem not in done_blind)]
        if not todo:
            write_phase(run_dir, phase="idle")
            if args.once or stopped_for_good(run_dir):
                break
            time.sleep(args.poll)
            continue
        if stopped_for_good(run_dir):
            say("run stopped; leaving the remaining checkpoints unevaluated")
            break
        # Newest first: one evaluation takes longer than the gap between checkpoints,
        # so the oldest-first order falls hours behind the trainer. Skipped
        # checkpoints are simply never evaluated.
        ckpt = todo[-1]
        step = step_of(ckpt)
        gen = generated_at(run_dir, step)
        for anchor in anchors:
            key = f"{ckpt.stem}|{anchor.stem}"
            if key in done_elo:
                continue
            t0 = time.time()
            try:
                r = run_match(args, ckpt, anchor)
            except subprocess.CalledProcessError as e:
                say(f"match failed for {ckpt.stem} vs {anchor.stem}: {e}")
                continue
            row = {"key": key, "ckpt": ckpt.stem, "step": step, "generated": gen, "anchor": anchor.stem,
                   "pairs": r["pairs"], "w": r["w"], "l": r["l"], "d": r["d"], "score": r["score"],
                   "elo": r["elo"], "ci": r["ci"], "nodes": args.nodes, "seconds": round(time.time() - t0, 1),
                   "time": time.strftime("%Y-%m-%d %H:%M"), "reasons": r.get("reasons")}
            with open(elo_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            say(f"{ckpt.stem} vs {anchor.stem}: {r['w']}-{r['l']}-{r['d']} elo {r['elo']:+.0f} [{r['ci'][0]:+.0f}, {r['ci'][1]:+.0f}]")
        if args.blind and ckpt.stem not in done_blind:
            n_seen += 1
            try:
                run_blind(args, ckpt, args.playout_every > 0 and n_seen % args.playout_every == 1, eval_path)
                say(f"{ckpt.stem}: blind-spot metrics written")
            except subprocess.CalledProcessError as e:
                say(f"blind eval failed for {ckpt.stem}: {e}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
