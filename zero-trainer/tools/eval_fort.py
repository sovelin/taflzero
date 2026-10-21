"""Fort metric: does a net see a defenders' fort coming?

    python zero-trainer/tools/eval_fort.py <taflzero.exe> <net.onnx>[,<net.onnx>...] [--out fort.jsonl]

Positions come from bench/fort.jsonl (tools/build_fort_bench.py): attacker to move, k
plies before a strong defender (g0213) completed a fort, so the truth is "defenders
win". All values are the bare net's (`go nodes 1`, attacker winrate in percent):

  fort_k<k>  mean attacker winrate k plies before the fort. Lower = the net sees it.
  fort_near  mean over k <= 8;  fort_far  mean over k >= 24.
  fort_ref   mean |winrate - g0213's own winrate| on the same positions (0 = agrees
             with the reference; g0213 itself scores 0).
  won_pos    the guard from eval_blind: mean attacker winrate on positions the
             attacker did win (high is right). A net that always says "defenders win"
             fakes a low fort_*; read the two together.

One JSON line per net is appended to --out (default: fort.jsonl next to the first net).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uci_client import Engine  # noqa: E402

BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench")
FORT = os.path.join(BENCH_DIR, "fort.jsonl")
BLIND = os.path.join(BENCH_DIR, "blindspot.jsonl")


def load(path: str, kind: str | None = None) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [r for r in rows if kind is None or r["kind"] == kind]


def fort_metrics(eng: Engine, rows: list[dict]) -> dict:
    """The fort_* fields for one engine; {} without a bench."""
    by_k: dict[int, list[float]] = {}
    diff = []
    for r in rows:
        wr = eng.search(r["moves"].split(), 1)[1]
        if wr is None:
            continue
        by_k.setdefault(r["k"], []).append(wr)
        if r.get("ref_wr") is not None:
            diff.append(abs(wr - r["ref_wr"]))
    if not by_k:
        return {}
    mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    out = {f"fort_k{k}": mean(v) for k, v in sorted(by_k.items())}
    out["fort_near"] = mean([w for k, v in by_k.items() if k <= 8 for w in v])
    out["fort_far"] = mean([w for k, v in by_k.items() if k >= 24 for w in v])
    out["fort_ref"] = mean(diff)
    out["fort_n"] = sum(len(v) for v in by_k.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("nets", help="comma-separated .onnx paths")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bench", default=FORT)
    args = ap.parse_args()

    rows = load(args.bench)
    won = load(BLIND, "won")
    nets = [n.strip() for n in args.nets.split(",") if n.strip()]
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(nets[0])), "fort.jsonl")
    env = dict(os.environ, TAFLZERO_ORT_THREADS="1")
    for net in nets:
        eng = Engine(args.exe, net, env=env)
        try:
            out = {"net": os.path.splitext(os.path.basename(net))[0], "time": time.strftime("%Y-%m-%d %H:%M")}
            out.update(fort_metrics(eng, rows))
            w = [eng.search(r["moves"].split(), 1)[1] for r in won]
            w = [x for x in w if x is not None]
            out["won_pos"] = round(sum(w) / max(len(w), 1), 1)
        finally:
            eng.close()
        print(json.dumps(out), flush=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
