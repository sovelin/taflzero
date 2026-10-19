"""Blind-spot metrics of one net against the external benchmark positions.

    python zero-trainer/tools/eval_blind.py <taflzero.exe> <net.onnx> [--playout] [--out eval.jsonl]

Positions come from zero-trainer/bench/blindspot.jsonl (built once from his self-play
and the arena games; see docs/proxy-ladder.md for the provenance). Metrics, all from
the bare net (`go nodes 1`, attacker winrate in percent):

  brier_his  Brier score on ~4400 of his self-play positions, attacker to move.
             Random net 0.25, collapsed nets 0.54-0.67, T2 g0213 0.096, his 0.087.
  lost_pos   mean attacker winrate on 72 positions his engine proves lost for the
             attacker (truth <= 10%). Random 49.5, g0213 ~45, his ~5.
  won_pos    mean attacker winrate on 81 positions his net gives >= 90% and the
             attacker did win. Random 49.4, collapsed 6-25, his ~95. Read the pair
             lost_pos / won_pos: a net that always says "defenders win" fakes lost_pos.
  --playout  conv_def/conv_atk/conv_unf: 12 post-blunder positions played out by this
             engine against itself at 400 nodes, 200-ply cap. g0213 at 800 nodes: 7/2/3;
             his defence 12/0/0.

Appends one JSON line to --out (default: eval.jsonl next to the net).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uci_client import Engine  # noqa: E402

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "blindspot.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("net")
    ap.add_argument("--playout", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--label", default=None, help="value for the 'net' field (default: file name)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.bench, encoding="utf-8") if l.strip()]
    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)

    env = dict(os.environ, TAFLZERO_ORT_THREADS="1")
    eng = Engine(args.exe, args.net, env=env)
    out = {"net": args.label or os.path.basename(args.net), "time": time.strftime("%Y-%m-%d %H:%M")}
    try:
        se, n = 0.0, 0
        for r in by_kind.get("brier", []):
            _, wr = eng.search(r["moves"].split(), 1)
            if wr is None:
                continue
            se += (wr / 100 - (1.0 if r["atk_won"] else 0.0)) ** 2
            n += 1
        out["brier_his"] = round(se / max(n, 1), 4)

        won = [eng.search(r["moves"].split(), 1)[1] for r in by_kind.get("won", [])]
        won = [w for w in won if w is not None]
        out["won_pos"] = round(sum(won) / max(len(won), 1), 1)
        out["won_n"] = len(won)

        lost = [eng.search(r["moves"].split(), 1)[1] for r in by_kind.get("lost", [])]
        lost = [w for w in lost if w is not None]
        out["lost_pos"] = round(sum(lost) / max(len(lost), 1), 1)
        out["lost_n"] = len(lost)

        if args.playout:
            res = {"defenders": 0, "attackers": 0, "unfinished": 0}
            for r in by_kind.get("convert", []):
                cur = r["moves"].split()
                start_len = len(cur)
                outcome = None
                while len(cur) < start_len + 200:
                    outcome = eng.result(cur)
                    if outcome:
                        break
                    best, _ = eng.search(cur, 400)
                    if best is None:
                        break
                    cur.append(best)
                res[outcome or "unfinished"] += 1
            out["conv_def"], out["conv_atk"], out["conv_unf"] = res["defenders"], res["attackers"], res["unfinished"]
    finally:
        eng.close()

    print(json.dumps(out))
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.net)), "eval.jsonl")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
