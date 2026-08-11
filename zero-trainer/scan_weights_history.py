"""Scan the WHOLE T2 run (4x48 -> 6x64 -> 10x128) and dump per-generation
metrics to JSON for the interactive HTML viewer.

Architecture detected by block count:
  4 blocks  -> 4x48   (from scratch; no net2net, no "new" blocks)
  6 blocks  -> 6x64   (net2net from 4x48 at g67; inherited 4, new 4-5; widen 48->64)
  10 blocks -> 10x128 (net2net from 6x64 at g157; inherited 6, new 6-9; widen 64->128)

Elo is stitched into one continuous curve from the chained anchor CSVs
(random -> g8 -> g67 -> g100 -> g157).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from qnxx_io import load_qnxx
from net2net import _wider_mapping

W = Path("weights/T2")
OUT = Path("weights_history.json")

# era config by block count -> (inherited_blocks, old_channels, new_channels)
ERA = {4: (4, None, None), 6: (4, 48, 64), 10: (6, 64, 128)}
# stitched anchor chain: (csv, anchor_gen_in_prev_segment or None for baseline)
ANCHORS = [
    ("anchor-results-random.csv", None),
    ("anchor-results-gen0008.csv", 8),
    ("anchor-results-gen0067.csv", 67),
    ("anchor-results-gen0100.csv", 100),
    ("anchor-results.csv", 157),
]


def stitched_elo():
    """Chain anchor segments into one continuous curve (baseline = vs random)."""
    raw = []
    for fn, _ in ANCHORS:
        seg = {}
        p = W / fn
        if p.exists():
            for row in csv.DictReader(open(p)):
                g = int(row["generation"].replace("gen", ""))
                seg[g] = float(row["elo_vs_anchor"])
        raw.append(seg)
    chained = {}
    offset = 0.0
    for i, (seg, (_, anchor_gen)) in enumerate(zip(raw, ANCHORS)):
        if anchor_gen is not None:
            offset = chained.get(anchor_gen, offset)
        for g, e in seg.items():
            chained[g] = e + offset
    return chained


def conv2_norms(net):
    return [float(b.conv2.weight.detach().norm()) for b in net.trunk]


def dupdiv(net, oc, nc):
    if oc is None:
        return None
    mp, _ = _wider_mapping(oc, nc, 42)
    w = net.stem[0].weight.detach()
    d = [float((w[i].flatten() - w[mp[i]].flatten()).norm() / (w[mp[i]].flatten().norm() + 1e-9))
         for i in range(oc, nc)]
    return round(float(np.mean(d)), 4)


def main():
    elo = stitched_elo()
    rows = []
    import glob, re
    gens = sorted(int(re.search(r"gen(\d+)\.onxx", f).group(1))
                  for f in glob.glob(str(W / "gen0*.onxx"))
                  if re.search(r"gen\d+\.onxx$", f))
    for g in gens:
        net = load_qnxx(W / f"gen{g:04d}.onxx").eval()
        nb = len(net.trunk)
        if nb not in ERA:
            continue
        inh, oc, nc = ERA[nb]
        c2 = conv2_norms(net)
        old_m = float(np.mean(c2[:inh]))
        new_m = float(np.mean(c2[inh:])) if nb > inh else None
        rows.append({
            "gen": g,
            "arch": {4: "4x48", 6: "6x64", 10: "10x128"}[nb],
            "old_c2": round(old_m, 2),
            "new_c2": round(new_m, 2) if new_m is not None else None,
            "ratio": round(new_m / old_m, 3) if new_m is not None else None,
            "dupdiv": dupdiv(net, oc, nc),
            "elo": round(elo[g], 1) if g in elo else None,
        })
        if g % 10 == 0 or g in (68, 158):
            print(f"g{g} [{rows[-1]['arch']}] elo={rows[-1]['elo']} ratio={rows[-1]['ratio']}")
    OUT.write_text(json.dumps({
        "transitions": [68, 158],  # net2net boundaries
        "data": rows,
    }, separators=(",", ":")))
    print(f"\nWrote {OUT}  ({len(rows)} generations, {gens[0]}-{gens[-1]})")


if __name__ == "__main__":
    main()
