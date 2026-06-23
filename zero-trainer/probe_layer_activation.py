"""Measure the real forward-pass contribution of each residual block.

conv2 weight norm tells you the block's *capacity*; this tells you how much the
block actually *moves the trunk* on real positions:  for each ResidualBlock,
the residual branch is  out = branch(x) + x, so the block's contribution is
||out - x|| relative to ||x||.  A block that does nothing has ratio ~0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from qnxx_io import load_qnxx
from dataset import SelfPlayDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("selfplay-t2.bin"))
    ap.add_argument("--n", type=int, default=512)
    args = ap.parse_args()

    net = load_qnxx(args.net).eval()

    ds = SelfPlayDataset(args.data, window_size=args.n * 4, augment=False)
    idx = np.linspace(0, len(ds) - 1, args.n).astype(int)
    planes = torch.stack([ds[i][0] for i in idx])  # (N, 11, 11, 11)

    rel = [[] for _ in net.trunk]      # ||out-x|| / ||x||
    absd = [[] for _ in net.trunk]     # ||out-x||
    hooks = []
    for bi, block in enumerate(net.trunk):
        def mk(bi):
            def hook(mod, inp, out):
                x = inp[0]
                d = (out - x).flatten(1).norm(dim=1)
                xn = x.flatten(1).norm(dim=1)
                rel[bi].append((d / (xn + 1e-9)).detach())
                absd[bi].append(d.detach())
            return hook
        hooks.append(block.register_forward_hook(mk(bi)))

    with torch.no_grad():
        net(planes)

    print(f"net={args.net.name}   batch={len(planes)} real positions\n")
    print(f"{'block':>5} | {'rel.contrib':>12} | {'abs delta':>10} | type")
    print("-" * 46)
    for bi in range(len(net.trunk)):
        r = torch.cat(rel[bi]).mean().item()
        a = torch.cat(absd[bi]).mean().item()
        typ = "inherited" if bi < 6 else "NEW"
        bar = "#" * int(r * 200)
        print(f"{bi:>5} | {r:>12.4f} | {a:>10.2f} | {typ:<9} {bar}")
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
