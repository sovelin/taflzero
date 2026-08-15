"""Export channel-to-channel wiring for the 3D visualizer.

For each residual block the influence of input channel i on output channel o
(through conv1 -> conv2, ignoring the nonlinearity/SE) is approximated by the
matmul of per-(out,in) kernel norms:  M = ||conv2|| @ ||conv1||, plus the
residual identity.  We keep the top-K input channels per output channel so the
JS side draws a readable graph (~top-8) instead of a 128x128 hairball.

Output viz/wiring.json:
  { "n_blocks", "topk", "channels",
    "gaps": [ {"o":[...], "i":[...], "w":[...]}, ... ] }   # one per block
Each gap g connects activation slab g (block input) to slab g+1 (block output).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from qnxx_io import load_qnxx

TOPK = 8
RESID = 0.5  # weight of the residual identity relative to the conv path


def kernel_norms(conv):
    w = conv.weight.detach()               # (out, in, k, k)
    return w.flatten(2).norm(dim=2).cpu().numpy()   # (out, in)


def main():
    net = load_qnxx("weights/T2/gen0195.onxx").eval()
    C = net.stem[0].out_channels
    gaps = []
    for blk in net.trunk:
        Kc1 = kernel_norms(blk.conv1)      # (hidden, in)
        Kc2 = kernel_norms(blk.conv2)      # (out, hidden)
        M = Kc2 @ Kc1                      # (out, in) first-order influence
        # normalize each row, add residual identity so "channel keeps itself"
        M = M / (M.max() + 1e-9)
        M += RESID * np.eye(C, dtype=M.dtype)
        o_idx, i_idx, w_val = [], [], []
        for o in range(C):
            top = np.argsort(-M[o])[:TOPK]
            for i in top:
                if M[o, i] <= 1e-6:
                    continue
                o_idx.append(int(o)); i_idx.append(int(i)); w_val.append(round(float(M[o, i]), 4))
        gaps.append({"o": o_idx, "i": i_idx, "w": w_val})
        print(f"block {len(gaps)-1}: {len(o_idx)} edges")

    out = {"n_blocks": len(net.trunk), "topk": TOPK, "channels": C, "gaps": gaps}
    p = Path("../viz/wiring.json")
    p.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {p}  ({sum(len(g['o']) for g in gaps)} edges total)")


if __name__ == "__main__":
    main()
