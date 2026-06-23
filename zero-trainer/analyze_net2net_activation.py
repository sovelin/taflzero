"""Check whether net2net-added capacity actually activated during training.

net2net leaves two fingerprints we can measure directly:
  1. Deepening: the new residual blocks were initialized as identity (conv2 = 0).
     If capacity is unused, their conv2 stays near-zero after training.
  2. Widening: new channels were duplicates of old ones (seed-42 mapping).
     If unused, each (original, duplicate) pair stays near-identical.

Compares the pristine net2net checkpoint against a trained one to see how much
the added capacity moved.  If the new blocks' conv2 norm is still tiny and the
duplicated channels haven't diverged, the LR was too low to break symmetry.

Usage:
    python analyze_net2net_activation.py --base weights/T2/gen0157_10x128.onxx --trained weights/T2/gen0169.onxx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from qnxx_io import load_qnxx
from net2net import _wider_mapping


def conv2_norms(net) -> list[float]:
    """L2 norm of conv2 weight per residual block (identity blocks start at 0)."""
    return [float(block.conv2.weight.norm()) for block in net.trunk]


def block_movement(base, trained) -> list[float]:
    """How much each block's total weights moved base -> trained (relative)."""
    out = []
    for b0, b1 in zip(base.trunk, trained.trunk):
        num, den = 0.0, 0.0
        for (n, p0), (_, p1) in zip(b0.named_parameters(), b1.named_parameters()):
            num += float((p1.detach() - p0.detach()).norm() ** 2)
            den += float(p0.detach().norm() ** 2)
        out.append((num ** 0.5) / (den ** 0.5 + 1e-9))
    return out


def duplicate_channel_divergence(trained, old_ch: int, new_ch: int, seed: int = 42):
    """For the stem conv output channels, measure how different each duplicated
    channel is from the original it was copied from. 0 = still identical."""
    mapping, _ = _wider_mapping(old_ch, new_ch, seed)
    w = trained.stem[0].weight.detach()  # (new_ch, in, k, k)
    divs = []
    for new_i in range(old_ch, new_ch):
        src = mapping[new_i]           # original channel this was a copy of
        a = w[new_i].flatten()
        b = w[src].flatten()
        # relative L2 distance between the (now trained) copy and its source
        divs.append(float((a - b).norm() / (b.norm() + 1e-9)))
    return divs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="pristine net2net checkpoint")
    ap.add_argument("--trained", type=Path, required=True, help="checkpoint after training")
    args = ap.parse_args()

    base = load_qnxx(args.base).eval()
    trained = load_qnxx(args.trained).eval()

    old_blocks = None
    # infer how many blocks were inherited: identity blocks in base have conv2 norm ~0
    base_c2 = conv2_norms(base)
    n_blocks = len(base_c2)
    # inherited blocks have nonzero conv2 in base; new (identity) blocks ~0
    inherited = [i for i, v in enumerate(base_c2) if v > 1e-4]
    new_blocks = [i for i, v in enumerate(base_c2) if v <= 1e-4]

    print(f"Base:    {args.base.name}   ({n_blocks} blocks)")
    print(f"Trained: {args.trained.name}")
    print(f"Inherited blocks: {inherited}   New (identity) blocks: {new_blocks}")
    print()

    # 1. conv2 norm: did the new identity blocks come alive?
    tr_c2 = conv2_norms(trained)
    print("=== conv2 L2 norm per block (identity blocks start at 0) ===")
    print(f"{'block':>6} | {'base':>9} | {'trained':>9} | {'type':>10}")
    for i in range(n_blocks):
        typ = "inherited" if i in inherited else "NEW"
        print(f"{i:>6} | {base_c2[i]:>9.4f} | {tr_c2[i]:>9.4f} | {typ:>10}")
    inh_mean = np.mean([tr_c2[i] for i in inherited]) if inherited else 0.0
    new_mean = np.mean([tr_c2[i] for i in new_blocks]) if new_blocks else 0.0
    print(f"\n  mean conv2 norm  inherited={inh_mean:.4f}  NEW={new_mean:.4f}  "
          f"ratio NEW/inherited={new_mean/(inh_mean+1e-9):.3f}")
    print("  (ratio near 0 = new depth capacity NOT activated; near 1 = fully used)")
    print()

    # 2. how much did each block move overall?
    mv = block_movement(base, trained)
    print("=== relative weight movement base->trained per block ===")
    for i in range(n_blocks):
        typ = "inherited" if i in inherited else "NEW"
        print(f"{i:>6} | rel.move={mv[i]:>7.3f} | {typ:>10}")
    print()

    # 3. duplicated stem channels: did copies diverge from originals?
    old_ch = base.stem[0].in_channels  # not it; need old trunk width
    old_width = min(inherited_width(base), trained.stem[0].out_channels)
    new_width = trained.stem[0].out_channels
    # old trunk width = number of channels that were "identity" in the mapping.
    # Infer from model_kwargs if available.
    old_w = infer_old_width(base, trained)
    if old_w is not None and old_w < new_width:
        divs = duplicate_channel_divergence(trained, old_w, new_width)
        divs = np.array(divs)
        print(f"=== duplicated stem channels ({old_w}->{new_width}): divergence from source ===")
        print(f"  n duplicated channels: {len(divs)}")
        print(f"  divergence  mean={divs.mean():.3f}  median={np.median(divs):.3f}  "
              f"min={divs.min():.3f}  max={divs.max():.3f}")
        print(f"  fraction still near-identical (<0.05): {float((divs < 0.05).mean()):.1%}")
        print("  (near 0 = copies never diverged = width capacity NOT used)")


def inherited_width(net):
    return net.stem[0].out_channels


def infer_old_width(base, trained):
    # The base IS the net2net output at full new width; we need the ORIGINAL
    # (pre-widening) width. Stored nowhere directly, so infer from the mapping
    # fingerprint: net2net divides duplicated input weights by replication count,
    # but simplest is to accept it as a CLI-known value. Default: assume 6x64->10x128.
    nw = trained.stem[0].out_channels
    # common case
    if nw == 128:
        return 64
    if nw == 96:
        return 64
    return None


if __name__ == "__main__":
    main()
