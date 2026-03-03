"""Net2Net expansion utility for TaflAlphaZeroNet.

Expands a trained network to a wider/deeper architecture while preserving
the function (Net2WiderNet + Net2DeeperNet).

Usage:
    python net2net.py --input weights/gen0350.qnxx --channels 128 --blocks 10 --output weights/gen0350_10x128.qnxx
    python net2net.py --input weights/gen0350.qnxx --channels 128 --blocks 10 --output weights/gen0350_10x128.qnxx --onnx weights/gen0350_10x128.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))

from az_micro_net import TaflAlphaZeroNet, ResidualBlock
from qnxx_io import load_qnxx, save_qnxx
from export_onnx import export_model_to_onnx


# ── Net2WiderNet helpers ───────────────────────────────────────────────────────

def _wider_mapping(old_w: int, new_w: int, seed: int = 42):
    """Generate index mapping from new channels to old channels.

    First old_w indices are identity. Remaining (new_w - old_w) are random
    duplicates from [0, old_w). Returns (mapping, count) where count[i] is
    how many times old index i is reused.
    """
    rng = np.random.RandomState(seed)
    extra = new_w - old_w
    extra_idx = rng.randint(0, old_w, size=extra).tolist()
    mapping = list(range(old_w)) + extra_idx
    count = np.ones(old_w, dtype=np.float32)
    for idx in extra_idx:
        count[idx] += 1.0
    return mapping, count


def _widen_conv_out(conv: nn.Conv2d, mapping, new_out: int) -> nn.Conv2d:
    """Widen output channels: new_weight[i] = old_weight[mapping[i]]."""
    new_conv = nn.Conv2d(
        conv.in_channels, new_out,
        kernel_size=conv.kernel_size, padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        for new_i, old_i in enumerate(mapping):
            new_conv.weight[new_i] = conv.weight[old_i].clone()
        if conv.bias is not None:
            for new_i, old_i in enumerate(mapping):
                new_conv.bias[new_i] = conv.bias[old_i].clone()
    return new_conv


def _widen_conv_in(conv: nn.Conv2d, mapping, count, new_in: int) -> nn.Conv2d:
    """Widen input channels, dividing duplicated weights by replication count."""
    new_conv = nn.Conv2d(
        new_in, conv.out_channels,
        kernel_size=conv.kernel_size, padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        for new_i, old_i in enumerate(mapping):
            new_conv.weight[:, new_i] = conv.weight[:, old_i].clone() / count[old_i]
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _widen_bn(bn: nn.BatchNorm2d, mapping, new_ch: int) -> nn.BatchNorm2d:
    """Widen BatchNorm2d by duplicating stats/params for replicated channels."""
    new_bn = nn.BatchNorm2d(new_ch, eps=bn.eps, momentum=bn.momentum)
    with torch.no_grad():
        for new_i, old_i in enumerate(mapping):
            new_bn.weight[new_i] = bn.weight[old_i].clone()
            new_bn.bias[new_i] = bn.bias[old_i].clone()
            new_bn.running_mean[new_i] = bn.running_mean[old_i].clone()
            new_bn.running_var[new_i] = bn.running_var[old_i].clone()
    return new_bn


def _widen_residual_block(block: ResidualBlock, mapping, count, new_ch: int) -> ResidualBlock:
    """Widen a ResidualBlock from old_ch to new_ch."""
    new_block = ResidualBlock(new_ch)
    with torch.no_grad():
        # conv1: (old_ch→old_ch) → (new_ch→new_ch)
        tmp = _widen_conv_out(block.conv1, mapping, new_ch)
        new_block.conv1 = _widen_conv_in(tmp, mapping, count, new_ch)
        new_block.bn1 = _widen_bn(block.bn1, mapping, new_ch)
        # conv2: (old_ch→old_ch) → (new_ch→new_ch)
        tmp = _widen_conv_out(block.conv2, mapping, new_ch)
        new_block.conv2 = _widen_conv_in(tmp, mapping, count, new_ch)
        new_block.bn2 = _widen_bn(block.bn2, mapping, new_ch)
    return new_block


# ── Net2DeeperNet helper ───────────────────────────────────────────────────────

def _identity_residual_block(channels: int) -> ResidualBlock:
    """Create a residual block that acts as identity (conv2 zeroed out)."""
    block = ResidualBlock(channels)
    with torch.no_grad():
        nn.init.zeros_(block.conv2.weight)
        block.bn2.weight.fill_(1.0)
        block.bn2.bias.zero_()
        block.bn2.running_mean.zero_()
        block.bn2.running_var.fill_(1.0)
    return block


# ── Main expansion ─────────────────────────────────────────────────────────────

def net2net(old_net: TaflAlphaZeroNet, new_channels: int, new_blocks: int, seed: int = 42) -> TaflAlphaZeroNet:
    """Expand TaflAlphaZeroNet to new_channels/new_blocks preserving function."""
    old_channels = old_net.stem[0].out_channels
    old_blocks = len(old_net.trunk)

    assert new_channels >= old_channels, f"new_channels ({new_channels}) must be >= old ({old_channels})"
    assert new_blocks >= old_blocks, f"new_blocks ({new_blocks}) must be >= old ({old_blocks})"

    print(f"  Widening:  {old_channels} → {new_channels} filters")
    print(f"  Deepening: {old_blocks} → {new_blocks} blocks")

    mapping, count = _wider_mapping(old_channels, new_channels, seed)

    new_net = TaflAlphaZeroNet(
        in_channels=old_net.stem[0].in_channels,
        trunk_channels=new_channels,
        num_blocks=new_blocks,
    )

    with torch.no_grad():
        # ── Stem ──────────────────────────────────────────────────────────────
        new_net.stem[0] = _widen_conv_out(old_net.stem[0], mapping, new_channels)
        new_net.stem[1] = _widen_bn(old_net.stem[1], mapping, new_channels)
        # stem[2] is ReLU — no params

        # ── Trunk ─────────────────────────────────────────────────────────────
        widened = [_widen_residual_block(b, mapping, count, new_channels) for b in old_net.trunk]
        identity = [_identity_residual_block(new_channels) for _ in range(new_blocks - old_blocks)]
        new_net.trunk = nn.Sequential(*(widened + identity))

        # ── Policy head ───────────────────────────────────────────────────────
        # [0] Conv(old→old, 1×1) → Conv(new→new, 1×1)
        tmp = _widen_conv_out(old_net.policy_head[0], mapping, new_channels)
        new_net.policy_head[0] = _widen_conv_in(tmp, mapping, count, new_channels)
        # [1] BN(old) → BN(new)
        new_net.policy_head[1] = _widen_bn(old_net.policy_head[1], mapping, new_channels)
        # [2] ReLU — no params
        # [3] Conv(old→40, 1×1) → Conv(new→40, 1×1): only input changes
        new_net.policy_head[3] = _widen_conv_in(old_net.policy_head[3], mapping, count, new_channels)

        # ── Value head ────────────────────────────────────────────────────────
        # [0] Conv(old→1, 1×1) → Conv(new→1, 1×1): only input changes
        new_net.value_head[0] = _widen_conv_in(old_net.value_head[0], mapping, count, new_channels)
        # [1] BN(1) — output stays 1 channel, just copy
        new_net.value_head[1].weight.copy_(old_net.value_head[1].weight)
        new_net.value_head[1].bias.copy_(old_net.value_head[1].bias)
        new_net.value_head[1].running_mean.copy_(old_net.value_head[1].running_mean)
        new_net.value_head[1].running_var.copy_(old_net.value_head[1].running_var)
        # [2] ReLU — no params

        # ── Value MLP ─────────────────────────────────────────────────────────
        # Linear(121→64) and Linear(64→1): value head output is still (B,1,11,11)
        # → flatten to 121 → no change needed
        new_net.value_mlp[0].weight.copy_(old_net.value_mlp[0].weight)
        new_net.value_mlp[0].bias.copy_(old_net.value_mlp[0].bias)
        new_net.value_mlp[2].weight.copy_(old_net.value_mlp[2].weight)
        new_net.value_mlp[2].bias.copy_(old_net.value_mlp[2].bias)

    return new_net


# ── Verification ──────────────────────────────────────────────────────────────

def verify(old_net: TaflAlphaZeroNet, new_net: TaflAlphaZeroNet, tol: float = 1e-4) -> bool:
    """Check that outputs are preserved after expansion."""
    old_net.eval()
    new_net.eval()
    x = torch.randn(4, 6, 11, 11)
    with torch.no_grad():
        p_old, v_old = old_net(x)
        p_new, v_new = new_net(x)
    p_diff = (p_old - p_new).abs().max().item()
    v_diff = (v_old - v_new).abs().max().item()
    ok = p_diff < tol and v_diff < tol
    status = "✅" if ok else "⚠️ "
    print(f"  {status} Policy diff: {p_diff:.2e}  Value diff: {v_diff:.2e}  (tol={tol})")
    if not ok:
        print("     Function not preserved — check implementation!")
    return ok


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Net2Net expansion for TaflAlphaZeroNet")
    parser.add_argument("--input", type=Path, required=True, help="Input .qnxx checkpoint")
    parser.add_argument("--channels", type=int, required=True, help="New channel width (e.g. 128)")
    parser.add_argument("--blocks", type=int, required=True, help="New block count (e.g. 10)")
    parser.add_argument("--output", type=Path, required=True, help="Output .qnxx path")
    parser.add_argument("--onnx", type=Path, default=None, help="Also export ONNX to this path")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for channel mapping")
    parser.add_argument("--no-verify", action="store_true", help="Skip function preservation check")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    old_net = load_qnxx(args.input)
    old_ch = old_net.stem[0].out_channels
    old_bl = len(old_net.trunk)
    print(f"  Architecture: {old_bl}×{old_ch}")

    print(f"Expanding to {args.blocks}×{args.channels}...")
    new_net = net2net(old_net, args.channels, args.blocks, args.seed)

    if not args.no_verify:
        print("Verifying function preservation...")
        verify(old_net, new_net)

    print(f"Saving: {args.output}")
    save_qnxx(new_net, args.output)

    if args.onnx:
        print(f"Exporting ONNX: {args.onnx}")
        export_model_to_onnx(new_net, args.onnx)

    params_old = sum(p.numel() for p in old_net.parameters())
    params_new = sum(p.numel() for p in new_net.parameters())
    print(f"Parameters: {params_old:,} → {params_new:,} ({params_new/params_old:.1f}×)")
    print("Done!")


if __name__ == "__main__":
    main()
