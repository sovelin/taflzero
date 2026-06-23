"""Net2Net expansion utility for TaflAlphaZeroNet (T1 and T2 architectures).

Expands a trained network to a wider/deeper architecture while preserving
the function (Net2WiderNet + Net2DeeperNet).  Supports the T2 modules:
squeeze-excitation, global pooling bias, multi-channel value head, aux heads.

Rules:
  - use_se must match between source and target: an SE gate is multiplicative,
    so enabling it on a trained block cannot preserve the function.
  - Global pooling CAN be enabled on a source without it (--gpool-channels):
    the projection is zero-initialized, so the added bias starts at exactly 0.
  - Aux heads can be added freely (--add-aux-heads): they are training-only
    and never touch forward().

Usage:
    python net2net.py --input weights/T2/gen0350.onxx --channels 96 --blocks 12 --output weights/T2/gen0350_12x96.onxx
    python net2net.py --input ... --channels 96 --blocks 12 --gpool-channels 32 --output ... --onnx ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))

from az_micro_net import GlobalPoolBias, ResidualBlock, SqueezeExcite, TaflAlphaZeroNet
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


def _widen_linear_in(linear: nn.Linear, mapping, count) -> nn.Linear:
    """Widen Linear input features: duplicated inputs carry identical values,
    so their weight columns are divided by the replication count."""
    new_in = len(mapping)
    new_linear = nn.Linear(new_in, linear.out_features, bias=linear.bias is not None)
    with torch.no_grad():
        for new_i, old_i in enumerate(mapping):
            new_linear.weight[:, new_i] = linear.weight[:, old_i].clone() / count[old_i]
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias)
    return new_linear


def _widen_linear_out(linear: nn.Linear, mapping) -> nn.Linear:
    """Widen Linear output features by duplicating rows per mapping."""
    new_out = len(mapping)
    new_linear = nn.Linear(linear.in_features, new_out, bias=linear.bias is not None)
    with torch.no_grad():
        for new_i, old_i in enumerate(mapping):
            new_linear.weight[new_i] = linear.weight[old_i].clone()
            if linear.bias is not None:
                new_linear.bias[new_i] = linear.bias[old_i].clone()
    return new_linear


def _widen_se(se: SqueezeExcite, mapping, count, new_ch: int, seed: int) -> SqueezeExcite:
    """Widen SqueezeExcite: channel dims via mapping, hidden dim via its own mapping.

    Input to fc1 is per-channel spatial means — duplicated channels contribute
    identical values, handled like conv input widening.  fc2 output rows follow
    the channel mapping so duplicated channels receive identical gates.
    """
    new_se = SqueezeExcite(new_ch)
    old_hidden = se.fc1.out_features
    new_hidden = new_se.fc1.out_features
    assert new_hidden >= old_hidden, "SE hidden dim shrank — unsupported"
    h_mapping, h_count = _wider_mapping(old_hidden, new_hidden, seed + 1)

    with torch.no_grad():
        fc1 = _widen_linear_in(se.fc1, mapping, count)   # (old_h, new_ch)
        fc1 = _widen_linear_out(fc1, h_mapping)          # (new_h, new_ch)
        new_se.fc1 = fc1

        fc2 = _widen_linear_in(se.fc2, h_mapping, h_count)  # (old_ch, new_h)
        fc2 = _widen_linear_out(fc2, mapping)               # (new_ch, new_h)
        new_se.fc2 = fc2
    return new_se


def _widen_gpool(
    gpool: GlobalPoolBias | None,
    mapping,
    new_ch: int,
    old_pool: int,
    new_pool: int,
) -> GlobalPoolBias:
    """Widen (or freshly enable) GlobalPoolBias.

    The pooled input is [mean_0..mean_P-1, max_0..max_P-1].  The first old_ch
    channel indices are identity under our mapping, so pooled values for the
    first old_pool channels are unchanged — their columns transfer directly.
    Columns for newly pooled channels (old_pool..new_pool) are zero-initialized,
    as is the whole projection when the source block had no gpool: the added
    bias starts at exactly 0 and the function is preserved.
    """
    new_gpool = GlobalPoolBias(new_ch, new_pool)
    with torch.no_grad():
        new_gpool.proj.weight.zero_()
        new_gpool.proj.bias.zero_()
        if gpool is not None:
            assert new_pool >= old_pool, "gpool_channels shrank — unsupported"
            for new_i, old_i in enumerate(mapping):
                # mean block: columns [0, old_pool) → [0, old_pool)
                new_gpool.proj.weight[new_i, :old_pool] = gpool.proj.weight[old_i, :old_pool]
                # max block: columns [old_pool, 2*old_pool) → [new_pool, new_pool+old_pool)
                new_gpool.proj.weight[new_i, new_pool:new_pool + old_pool] = \
                    gpool.proj.weight[old_i, old_pool:2 * old_pool]
                new_gpool.proj.bias[new_i] = gpool.proj.bias[old_i]
    return new_gpool


def _widen_residual_block(
    block: ResidualBlock,
    mapping,
    count,
    new_ch: int,
    use_se: bool,
    gpool_channels: int,
    seed: int,
) -> ResidualBlock:
    """Widen a ResidualBlock from old_ch to new_ch, carrying SE/gpool over."""
    new_block = ResidualBlock(new_ch, use_se=use_se, gpool_channels=gpool_channels)
    with torch.no_grad():
        tmp = _widen_conv_out(block.conv1, mapping, new_ch)
        new_block.conv1 = _widen_conv_in(tmp, mapping, count, new_ch)
        new_block.bn1 = _widen_bn(block.bn1, mapping, new_ch)
        tmp = _widen_conv_out(block.conv2, mapping, new_ch)
        new_block.conv2 = _widen_conv_in(tmp, mapping, count, new_ch)
        new_block.bn2 = _widen_bn(block.bn2, mapping, new_ch)

        if use_se:
            assert block.se is not None, "target has SE but source block does not"
            new_block.se = _widen_se(block.se, mapping, count, new_ch, seed)

        if gpool_channels > 0:
            old_pool = block.gpool.pool_channels if block.gpool is not None else 0
            new_block.gpool = _widen_gpool(
                block.gpool, mapping, new_ch, old_pool, gpool_channels,
            )
    return new_block


# ── Net2DeeperNet helper ───────────────────────────────────────────────────────

def _identity_residual_block(channels: int, use_se: bool, gpool_channels: int) -> ResidualBlock:
    """Create a residual block that acts as identity (conv2 zeroed out).

    SE and gpool weights may stay randomly initialized: everything they touch
    flows through the zeroed conv2 path, so the block output is exactly the
    residual input regardless.
    """
    block = ResidualBlock(channels, use_se=use_se, gpool_channels=gpool_channels)
    with torch.no_grad():
        nn.init.zeros_(block.conv2.weight)
        block.bn2.weight.fill_(1.0)
        block.bn2.bias.zero_()
        block.bn2.running_mean.zero_()
        block.bn2.running_var.fill_(1.0)
        if block.se is not None:
            # gate applies to the zero conv2 path only — value irrelevant,
            # but zero fc2 makes the initial gate uniform (0.5) for stability
            nn.init.zeros_(block.se.fc2.weight)
            nn.init.zeros_(block.se.fc2.bias)
    return block


# ── Main expansion ─────────────────────────────────────────────────────────────

def net2net(
    old_net: TaflAlphaZeroNet,
    new_channels: int,
    new_blocks: int,
    new_gpool_channels: int | None = None,
    add_aux_heads: bool = False,
    seed: int = 42,
) -> TaflAlphaZeroNet:
    """Expand TaflAlphaZeroNet to new_channels/new_blocks preserving function."""
    old_kwargs = dict(old_net.model_kwargs) if hasattr(old_net, "model_kwargs") else {
        "in_channels": old_net.stem[0].in_channels,
        "trunk_channels": old_net.stem[0].out_channels,
        "num_blocks": len(old_net.trunk),
        "value_channels": 1,
        "use_se": False,
        "gpool_channels": 0,
        "aux_heads": False,
    }
    old_channels = old_kwargs["trunk_channels"]
    old_blocks = old_kwargs["num_blocks"]
    use_se = old_kwargs.get("use_se", False)
    old_gpool = old_kwargs.get("gpool_channels", 0)
    old_aux = old_kwargs.get("aux_heads", False)
    gpool_channels = old_gpool if new_gpool_channels is None else new_gpool_channels

    assert new_channels >= old_channels, f"new_channels ({new_channels}) must be >= old ({old_channels})"
    assert new_blocks >= old_blocks, f"new_blocks ({new_blocks}) must be >= old ({old_blocks})"
    assert gpool_channels >= old_gpool, f"gpool_channels ({gpool_channels}) must be >= old ({old_gpool})"
    assert gpool_channels <= old_channels, (
        "gpool_channels must not exceed OLD trunk width: pooled channels beyond "
        "it are not identity-mapped and function preservation would break"
    )

    print(f"  Widening:  {old_channels} → {new_channels} filters")
    print(f"  Deepening: {old_blocks} → {new_blocks} blocks")
    if gpool_channels != old_gpool:
        print(f"  GPool:     {old_gpool} → {gpool_channels} pooled channels (new columns zero-init)")

    mapping, count = _wider_mapping(old_channels, new_channels, seed)

    new_kwargs = dict(old_kwargs)
    new_kwargs.update(
        trunk_channels=new_channels,
        num_blocks=new_blocks,
        gpool_channels=gpool_channels,
        aux_heads=old_aux or add_aux_heads,
    )
    new_net = TaflAlphaZeroNet(**new_kwargs)

    def block_gpool(i: int) -> int:
        # constructor convention: global pooling on odd blocks
        return gpool_channels if i % 2 == 1 else 0

    with torch.no_grad():
        # ── Stem ──────────────────────────────────────────────────────────────
        new_net.stem[0] = _widen_conv_out(old_net.stem[0], mapping, new_channels)
        new_net.stem[1] = _widen_bn(old_net.stem[1], mapping, new_channels)

        # ── Trunk ─────────────────────────────────────────────────────────────
        widened = [
            _widen_residual_block(
                b, mapping, count, new_channels, use_se, block_gpool(i), seed,
            )
            for i, b in enumerate(old_net.trunk)
        ]
        identity = [
            _identity_residual_block(new_channels, use_se, block_gpool(i))
            for i in range(old_blocks, new_blocks)
        ]
        new_net.trunk = nn.Sequential(*(widened + identity))

        # ── Policy head ───────────────────────────────────────────────────────
        tmp = _widen_conv_out(old_net.policy_head[0], mapping, new_channels)
        new_net.policy_head[0] = _widen_conv_in(tmp, mapping, count, new_channels)
        new_net.policy_head[1] = _widen_bn(old_net.policy_head[1], mapping, new_channels)
        new_net.policy_head[3] = _widen_conv_in(old_net.policy_head[3], mapping, count, new_channels)

        # ── Value head (value_channels output unchanged, only input widens) ──
        new_net.value_head[0] = _widen_conv_in(old_net.value_head[0], mapping, count, new_channels)
        new_net.value_head[1].load_state_dict(old_net.value_head[1].state_dict())
        new_net.value_mlp.load_state_dict(old_net.value_mlp.state_dict())

        # ── Aux heads ─────────────────────────────────────────────────────────
        if old_aux:
            tmp = _widen_conv_out(old_net.aux_policy_head[0], mapping, new_channels)
            new_net.aux_policy_head[0] = _widen_conv_in(tmp, mapping, count, new_channels)
            new_net.aux_policy_head[1] = _widen_bn(old_net.aux_policy_head[1], mapping, new_channels)
            new_net.aux_policy_head[3] = _widen_conv_in(old_net.aux_policy_head[3], mapping, count, new_channels)
            new_net.aux_corner_head[0] = _widen_conv_in(old_net.aux_corner_head[0], mapping, count, new_channels)
            new_net.aux_corner_head[1].load_state_dict(old_net.aux_corner_head[1].state_dict())
            new_net.aux_corner_mlp.load_state_dict(old_net.aux_corner_mlp.state_dict())
        # if aux heads are newly added, constructor's random init is kept

    return new_net


# ── Verification ──────────────────────────────────────────────────────────────

def verify(old_net: TaflAlphaZeroNet, new_net: TaflAlphaZeroNet, tol: float = 1e-4) -> bool:
    """Check that outputs are preserved after expansion."""
    old_net.eval()
    new_net.eval()
    x = torch.randn(4, old_net.stem[0].in_channels, 11, 11)
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
    parser.add_argument("--channels", type=int, required=True, help="New channel width (e.g. 96)")
    parser.add_argument("--blocks", type=int, required=True, help="New block count (e.g. 12)")
    parser.add_argument("--gpool-channels", type=int, default=None,
                        help="New gpool pooled-channel count (default: keep source; may exceed it, new columns zero-init)")
    parser.add_argument("--add-aux-heads", action="store_true",
                        help="Add training-only aux heads (random init) if the source lacks them")
    parser.add_argument("--output", type=Path, required=True, help="Output .qnxx path")
    parser.add_argument("--onnx", type=Path, default=None, help="Also export ONNX to this path")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for channel mapping")
    parser.add_argument("--no-verify", action="store_true", help="Skip function preservation check")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    old_net = load_qnxx(args.input)
    print(f"  Architecture: {getattr(old_net, 'model_kwargs', 'legacy (pre-kwargs)')}")

    print(f"Expanding to {args.blocks}×{args.channels}...")
    new_net = net2net(
        old_net, args.channels, args.blocks,
        new_gpool_channels=args.gpool_channels,
        add_aux_heads=args.add_aux_heads,
        seed=args.seed,
    )

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
