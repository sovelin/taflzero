"""Tiny AlphaZero-style network for 11x11 Tafl.

Default arguments reproduce the original T1 architecture exactly (same module
tree, same state_dict keys), so old .qnxx checkpoints keep loading.  The T2
extensions are opt-in via constructor flags:

  - value_channels > 1: widens the value-head bottleneck (T1 squeezed the whole
    position eval through a single 1x1 channel).
  - use_se: squeeze-excitation gate in every residual block.
  - gpool_channels > 0: KataGo-style global pooling bias on every odd block —
    mean+max over the board of the first `gpool_channels` channels is projected
    and added as a per-channel bias, giving every square global context.
  - aux_heads: training-only heads (opponent reply policy + king escape corner).
    They are NOT part of forward(), so ONNX export is unaffected.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        w = x.mean(dim=(2, 3))
        w = self.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class GlobalPoolBias(nn.Module):
    """Project mean+max of the first `pool_channels` channels into a
    per-channel bias for all channels (KataGo global pooling, simplified)."""

    def __init__(self, channels: int, pool_channels: int) -> None:
        super().__init__()
        self.pool_channels = pool_channels
        self.proj = nn.Linear(2 * pool_channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        g = x[:, : self.pool_channels]
        pooled = torch.cat([g.mean(dim=(2, 3)), g.amax(dim=(2, 3))], dim=1)
        bias = self.proj(pooled)
        return x + bias.unsqueeze(-1).unsqueeze(-1)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int = 8,
        use_se: bool = False,
        gpool_channels: int = 0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.gpool = GlobalPoolBias(channels, gpool_channels) if gpool_channels > 0 else None
        self.se = SqueezeExcite(channels) if use_se else None

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        if self.gpool is not None:
            x = self.gpool(x)
        x = self.bn2(self.conv2(x))
        if self.se is not None:
            x = self.se(x)
        x = x + residual
        x = self.relu(x)
        return x


class TaflAlphaZeroNet(nn.Module):
    """Input: (B, in_channels, 11, 11). Outputs: policy logits (B, 4840), value (B, 1)."""

    def __init__(
        self,
        in_channels: int = 11,
        trunk_channels: int = 48,
        num_blocks: int = 4,
        value_channels: int = 1,
        use_se: bool = False,
        gpool_channels: int = 0,
        aux_heads: bool = False,
    ) -> None:
        super().__init__()

        self.model_kwargs = {
            "in_channels": in_channels,
            "trunk_channels": trunk_channels,
            "num_blocks": num_blocks,
            "value_channels": value_channels,
            "use_se": use_se,
            "gpool_channels": gpool_channels,
            "aux_heads": aux_heads,
        }

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
        )

        # Global pooling on every odd block (1, 3, ...) so global context is
        # injected mid-trunk and refined by later local convolutions.
        self.trunk = nn.Sequential(*[
            ResidualBlock(
                trunk_channels,
                use_se=use_se,
                gpool_channels=gpool_channels if i % 2 == 1 else 0,
            )
            for i in range(num_blocks)
        ])

        # Policy head: Conv1x1 -> BN -> ReLU -> Conv1x1, flatten 40 * 11 * 11 = 4840
        self.policy_head = nn.Sequential(
            nn.Conv2d(trunk_channels, trunk_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(trunk_channels, 40, kernel_size=1, bias=True),
        )

        # Value head: Conv1x1 -> BN -> ReLU -> flatten -> MLP -> tanh
        self.value_head = nn.Sequential(
            nn.Conv2d(trunk_channels, value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(inplace=True),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(value_channels * 11 * 11, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # Training-only auxiliary heads: not referenced by forward(), so they
        # never appear in the exported ONNX graph.
        if aux_heads:
            # Opponent reply policy over the same 4840 action space
            self.aux_policy_head = nn.Sequential(
                nn.Conv2d(trunk_channels, trunk_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(trunk_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(trunk_channels, 40, kernel_size=1, bias=True),
            )
            # King escape corner: 4 corners + "no corner escape"
            self.aux_corner_head = nn.Sequential(
                nn.Conv2d(trunk_channels, 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True),
            )
            self.aux_corner_mlp = nn.Sequential(
                nn.Linear(2 * 11 * 11, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 5),
            )

    def _trunk_out(self, x: Tensor) -> Tensor:
        return self.trunk(self.stem(x))

    def _policy_from(self, head: nn.Module, x: Tensor) -> Tensor:
        # permute to square-major order: (B,40,11,11) -> (B,11,11,40) -> (B,4840)
        # matches Rust index: from_square * 40 + move_type
        return head(x).permute(0, 2, 3, 1).flatten(start_dim=1)

    def _value_from(self, x: Tensor) -> Tensor:
        return self.value_mlp(self.value_head(x).flatten(start_dim=1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._trunk_out(x)
        return self._policy_from(self.policy_head, x), self._value_from(x)

    def forward_with_aux(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Training forward: (policy, value, opponent_policy, corner_logits)."""
        if "aux_policy_head" not in self._modules:
            raise RuntimeError("model was built without aux_heads=True")
        x = self._trunk_out(x)
        policy = self._policy_from(self.policy_head, x)
        value = self._value_from(x)
        aux_policy = self._policy_from(self.aux_policy_head, x)
        corner = self.aux_corner_mlp(self.aux_corner_head(x).flatten(start_dim=1))
        return policy, value, aux_policy, corner
