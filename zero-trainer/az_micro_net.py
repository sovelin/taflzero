"""Tiny AlphaZero-style network for 11x11 Tafl."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = self.relu(x)
        return x


class TaflAlphaZeroNet(nn.Module):
    """Input: (B, 6, 11, 11). Outputs: policy logits (B, 4840), value (B, 1)."""

    def __init__(self, in_channels: int = 6, trunk_channels: int = 32, num_blocks: int = 4) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
        )

        self.trunk = nn.Sequential(*[ResidualBlock(trunk_channels) for _ in range(num_blocks)])

        # Policy head: Conv1x1 -> BN -> ReLU -> Conv1x1, flatten 40 * 11 * 11 = 4840
        self.policy_head = nn.Sequential(
            nn.Conv2d(trunk_channels, trunk_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(trunk_channels, 40, kernel_size=1, bias=True),
        )

        # Value head: Conv1x1 -> BN -> ReLU -> flatten 121 -> 64 -> 1 -> tanh
        self.value_head = nn.Sequential(
            nn.Conv2d(trunk_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(11 * 11, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem(x)
        x = self.trunk(x)

        # permute to square-major order: (B,40,11,11) -> (B,11,11,40) -> (B,4840)
        # matches Rust index: from_square * 40 + move_type
        policy_logits = self.policy_head(x).permute(0, 2, 3, 1).flatten(start_dim=1)
        value = self.value_head(x).flatten(start_dim=1)
        value = self.value_mlp(value)
        return policy_logits, value
