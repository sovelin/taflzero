"""Tiny AlphaZero-style network for 11x11 Tafl."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + residual
        x = self.relu(x)
        return x


class TaflAlphaZeroNet(nn.Module):
    """Input: (B, 6, 11, 11). Outputs: policy logits (B, 4840), value (B, 1)."""

    def __init__(self, in_channels: int = 6, trunk_channels: int = 8, num_blocks: int = 3) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.trunk = nn.Sequential(*[ResidualBlock(trunk_channels) for _ in range(num_blocks)])

        # Policy head: Conv1x1(8 -> 40), flatten 40 * 11 * 11 = 4840
        self.policy_head = nn.Conv2d(trunk_channels, 40, kernel_size=1, bias=True)

        # Value head: Conv1x1(8 -> 1), flatten 121 -> 64 -> 1 -> tanh
        self.value_conv = nn.Conv2d(trunk_channels, 1, kernel_size=1, bias=True)
        self.value_mlp = nn.Sequential(
            nn.Linear(11 * 11, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem(x)
        x = self.trunk(x)

        policy_logits = self.policy_head(x).flatten(start_dim=1)
        value = self.value_conv(x).flatten(start_dim=1)
        value = self.value_mlp(value)
        return policy_logits, value
