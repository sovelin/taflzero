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
  - value_wdl: the value MLP ends in 3 logits (win, draw, loss for the side to move)
    instead of tanh; forward() still returns one scalar, p_win - p_loss, so the
    exported graph keeps its outputs (docs/plan-T9.md §4).
  - td_heads > 0: training-only short-term value heads (§3), tanh, one per horizon.
  - soft_policy: a training-only second policy head for the soft policy target (§7).
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
    """Input: (B, in_channels, N, N). Outputs: policy logits (B, N*N*4*(N-1)), value (B, 1).

    `board_size` is the side of the board (11 for Hnefatafl, 9 for Tablut). It sets the
    policy head width and the flattened size of the value/corner MLPs.
    """

    def __init__(
        self,
        in_channels: int = 11,
        board_size: int = 11,
        trunk_channels: int = 48,
        num_blocks: int = 4,
        value_channels: int = 1,
        use_se: bool = False,
        gpool_channels: int = 0,
        aux_heads: bool = False,
        moves_left: bool = False,
        planes: str | None = None,
        value_wdl: bool = False,
        td_heads: int = 0,
        soft_policy: bool = False,
    ) -> None:
        super().__init__()

        # The input plane list (dataset.parse_planes) travels with the weights and
        # into the ONNX metadata, so the engine knows what to feed the net. None =
        # an old checkpoint: `base` when in_channels is 11.
        if planes is not None:
            from dataset import parse_planes

            spec = parse_planes(planes)
            if spec.num_planes != in_channels:
                raise ValueError(f"planes {planes!r} are {spec.num_planes} channels, in_channels is {in_channels}")
            planes = spec.canonical

        self.board_size = board_size
        self.sqs = board_size * board_size
        # 4 directions x (board_size - 1) possible slide distances
        self.policy_channels = 4 * (board_size - 1)

        self.model_kwargs = {
            "in_channels": in_channels,
            "board_size": board_size,
            "trunk_channels": trunk_channels,
            "num_blocks": num_blocks,
            "value_channels": value_channels,
            "use_se": use_se,
            "gpool_channels": gpool_channels,
            "aux_heads": aux_heads,
            "moves_left": moves_left,
        }
        if planes is not None:
            self.model_kwargs["planes"] = planes
        # Recorded only when on, so the checkpoints of older configurations stay
        # byte-identical in their kwargs and loadable by older code.
        if value_wdl:
            self.model_kwargs["value_wdl"] = True
        if td_heads:
            self.model_kwargs["td_heads"] = int(td_heads)
        if soft_policy:
            self.model_kwargs["soft_policy"] = True
        self.value_wdl = bool(value_wdl)

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

        # Policy head: Conv1x1 -> BN -> ReLU -> Conv1x1, flattened to
        # policy_channels * board_size * board_size (4840 on an 11x11 board).
        self.policy_head = nn.Sequential(
            nn.Conv2d(trunk_channels, trunk_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(trunk_channels, self.policy_channels, kernel_size=1, bias=True),
        )

        # Value head: Conv1x1 -> BN -> ReLU -> flatten -> MLP -> tanh
        self.value_head = nn.Sequential(
            nn.Conv2d(trunk_channels, value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(inplace=True),
        )
        if value_wdl:
            self.value_mlp = nn.Sequential(
                nn.Linear(value_channels * self.sqs, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3),
            )
        else:
            self.value_mlp = nn.Sequential(
                nn.Linear(value_channels * self.sqs, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Tanh(),
            )

        # Moves-left head: predicts plies remaining to game end (scaled by
        # MOVES_LEFT_SCALE).  UNLIKE the aux heads this IS part of forward(), so
        # it appears in the exported ONNX as a third output the engine can read.
        if moves_left:
            self.moves_left_head = nn.Sequential(
                nn.Conv2d(trunk_channels, value_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(value_channels),
                nn.ReLU(inplace=True),
            )
            self.moves_left_mlp = nn.Sequential(
                nn.Linear(value_channels * self.sqs, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )

        # Training-only auxiliary heads: not referenced by forward(), so they
        # never appear in the exported ONNX graph.
        if aux_heads:
            # Opponent reply policy over the same action space
            self.aux_policy_head = nn.Sequential(
                nn.Conv2d(trunk_channels, trunk_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(trunk_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(trunk_channels, self.policy_channels, kernel_size=1, bias=True),
            )
            # King escape corner: 4 corners + "no corner escape"
            self.aux_corner_head = nn.Sequential(
                nn.Conv2d(trunk_channels, 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True),
            )
            self.aux_corner_mlp = nn.Sequential(
                nn.Linear(2 * self.sqs, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 5),
            )

        # Training-only short-term value heads: never referenced by forward().
        if td_heads:
            self.td_head = nn.Sequential(
                nn.Conv2d(trunk_channels, value_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(value_channels),
                nn.ReLU(inplace=True),
            )
            self.td_mlp = nn.Sequential(
                nn.Linear(value_channels * self.sqs, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, int(td_heads)),
                nn.Tanh(),
            )
        # Training-only soft policy head (same shape as the policy head).
        if soft_policy:
            self.soft_policy_head = nn.Sequential(
                nn.Conv2d(trunk_channels, trunk_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(trunk_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(trunk_channels, self.policy_channels, kernel_size=1, bias=True),
            )

    def _trunk_out(self, x: Tensor) -> Tensor:
        return self.trunk(self.stem(x))

    def _policy_from(self, head: nn.Module, x: Tensor) -> Tensor:
        # permute to square-major order: (B,C,N,N) -> (B,N,N,C) -> (B,N*N*C)
        # matches Rust index: from_square * policy_channels + move_type
        return head(x).permute(0, 2, 3, 1).flatten(start_dim=1)

    def _value_raw(self, x: Tensor) -> Tensor:
        """(B, 1) tanh value, or (B, 3) win/draw/loss logits for a WDL head."""
        return self.value_mlp(self.value_head(x).flatten(start_dim=1))

    @staticmethod
    def wdl_to_value(logits: Tensor) -> Tensor:
        """Expected score p_win - p_loss, (B, 1), from (B, 3) win/draw/loss logits."""
        p = torch.softmax(logits, dim=1)
        return (p[:, 0] - p[:, 2]).unsqueeze(1)

    def _value_from(self, x: Tensor) -> Tensor:
        raw = self._value_raw(x)
        return self.wdl_to_value(raw) if self.value_wdl else raw

    def _moves_left_from(self, x: Tensor) -> Tensor:
        return self.moves_left_mlp(self.moves_left_head(x).flatten(start_dim=1))

    @property
    def has_moves_left(self) -> bool:
        return "moves_left_head" in self._modules

    def enable_moves_left(self) -> None:
        """Graft a fresh moves-left head onto an existing (loaded) model so a
        mature checkpoint can gain MLH without retraining from scratch. No-op if
        the head already exists."""
        if self.has_moves_left:
            return
        tc = self.model_kwargs["trunk_channels"]
        vc = self.model_kwargs["value_channels"]
        self.moves_left_head = nn.Sequential(
            nn.Conv2d(tc, vc, kernel_size=1, bias=False),
            nn.BatchNorm2d(vc),
            nn.ReLU(inplace=True),
        )
        self.moves_left_mlp = nn.Sequential(
            nn.Linear(vc * self.sqs, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.model_kwargs["moves_left"] = True

    def forward(self, x: Tensor):
        """Inference forward. Returns (policy, value), plus moves_left as a third
        output when the model has the head — so the exported ONNX gains a third
        output only for MLH models (old models stay 2-output)."""
        x = self._trunk_out(x)
        policy = self._policy_from(self.policy_head, x)
        value = self._value_from(x)
        if self.has_moves_left:
            return policy, value, self._moves_left_from(x)
        return policy, value

    def forward_with_aux(self, x: Tensor):
        """Training forward: (policy, value, opponent_policy, corner_logits,
        moves_left_or_None). moves_left is None when the head is absent."""
        if "aux_policy_head" not in self._modules:
            raise RuntimeError("model was built without aux_heads=True")
        o = self.forward_train(x)
        return o["policy"], o["value"], o["aux_policy"], o["corner"], o["moves_left"]

    def forward_train(self, x: Tensor) -> dict:
        """Every head the model has, for the trainer. Keys: policy, value (B, 1 scalar),
        value_logits (B, 3, WDL head only), aux_policy, corner, moves_left, td (B, H),
        soft_policy; heads the model lacks are None."""
        x = self._trunk_out(x)
        raw = self._value_raw(x)
        m = self._modules
        return {
            "policy": self._policy_from(self.policy_head, x),
            "value": self.wdl_to_value(raw) if self.value_wdl else raw,
            "value_logits": raw if self.value_wdl else None,
            "aux_policy": self._policy_from(self.aux_policy_head, x) if "aux_policy_head" in m else None,
            "corner": (self.aux_corner_mlp(self.aux_corner_head(x).flatten(start_dim=1))
                       if "aux_corner_head" in m else None),
            "moves_left": self._moves_left_from(x) if self.has_moves_left else None,
            "td": self.td_mlp(self.td_head(x).flatten(start_dim=1)) if "td_head" in m else None,
            "soft_policy": self._policy_from(self.soft_policy_head, x) if "soft_policy_head" in m else None,
        }
