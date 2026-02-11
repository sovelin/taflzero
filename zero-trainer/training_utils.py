"""Losses and helpers for tiny AlphaZero training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


NEG_LARGE = -1e9


def masked_policy_logits(policy_logits: Tensor, legal_mask: Tensor) -> Tensor:
    """
    Apply legal move mask to policy logits.

    Args:
        policy_logits: (B, 4840)
        legal_mask: (B, 4840), bool tensor where True means legal action.
    """
    if legal_mask.dtype is not torch.bool:
        raise TypeError("legal_mask must be torch.bool")

    if policy_logits.shape != legal_mask.shape:
        raise ValueError(
            f"shape mismatch: logits={policy_logits.shape}, legal_mask={legal_mask.shape}"
        )

    if not torch.all(legal_mask.any(dim=1)):
        raise ValueError("each sample must have at least one legal move")

    return policy_logits.masked_fill(~legal_mask, NEG_LARGE)


def normalize_visit_counts(visit_counts: Tensor, legal_mask: Tensor) -> Tensor:
    """
    Convert raw visit counts into pi_target distribution.

    Args:
        visit_counts: (B, 4840), non-negative counts for moves.
        legal_mask: (B, 4840), bool tensor.
    """
    if visit_counts.shape != legal_mask.shape:
        raise ValueError(
            f"shape mismatch: visit_counts={visit_counts.shape}, legal_mask={legal_mask.shape}"
        )

    if torch.any(visit_counts < 0):
        raise ValueError("visit_counts must be non-negative")

    pi = visit_counts.clone()
    pi = pi.masked_fill(~legal_mask, 0.0)

    denom = pi.sum(dim=1, keepdim=True)
    if torch.any(denom <= 0):
        raise ValueError("sum of legal visit counts must be > 0 for each sample")
    pi = pi / denom
    return pi


def policy_loss(policy_logits: Tensor, pi_target: Tensor, legal_mask: Tensor) -> Tensor:
    """
    Cross-entropy with soft target pi_target:
        -sum_a pi_target(a) * log softmax(masked_logits)_a
    """
    if policy_logits.shape != pi_target.shape:
        raise ValueError(
            f"shape mismatch: logits={policy_logits.shape}, pi_target={pi_target.shape}"
        )
    masked_logits = masked_policy_logits(policy_logits, legal_mask)
    log_probs = F.log_softmax(masked_logits, dim=1)
    return -(pi_target * log_probs).sum(dim=1).mean()


def value_loss(value_pred: Tensor, value_target: Tensor) -> Tensor:
    """
    MSE loss for value head.

    Args:
        value_pred: (B, 1)
        value_target: (B, 1) or (B,)
    """
    if value_target.ndim == 1:
        value_target = value_target.unsqueeze(1)
    return F.mse_loss(value_pred, value_target)


def alpha_zero_loss(
    policy_logits: Tensor,
    value_pred: Tensor,
    pi_target: Tensor,
    value_target: Tensor,
    legal_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    p_loss = policy_loss(policy_logits, pi_target, legal_mask)
    v_loss = value_loss(value_pred, value_target)
    total = p_loss + v_loss
    return total, p_loss, v_loss
