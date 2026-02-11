from __future__ import annotations

import torch
import pytest

from az_micro_net import TaflAlphaZeroNet
from training_utils import NEG_LARGE, alpha_zero_loss, masked_policy_logits, normalize_visit_counts


def test_model_output_shapes() -> None:
    model = TaflAlphaZeroNet()
    x = torch.randn(4, 6, 11, 11)
    policy_logits, value = model(x)

    assert policy_logits.shape == (4, 4840)
    assert value.shape == (4, 1)


def test_masked_logits_block_illegal_moves() -> None:
    logits = torch.zeros(2, 4840)
    legal_mask = torch.zeros(2, 4840, dtype=torch.bool)
    legal_mask[:, :3] = True

    masked = masked_policy_logits(logits, legal_mask)
    probs = torch.softmax(masked, dim=1)

    assert torch.all(masked[:, 10:] == NEG_LARGE)
    assert torch.all(probs[:, 10:] == 0)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)


def test_alpha_zero_loss_is_finite() -> None:
    model = TaflAlphaZeroNet()
    x = torch.randn(3, 6, 11, 11)
    policy_logits, value_pred = model(x)

    legal_mask = torch.rand(3, 4840) < 0.1
    legal_mask[:, 0] = True

    visit_counts = torch.randint(0, 100, (3, 4840), dtype=torch.float32)
    visit_counts[:, 0] += 1
    pi_target = normalize_visit_counts(visit_counts, legal_mask)

    value_target = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32).unsqueeze(1)

    total, p_loss, v_loss = alpha_zero_loss(
        policy_logits=policy_logits,
        value_pred=value_pred,
        pi_target=pi_target,
        value_target=value_target,
        legal_mask=legal_mask,
    )

    assert torch.isfinite(total)
    assert torch.isfinite(p_loss)
    assert torch.isfinite(v_loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_model_forward_on_cuda() -> None:
    device = torch.device("cuda")
    model = TaflAlphaZeroNet().to(device)

    x = torch.randn(2, 6, 11, 11, device=device)
    policy_logits, value = model(x)

    assert policy_logits.is_cuda
    assert value.is_cuda
    assert policy_logits.shape == (2, 4840)
    assert value.shape == (2, 1)
