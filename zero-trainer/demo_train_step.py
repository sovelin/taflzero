from __future__ import annotations

import torch

from az_micro_net import TaflAlphaZeroNet
from training_utils import alpha_zero_loss, normalize_visit_counts


def make_random_batch(
    batch_size: int, device: torch.device, actions: int = 4840
) -> tuple[torch.Tensor, ...]:
    x = torch.rand(batch_size, 6, 11, 11, dtype=torch.float32, device=device)

    # Build legal mask with at least one legal action per sample.
    legal_mask = torch.rand(batch_size, actions, device=device) < 0.12
    legal_mask[:, 0] = True

    # Simulated MCTS visit counts over legal actions.
    visit_counts = torch.randint(
        0, 200, (batch_size, actions), dtype=torch.float32, device=device
    )
    visit_counts = visit_counts.masked_fill(~legal_mask, 0.0)
    visit_counts[:, 0] += 1.0

    pi_target = normalize_visit_counts(visit_counts, legal_mask)

    # Value target in [-1, 1] (game outcome from side-to-move perspective).
    value_target = torch.empty(batch_size, 1, device=device).uniform_(-1.0, 1.0)
    return x, legal_mask, pi_target, value_target


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TaflAlphaZeroNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x, legal_mask, pi_target, value_target = make_random_batch(batch_size=8, device=device)

    policy_logits, value_pred = model(x)
    total_loss, p_loss, v_loss = alpha_zero_loss(
        policy_logits=policy_logits,
        value_pred=value_pred,
        pi_target=pi_target,
        value_target=value_target,
        legal_mask=legal_mask,
    )

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    print("Forward/Train step completed")
    print(f"device: {device}")
    print(f"policy_logits shape: {tuple(policy_logits.shape)}")
    print(f"value_pred shape: {tuple(value_pred.shape)}")
    print(f"total_loss: {total_loss.item():.6f}")
    print(f"policy_loss: {p_loss.item():.6f}")
    print(f"value_loss: {v_loss.item():.6f}")


if __name__ == "__main__":
    main()
