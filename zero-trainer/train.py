"""
AlphaZero training pipeline for Tafl.

Usage:
    python train.py --data selfplay.bin --out weights/gen1.onnx
    python train.py --data selfplay.bin --checkpoint weights/gen0.qnxx --out weights/gen1.onnx
    python train.py --data selfplay.bin --out weights/gen1.onnx --window 50000 --steps 1000 --batch 256
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from az_micro_net import TaflAlphaZeroNet
from dataset import SelfPlayDataset
from export_onnx import export_model_to_onnx
from qnxx_io import save_qnxx, load_qnxx
from training_utils import alpha_zero_loss

LOG_INTERVAL = 100  # print stats every N steps


def infinite_dataloader(loader: DataLoader):
    """Yield batches forever, reshuffling each pass."""
    while True:
        yield from loader


@torch.no_grad()
def compute_val_loss(
    model: TaflAlphaZeroNet,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> tuple[float, float, float]:
    """Compute average validation loss over a few batches."""
    model.eval()
    total_loss = 0.0
    total_p = 0.0
    total_v = 0.0
    n = 0
    for planes, legal_mask, pi_target, value_target in itertools.islice(val_loader, max_batches):
        planes = planes.to(device)
        legal_mask = legal_mask.to(device)
        pi_target = pi_target.to(device)
        value_target = value_target.to(device)

        policy_logits, value_pred = model(planes)
        loss, p_loss, v_loss = alpha_zero_loss(
            policy_logits=policy_logits,
            value_pred=value_pred,
            pi_target=pi_target,
            value_target=value_target,
            legal_mask=legal_mask,
        )
        total_loss += loss.item()
        total_p += p_loss.item()
        total_v += v_loss.item()
        n += 1

    model.train()
    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, total_p / n, total_v / n


def compute_sample_weights(
    planes: torch.Tensor, value_target: torch.Tensor, defender_weight: float,
) -> torch.Tensor:
    """Compute per-sample weights: defender-win positions get lower weight.

    stm=0 → attackers, stm=1 → defenders.  value=+1 → stm won, -1 → stm lost.
    Defender win: (stm==0 & value<0) | (stm==1 & value>0).
    """
    stm = planes[:, 3, 0, 0]  # (B,) 0.0=attackers, 1.0=defenders
    # attacker-perspective value: positive if attackers won
    atk_value = value_target * (1.0 - 2.0 * stm)
    is_defender_win = atk_value < 0
    weights = torch.ones_like(value_target)
    weights[is_defender_win] = defender_weight
    return weights


def train(
    model: TaflAlphaZeroNet,
    train_dataset,
    val_dataset,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    defender_weight: float,
    early_stopping_patience: int,
    device: torch.device,
) -> dict:
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    total_samples = len(train_dataset)
    print(f"Training: {steps} steps, {total_samples} train / {len(val_dataset)} val samples, batch={batch_size}, lr={lr}")
    print(f"Device: {device}, defender_weight={defender_weight}")
    print()

    running_loss = 0.0
    running_p = 0.0
    running_v = 0.0
    t0 = time.time()
    best_val_loss = float("inf")
    best_state_dict = None
    best_step = 0
    final_step = 0
    final_val_loss = 0.0
    patience_counter = 0

    for step, (planes, legal_mask, pi_target, value_target) in enumerate(
        itertools.islice(infinite_dataloader(loader), steps), 1
    ):
        planes = planes.to(device)
        legal_mask = legal_mask.to(device)
        pi_target = pi_target.to(device)
        value_target = value_target.to(device)

        policy_logits, value_pred = model(planes)

        weights = compute_sample_weights(planes, value_target, defender_weight)
        total_loss, p_loss, v_loss = alpha_zero_loss(
            policy_logits=policy_logits,
            value_pred=value_pred,
            pi_target=pi_target,
            value_target=value_target,
            legal_mask=legal_mask,
            sample_weights=weights,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        running_p += p_loss.item()
        running_v += v_loss.item()

        if step % LOG_INTERVAL == 0 or step == steps:
            dt = time.time() - t0
            avg_loss = running_loss / LOG_INTERVAL
            avg_p = running_p / LOG_INTERVAL
            avg_v = running_v / LOG_INTERVAL
            speed = LOG_INTERVAL * batch_size / dt

            val_loss, val_p, val_v = compute_val_loss(model, val_loader, device)

            print(
                f"Step {step:5d}/{steps}  "
                f"loss={avg_loss:.4f}  policy={avg_p:.4f}  value={avg_v:.4f}  "
                f"val_loss={val_loss:.4f}  val_p={val_p:.4f}  val_v={val_v:.4f}  "
                f"{speed:.0f} samples/s"
            )

            # Early stopping check
            if early_stopping_patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping at step {step} (no improvement for {patience_counter} checks)")
                        model.load_state_dict(best_state_dict)
                        final_step = step
                        final_val_loss = val_loss
                        break
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

            # Track final values
            final_step = step
            final_val_loss = val_loss

            running_loss = 0.0
            running_p = 0.0
            running_v = 0.0
            t0 = time.time()

    # Restore best model if we tracked it
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    return {
        "best_step": best_step,
        "total_steps": final_step,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero training for Tafl")
    parser.add_argument("--data", type=Path, required=True, help="Path to binary self-play data")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Resume from .qnxx checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="Output ONNX model path")
    parser.add_argument("--save-checkpoint", type=Path, default=None, help="Save .qnxx checkpoint after training")
    parser.add_argument("--window", type=int, default=0, help="Sliding window size (0 = use all data)")
    parser.add_argument("--steps", type=int, default=0, help="Training steps (0 = auto: dataset_size / batch)")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    parser.add_argument("--defender-weight", type=float, default=0.25, help="Loss weight for defender-win samples (1.0 = no reweighting)")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Stop if val_loss doesn't improve for N checks (0 = disabled)")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load or create model
    if args.checkpoint and args.checkpoint.exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        model = load_qnxx(args.checkpoint, device=device)
    else:
        print("Creating new model")
        model = TaflAlphaZeroNet()

    # Load dataset
    print(f"Loading data: {args.data}")
    dataset = SelfPlayDataset(args.data, window_size=args.window)
    print(f"Loaded {len(dataset)} samples" + (f" (window={args.window})" if args.window > 0 else ""))

    if len(dataset) == 0:
        print("No samples found!", file=sys.stderr)
        sys.exit(1)

    # Split into train/val (90/10)
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Split: {train_size} train / {val_size} val")

    # Auto-calculate steps: ~1 pass over the training data
    steps = args.steps if args.steps > 0 else max(1, train_size // args.batch)
    print(f"Steps: {steps}")

    # Train
    training_summary = train(
        model,
        train_dataset,
        val_dataset,
        steps=steps,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        defender_weight=args.defender_weight,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
    )

    # Save outputs
    model.cpu()
    model.eval()

    export_model_to_onnx(model, args.out)
    print(f"\nSaved ONNX model: {args.out}")

    if args.save_checkpoint:
        save_qnxx(model, args.save_checkpoint)
        print(f"Saved checkpoint: {args.save_checkpoint}")

        # Save training summary
        summary_path = args.save_checkpoint.with_suffix('.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        print(f"Saved training summary: {summary_path}")


if __name__ == "__main__":
    main()
