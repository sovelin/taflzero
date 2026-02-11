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
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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


def train(
    model: TaflAlphaZeroNet,
    dataset: SelfPlayDataset,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> None:
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=True,
    )

    total_samples = len(dataset)
    print(f"Training: {steps} steps, {total_samples} samples, batch={batch_size}, lr={lr}")
    print(f"Device: {device}")
    print()

    running_loss = 0.0
    running_p = 0.0
    running_v = 0.0
    t0 = time.time()

    for step, (planes, legal_mask, pi_target, value_target) in enumerate(
        itertools.islice(infinite_dataloader(loader), steps), 1
    ):
        planes = planes.to(device)
        legal_mask = legal_mask.to(device)
        pi_target = pi_target.to(device)
        value_target = value_target.to(device)

        policy_logits, value_pred = model(planes)

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

        running_loss += total_loss.item()
        running_p += p_loss.item()
        running_v += v_loss.item()

        if step % LOG_INTERVAL == 0 or step == steps:
            dt = time.time() - t0
            avg_loss = running_loss / LOG_INTERVAL
            avg_p = running_p / LOG_INTERVAL
            avg_v = running_v / LOG_INTERVAL
            speed = LOG_INTERVAL * batch_size / dt

            print(
                f"Step {step:5d}/{steps}  "
                f"loss={avg_loss:.4f}  policy={avg_p:.4f}  value={avg_v:.4f}  "
                f"{speed:.0f} samples/s"
            )

            running_loss = 0.0
            running_p = 0.0
            running_v = 0.0
            t0 = time.time()


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

    # Auto-calculate steps: ~1 pass over the data
    steps = args.steps if args.steps > 0 else max(1, len(dataset) // args.batch)
    print(f"Steps: {steps}")

    # Train
    train(
        model,
        dataset,
        steps=steps,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
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


if __name__ == "__main__":
    main()
