from __future__ import annotations

import argparse
from pathlib import Path

import torch

from az_micro_net import TaflAlphaZeroNet
from qnxx_io import save_qnxx


def save_random_qnxx(output_path: Path, seed: int) -> None:
    torch.manual_seed(seed)
    model = TaflAlphaZeroNet()
    save_qnxx(model, output_path, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a randomly initialized Tafl network checkpoint in .qnxx format."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("weights/random_init.qnxx"),
        help="Output path for the QNXX checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic initialization.",
    )
    args = parser.parse_args()

    save_random_qnxx(args.out, args.seed)
    print(f"Saved random QNXX checkpoint: {args.out}")


if __name__ == "__main__":
    main()
