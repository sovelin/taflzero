"""Save a randomly initialized Tafl network as an ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from az_micro_net import TaflAlphaZeroNet
from export_onnx import export_model_to_onnx


def save_random_onnx(output_path: Path, seed: int) -> None:
    torch.manual_seed(seed)
    model = TaflAlphaZeroNet()
    export_model_to_onnx(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a randomly initialized Tafl network in ONNX format."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("weights/random_init.onnx"),
        help="Output path for the ONNX model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic initialization.",
    )
    args = parser.parse_args()

    save_random_onnx(args.out, args.seed)
    print(f"Saved random ONNX model: {args.out}")


if __name__ == "__main__":
    main()
