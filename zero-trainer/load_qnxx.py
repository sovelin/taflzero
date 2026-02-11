from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qnxx_io import load_qnxx


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and validate a QNXX checkpoint.")
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=Path("weights/random_init.qnxx"),
        help="Path to QNXX checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for tensor mapping during load (example: cpu, cuda:0).",
    )
    args = parser.parse_args()

    model = load_qnxx(args.input_path, device=args.device)
    model = model.to(torch.device(args.device))
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded QNXX checkpoint: {args.input_path}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {num_params}")


if __name__ == "__main__":
    main()
