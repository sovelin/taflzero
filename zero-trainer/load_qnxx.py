"""Load and validate an ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and validate an ONNX model.")
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=Path("weights/random_init.onnx"),
        help="Path to ONNX model.",
    )
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.input_path))

    print(f"Loaded ONNX model: {args.input_path}")
    print("Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Smoke test with random input
    dummy = np.random.randn(1, 6, 11, 11).astype(np.float32)
    policy, value = session.run(None, {"input": dummy})
    print(f"\nSmoke test passed:")
    print(f"  policy shape: {policy.shape}")
    print(f"  value shape: {value.shape}")


if __name__ == "__main__":
    main()
