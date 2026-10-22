"""Convert ONNX model from FP32 to FP16.

Usage:
    python zero-trainer/convert_fp16.py --input weights/gen0360.onnx --output weights/gen0360_fp16.onnx
    python zero-trainer/convert_fp16.py --input weights/gen0360.onnx  # saves as gen0360_fp16.onnx automatically
"""

import argparse
import numpy as np
from pathlib import Path

import onnx

from export_onnx import get_planes_metadata, to_fp16


def convert(input_path: Path, output_path: Path) -> None:
    print(f"Loading: {input_path}")
    model = onnx.load(str(input_path))

    print("Converting to FP16 (keep_io_types=True)...")
    # to_fp16 carries the `planes` metadata over: the engine needs it to load the net.
    model_fp16 = to_fp16(model)
    print(f"planes:  {get_planes_metadata(model_fp16) or '(none: an 11-plane base net)'}")

    onnx.save(model_fp16, str(output_path))
    print(f"Saved:   {output_path}")


def validate(fp32_path: Path, fp16_path: Path, n_samples: int = 16) -> None:
    import onnxruntime as ort

    print("\nValidating outputs...")
    channels = onnx.load(str(fp32_path)).graph.input[0].type.tensor_type.shape.dim[1].dim_value or 11
    s32 = ort.InferenceSession(str(fp32_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    s16 = ort.InferenceSession(str(fp16_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    rng = np.random.default_rng(42)
    policy_diffs = []
    value_diffs = []

    for _ in range(n_samples):
        x = rng.standard_normal((1, channels, 11, 11)).astype(np.float32)
        p32, v32 = s32.run(None, {"input": x})
        p16, v16 = s16.run(None, {"input": x})
        policy_diffs.append(np.abs(p32 - p16).max())
        value_diffs.append(np.abs(v32 - v16).max())

    print(f"  policy diff  max={max(policy_diffs):.4f}  mean={np.mean(policy_diffs):.4f}")
    print(f"  value  diff  max={max(value_diffs):.4f}  mean={np.mean(value_diffs):.4f}")

    ok = max(policy_diffs) < 2.0 and max(value_diffs) < 0.1
    print(f"  {'OK' if ok else 'LARGE DIFF - check manually'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_stem(args.input.stem + "_fp16")

    convert(args.input, args.output)

    if not args.no_validate:
        validate(args.input, args.output)


if __name__ == "__main__":
    main()
