"""Export a PyTorch TaflAlphaZeroNet model to ONNX format."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from torch import nn


def export_model_to_onnx(model: nn.Module, onnx_path: Path) -> None:
    model.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 6, 11, 11)

    # Export FP32 to a temp file, then convert to FP16 in-place
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        torch.onnx.export(
            model,
            dummy,
            str(tmp_path),
            input_names=["input"],
            output_names=["policy", "value"],
            dynamic_axes={
                "input": {0: "batch"},
                "policy": {0: "batch"},
                "value": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )

        import onnx
        from onnxconverter_common import float16

        fp32_model = onnx.load(str(tmp_path))
        fp16_model = float16.convert_float_to_float16(fp32_model, keep_io_types=True)
        onnx.save(fp16_model, str(onnx_path))

        # Also save FP32 version for WASM/tract (which doesn't support mixed FP16)
        fp32_path = onnx_path.with_suffix(".fp32.onnx")
        onnx.save(fp32_model, str(fp32_path))
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a random model to ONNX.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/random_init.onnx"),
        help="Output ONNX model path.",
    )
    args = parser.parse_args()

    from az_micro_net import TaflAlphaZeroNet

    model = TaflAlphaZeroNet()
    export_model_to_onnx(model, args.output)
    print(f"Exported ONNX model: {args.output}")


if __name__ == "__main__":
    main()
