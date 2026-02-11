"""Export a PyTorch TaflAlphaZeroNet model to ONNX format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


def export_model_to_onnx(model: nn.Module, onnx_path: Path) -> None:
    model.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 6, 11, 11)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
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
