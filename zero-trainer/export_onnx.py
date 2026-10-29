"""Export a PyTorch TaflAlphaZeroNet model to ONNX format."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from torch import nn


PLANES_KEY = "planes"


def model_planes(model: nn.Module) -> str:
    """The canonical plane list of a model: its recorded one, else `base` for the
    11-channel nets that predate plane lists."""
    kwargs = getattr(model, "model_kwargs", {})
    planes = kwargs.get("planes")
    if planes:
        return planes
    if kwargs.get("in_channels", 11) != 11:
        raise ValueError(f"model reads {kwargs['in_channels']} planes but records no plane list")
    return "base"


def set_planes_metadata(onnx_model, planes: str) -> None:
    """Write (or replace) the `planes` metadata the engine reads at load."""
    for prop in list(onnx_model.metadata_props):
        if prop.key == PLANES_KEY:
            onnx_model.metadata_props.remove(prop)
    entry = onnx_model.metadata_props.add()
    entry.key = PLANES_KEY
    entry.value = planes


def get_planes_metadata(onnx_model) -> str | None:
    for prop in onnx_model.metadata_props:
        if prop.key == PLANES_KEY:
            return prop.value
    return None


def to_fp16(onnx_model):
    """Mixed fp16 copy that keeps the metadata (the converter is not relied on for it)."""
    from onnxconverter_common import float16

    out = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
    planes = get_planes_metadata(onnx_model)
    if planes is not None:
        set_planes_metadata(out, planes)
    return out


def export_model_to_onnx(model: nn.Module, onnx_path: Path, fp16: bool = True) -> None:
    """Export to ONNX.  By default (fp16=True) `onnx_path` gets a mixed-fp16 graph
    and `<stem>.fp32.onnx` the fp32 one; with fp16=False only the fp32 graph is
    written, directly to `onnx_path` (the continuous trainer's CPU workers)."""
    model.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    board_size = getattr(model, "board_size", 11)
    in_channels = model.model_kwargs["in_channels"]
    dummy = torch.randn(1, in_channels, board_size, board_size)

    # A moves-left model returns a 3rd output; name it and give it a batch axis so
    # the exported graph gains the extra output only when the head is present.
    has_ml = getattr(model, "has_moves_left", False)
    output_names = ["policy", "value"] + (["moves_left"] if has_ml else [])
    dynamic_axes = {
        "input": {0: "batch"},
        "policy": {0: "batch"},
        "value": {0: "batch"},
    }
    if has_ml:
        dynamic_axes["moves_left"] = {0: "batch"}

    planes = model_planes(model)

    # Export FP32 to a temp file, then convert to FP16 in-place
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        torch.onnx.export(
            model,
            dummy,
            str(tmp_path),
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
        )

        import onnx

        fp32_model = onnx.load(str(tmp_path))
        set_planes_metadata(fp32_model, planes)
        if not fp16:
            onnx.save(fp32_model, str(tmp_path))
            shutil.move(str(tmp_path), str(onnx_path))
            return

        fp16_model = to_fp16(fp32_model)
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
    parser.add_argument("--board-size", type=int, default=11, help="Board side (11 = Hnefatafl, 9 = Tablut)")
    parser.add_argument("--channels", type=int, default=None, help="Trunk channels")
    parser.add_argument("--blocks", type=int, default=None, help="Residual blocks")
    parser.add_argument("--value-channels", type=int, default=None, help="Value head channels")
    parser.add_argument("--se", action="store_true", help="Squeeze-excitation blocks")
    parser.add_argument("--gpool-channels", type=int, default=None, help="Global pooling channels (0 = off)")
    parser.add_argument("--aux-heads", action="store_true", help="Build training-only aux heads")
    parser.add_argument("--save-checkpoint", type=Path, default=None, help="Also save a .qnxx checkpoint")
    parser.add_argument("--planes", type=str, default=None, help="Input plane list (dataset.parse_planes); sets in_channels and the ONNX `planes` metadata")
    parser.add_argument("--fp32-only", action="store_true", help="Write only the fp32 graph, to --output")
    args = parser.parse_args()

    from az_micro_net import TaflAlphaZeroNet

    kwargs = {"board_size": args.board_size}
    if args.channels is not None:
        kwargs["trunk_channels"] = args.channels
    if args.blocks is not None:
        kwargs["num_blocks"] = args.blocks
    if args.value_channels is not None:
        kwargs["value_channels"] = args.value_channels
    if args.se:
        kwargs["use_se"] = True
    if args.gpool_channels is not None:
        kwargs["gpool_channels"] = args.gpool_channels
    if args.aux_heads:
        kwargs["aux_heads"] = True
    if args.planes is not None:
        from dataset import parse_planes

        kwargs["planes"] = args.planes
        kwargs["in_channels"] = parse_planes(args.planes).num_planes

    model = TaflAlphaZeroNet(**kwargs)
    print(f"Model: {model.model_kwargs}")
    export_model_to_onnx(model, args.output, fp16=not args.fp32_only)
    print(f"Exported ONNX model: {args.output}")

    if args.save_checkpoint:
        from qnxx_io import save_qnxx
        save_qnxx(model, args.save_checkpoint)
        print(f"Saved checkpoint: {args.save_checkpoint}")


if __name__ == "__main__":
    main()
