"""Re-export gen0135.onnx with corrected FP16 format."""

from pathlib import Path
import torch

from qnxx_io import load_qnxx
from export_onnx import export_model_to_onnx

# Load checkpoint
checkpoint_path = Path("weights/gen0135.onxx")
output_path = Path("weights/gen0135.onnx")

print(f"Loading checkpoint: {checkpoint_path}")
model = load_qnxx(checkpoint_path, device=torch.device("cpu"))

print(f"Re-exporting to: {output_path}")
export_model_to_onnx(model, output_path)

print("✓ Done!")
