from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from az_micro_net import TaflAlphaZeroNet


QNXX_FORMAT = "QNXX"
QNXX_VERSION = 1


def _make_payload(model: TaflAlphaZeroNet, seed: int | None = None) -> dict[str, Any]:
    # Infer architecture params from the model itself
    in_channels = model.stem[0].in_channels
    trunk_channels = model.stem[0].out_channels
    num_blocks = len(model.trunk)

    payload: dict[str, Any] = {
        "format": QNXX_FORMAT,
        "version": QNXX_VERSION,
        "model_class": "TaflAlphaZeroNet",
        "model_kwargs": {
            "in_channels": in_channels,
            "trunk_channels": trunk_channels,
            "num_blocks": num_blocks,
        },
        "state_dict": model.state_dict(),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    return payload


def save_qnxx(model: TaflAlphaZeroNet, output_path: Path, seed: int | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_make_payload(model, seed=seed), output_path)


def load_qnxx(path: Path, device: torch.device | str | None = None) -> TaflAlphaZeroNet:
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("QNXX checkpoint payload must be a dict")

    if payload.get("format") != QNXX_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format')!r}")

    if payload.get("version") != QNXX_VERSION:
        raise ValueError(f"Unsupported QNXX version: {payload.get('version')!r}")

    if payload.get("model_class") != "TaflAlphaZeroNet":
        raise ValueError(f"Unsupported model_class: {payload.get('model_class')!r}")

    model_kwargs = payload.get("model_kwargs") or {}
    model = TaflAlphaZeroNet(**model_kwargs)

    state_dict = payload.get("state_dict")
    if state_dict is None:
        raise ValueError("QNXX checkpoint has no state_dict")

    model.load_state_dict(state_dict)
    return model
