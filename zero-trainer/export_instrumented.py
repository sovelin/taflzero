"""Export gen0191 as an *instrumented* ONNX that returns, in addition to
policy+value, the activation tensor after the stem and after every residual
block.  This is what the in-browser 3D visualizer consumes: one forward pass
yields the full per-layer, per-channel activation stack.

Exported in fp32 (onnxruntime-web WASM is unreliable on fp16).

Outputs (all batch=1):
    policy   (1, 4840)
    value    (1, 1)
    stem     (1, C, 11, 11)
    block0   (1, C, 11, 11)
    ...
    block9   (1, C, 11, 11)

Also dumps viz_meta.json: constant planes (throne/corners/edges) + board
geometry, so the JS side never re-derives them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from qnxx_io import load_qnxx
import dataset as D


class Instrumented(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        acts = []
        h = self.net.stem(x)
        acts.append(h)                    # stem
        for blk in self.net.trunk:
            h = blk(h)
            acts.append(h)                # block i
        policy = self.net._policy_from(self.net.policy_head, h)
        value = self.net._value_from(h)
        return (policy, value, *acts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", type=Path, default=Path("weights/T2/gen0191.onxx"))
    ap.add_argument("--out", type=Path, default=Path("../viz/net_instrumented.onnx"))
    args = ap.parse_args()

    net = load_qnxx(args.net).eval()
    n_blocks = len(net.trunk)
    C = net.stem[0].out_channels
    wrapped = Instrumented(net).eval()

    dummy = torch.zeros(1, D.NUM_PLANES, D.BOARD_SIZE, D.BOARD_SIZE)
    act_names = ["stem"] + [f"block{i}" for i in range(n_blocks)]
    out_names = ["policy", "value"] + act_names

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, dummy, str(args.out),
        input_names=["input"], output_names=out_names,
        opset_version=17, dynamic_axes=None,
    )
    print(f"exported {args.out}  ({n_blocks} blocks, {C} channels)")

    meta = {
        "channels": C,
        "n_blocks": n_blocks,
        "board": D.BOARD_SIZE,
        "num_planes": D.NUM_PLANES,
        "act_names": act_names,
        "throne": D._THRONE_PLANE.reshape(11, 11).astype(int).tolist(),
        "corners": D._CORNERS_PLANE.reshape(11, 11).astype(int).tolist(),
        "edges": D._EDGES_PLANE.reshape(11, 11).astype(int).tolist(),
        "plane_names": ["attackers", "defenders", "king", "stm", "throne",
                        "corners", "edges", "group_bfs", "king_bfs",
                        "rep1", "rep2"],
    }
    meta_path = args.out.parent / "viz_meta.json"
    meta_path.write_text(json.dumps(meta))
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
