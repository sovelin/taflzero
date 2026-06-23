"""One-shot refresh of the browser visualizer (viz/) to a given generation.

Regenerates everything the 2D inspector and 3D map consume:
  viz/net.onnx        instrumented single-file model (stem+blocks+policy+value)
  viz/viz_meta.json   channels / layer names / constant planes
  viz/reference.json  golden planes+value (validated byte-for-byte vs the engine)
  viz/wiring.json     channel-to-channel influence graph
  index.html / inspect.html   version label bumped to the new gen

Usage:
    python refresh_viz.py 195          # or 0195, or gen0195
    python refresh_viz.py              # defaults to the latest gen*.onxx found
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

import dataset as D
from qnxx_io import load_qnxx
from export_instrumented import Instrumented
from make_reference import build_planes, decode_move
from export_wiring import kernel_norms, TOPK, RESID

WEIGHTS = Path("weights/T2")
VIZ = Path("../viz")
DEMO_FEN = "3aaaa4/8a2/7a3/a4d4a/a3ddd1k1a/aa1dd1dd1aa/a3d4da/a3ad1a3/3a1d5/5a5/5aa1a2 d"


def norm_gen(s: str | None) -> str:
    if s is None:
        gens = sorted(int(re.search(r"gen(\d+)\.onxx", f).group(1))
                      for f in glob.glob(str(WEIGHTS / "gen0*.onxx"))
                      if re.search(r"gen\d+\.onxx$", f))
        return f"{gens[-1]:04d}"
    return f"{int(''.join(ch for ch in s if ch.isdigit())):04d}"


def export_model(net, gen):
    wrapped = Instrumented(net).eval()
    dummy = torch.zeros(1, D.NUM_PLANES, D.BOARD_SIZE, D.BOARD_SIZE)
    act_names = ["stem"] + [f"block{i}" for i in range(len(net.trunk))]
    tmp = VIZ / "_tmp.onnx"
    VIZ.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapped, dummy, str(tmp), input_names=["input"],
                      output_names=["policy", "value"] + act_names, opset_version=18)
    onnx.save_model(onnx.load(str(tmp)), str(VIZ / "net.onnx"), save_as_external_data=False)
    for f in VIZ.glob("_tmp.onnx*"):
        f.unlink()
    C = net.stem[0].out_channels
    meta = {
        "channels": C, "n_blocks": len(net.trunk), "board": D.BOARD_SIZE,
        "num_planes": D.NUM_PLANES, "act_names": act_names,
        "throne": D._THRONE_PLANE.reshape(11, 11).astype(int).tolist(),
        "corners": D._CORNERS_PLANE.reshape(11, 11).astype(int).tolist(),
        "edges": D._EDGES_PLANE.reshape(11, 11).astype(int).tolist(),
        "plane_names": ["attackers", "defenders", "king", "stm", "throne", "corners",
                        "edges", "group_bfs", "king_bfs", "rep1", "rep2"],
    }
    (VIZ / "viz_meta.json").write_text(json.dumps(meta))
    print(f"  net.onnx + viz_meta.json  ({len(net.trunk)} blocks, {C} ch)")


def write_reference():
    planes = build_planes(DEMO_FEN)
    dump = VIZ / "nneval_mate.json"
    if dump.exists():
        eng = np.array(json.loads(dump.read_text())["input"], np.float32).reshape(1, 11, 11, 11)
        d = float(np.abs(eng - planes).max())
        print(f"  planes vs engine  max|delta| = {d}" + ("  ⚠ MISMATCH" if d > 1e-6 else "  ✓"))
    sess = ort.InferenceSession(str(VIZ / "net.onnx"), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"input": planes})
    res = dict(zip([o.name for o in sess.get_outputs()], outs))
    act_names = ["stem"] + [f"block{i}" for i in range(len([k for k in res if k.startswith("block")]))]
    ref = {
        "fen": DEMO_FEN, "input": planes.reshape(-1).tolist(),
        "value": float(res["value"][0][0]),
        "policy_top": [int(i) for i in np.argsort(-res["policy"][0])[:5]],
        "chan_mag": {n: np.linalg.norm(res[n][0].reshape(res[n][0].shape[0], -1), axis=1).tolist()
                     for n in act_names},
    }
    (VIZ / "reference.json").write_text(json.dumps(ref))
    sq, dr, dist = decode_move(ref["policy_top"][0])
    print(f"  reference.json  value={ref['value']:+.3f}  top={sq} {dr} {dist}")


def write_wiring(net):
    C = net.stem[0].out_channels
    gaps = []
    for blk in net.trunk:
        M = kernel_norms(blk.conv2) @ kernel_norms(blk.conv1)
        M = M / (M.max() + 1e-9) + RESID * np.eye(C, dtype=M.dtype)
        o_idx, i_idx, w_val = [], [], []
        for o in range(C):
            for i in np.argsort(-M[o])[:TOPK]:
                if M[o, i] > 1e-6:
                    o_idx.append(int(o)); i_idx.append(int(i)); w_val.append(round(float(M[o, i]), 4))
        gaps.append({"o": o_idx, "i": i_idx, "w": w_val})
    (VIZ / "wiring.json").write_text(json.dumps(
        {"n_blocks": len(net.trunk), "topk": TOPK, "channels": C, "gaps": gaps},
        separators=(",", ":")))
    print(f"  wiring.json  ({sum(len(g['o']) for g in gaps)} edges)")


def bump_labels(gen):
    for name in ("index.html", "inspect.html"):
        p = VIZ / name
        if p.exists():
            p.write_text(re.sub(r"gen0\d{3,4}", f"gen{gen}", p.read_text(encoding="utf-8")),
                         encoding="utf-8")
    print(f"  html labels -> gen{gen}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen", nargs="?", default=None, help="generation, e.g. 195 / 0195 (default: latest)")
    gen = norm_gen(ap.parse_args().gen)
    ckpt = WEIGHTS / f"gen{gen}.onxx"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint: {ckpt}")
    print(f"refreshing viz -> gen{gen}")
    net = load_qnxx(ckpt).eval()
    export_model(net, gen)
    write_reference()
    write_wiring(net)
    bump_labels(gen)
    print("done. reload the viz page (ort-web will fetch the new net.onnx).")


if __name__ == "__main__":
    main()
