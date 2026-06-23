"""Build FEN->planes (the spec the JS side ports), validate it byte-for-byte
against the engine's --nn-eval dump, run the instrumented ONNX via
onnxruntime, and emit viz/reference.json (golden data for the JS visualizer).
"""
from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

B = 11
SQS = 121
THRONE = 60
CORNERS = [0, 10, 110, 120]


def neighbors(sq):
    r, c = divmod(sq, B)
    out = []
    if r > 0: out.append(sq - B)
    if r < B - 1: out.append(sq + B)
    if c > 0: out.append(sq - 1)
    if c < B - 1: out.append(sq + 1)
    return out


def bfs(seeds, passable):
    vis = np.zeros(SQS, np.float32)
    q = deque()
    for s in seeds:
        if passable[s] and not vis[s]:
            vis[s] = 1.0; q.append(s)
    while q:
        sq = q.popleft()
        for nb in neighbors(sq):
            if passable[nb] and not vis[nb]:
                vis[nb] = 1.0; q.append(nb)
    return vis


def fen_to_masks(fen, top_is_row0=True):
    """Parse the board field of a FEN into attacker/defender/king masks.
    top_is_row0: whether the first FEN rank maps to sq row 0."""
    board = fen.split()[0]
    ranks = board.split("/")
    assert len(ranks) == B, f"{len(ranks)} ranks"
    atk = np.zeros(SQS, np.float32); dfn = np.zeros(SQS, np.float32); king = np.zeros(SQS, np.float32)
    for i, rank in enumerate(ranks):
        r = i if top_is_row0 else (B - 1 - i)
        c = 0
        for ch in rank:
            if ch.isdigit():
                # multi-digit runs: FEN uses single digits here (max 11 -> but 11 is "b"? no)
                c += int(ch)
            else:
                sq = r * B + c
                if ch == "a": atk[sq] = 1.0
                elif ch == "d": dfn[sq] = 1.0
                elif ch in "Kk": king[sq] = 1.0
                else: raise ValueError(ch)
                c += 1
    return atk, dfn, king


def build_planes(fen):
    # NOTE: FEN digits only go 1-9; a run of 10/11 empties is written as e.g. "9" + ... ?
    # The engine emits multi-char; handle >9 by treating consecutive digits as one number.
    board, side = fen.split()[0], fen.split()[1]
    # re-tokenize digits as full numbers
    ranks = board.split("/")
    atk = np.zeros(SQS, np.float32); dfn = np.zeros(SQS, np.float32); king = np.zeros(SQS, np.float32)
    for i, rank in enumerate(ranks):
        r = B - 1 - i  # FEN first rank is the TOP (rank 11) -> engine sq row 10
        c = 0
        for tok in re.findall(r"\d+|[adKk]", rank):
            if tok.isdigit():
                c += int(tok)
            else:
                sq = r * B + c
                if tok == "a": atk[sq] = 1.0
                elif tok == "d": dfn[sq] = 1.0
                else: king[sq] = 1.0
                c += 1
    stm = 1.0 if side == "d" else 0.0

    throne = np.zeros(SQS, np.float32); throne[THRONE] = 1.0
    corners = np.zeros(SQS, np.float32)
    for s in CORNERS: corners[s] = 1.0
    edges = np.zeros(SQS, np.float32)
    for sq in range(SQS):
        r, c = divmod(sq, B)
        if r in (0, B - 1) or c in (0, B - 1): edges[sq] = 1.0

    group_bfs = bfs(list(np.where((king > 0) | (dfn > 0))[0]), ~(atk > 0))
    king_bfs = bfs(list(np.where(king > 0)[0]), ~((atk > 0) | (dfn > 0)))
    stm_plane = np.full(SQS, stm, np.float32)
    rep1 = np.zeros(SQS, np.float32); rep2 = np.zeros(SQS, np.float32)

    planes = np.stack([atk, dfn, king, stm_plane, throne, corners, edges,
                       group_bfs, king_bfs, rep1, rep2])
    return planes.reshape(1, B, B, B).astype(np.float32)  # (1,11,11,11) as (1,plane,r,c)


DIRS = ["Up", "Down", "Left", "Right"]  # index 0..3


def decode_move(idx):
    sq = idx // 40
    rem = idx % 40
    d = rem // 10
    dist = rem % 10 + 1
    r, c = divmod(sq, B)
    return f"{'abcdefghijk'[c]}{r + 1}", DIRS[d], dist


def main():
    fen = "3aaaa4/8a2/7a3/a4d4a/a3ddd1k1a/aa1dd1dd1aa/a3d4da/a3ad1a3/3a1d5/5a5/5aa1a2 d"
    planes = build_planes(fen)  # (1,11,121)->(1,11,11,11)

    # validate against engine dump if available
    dump = Path("../viz/nneval_mate.json")
    if dump.exists():
        eng = np.array(json.loads(dump.read_text())["input"], np.float32).reshape(1, 11, 11, 11)
        diff = float(np.abs(eng - planes).max())
        print(f"engine-vs-python plane max|delta| = {diff}")
    else:
        print("(no engine dump to validate against; skipping)")

    sess = ort.InferenceSession("../viz/net.onnx", providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"input": planes})
    names = [o.name for o in sess.get_outputs()]
    res = dict(zip(names, outs))
    policy = res["policy"][0]
    top = np.argsort(-policy)[:5]
    print("value =", float(res["value"][0][0]))
    print("top policy moves:")
    for idx in top:
        sq, d, dist = decode_move(int(idx))
        print(f"  {sq} {d} {dist}   logit={policy[idx]:.2f}")

    # per-block channel activation magnitude (L2 over 11x11) for the JS golden ref
    act_names = ["stem"] + [f"block{i}" for i in range(10)]
    chan_mag = {n: np.linalg.norm(res[n][0].reshape(res[n][0].shape[0], -1), axis=1).tolist()
                for n in act_names}
    ref = {
        "fen": fen,
        "input": planes.reshape(-1).tolist(),
        "value": float(res["value"][0][0]),
        "policy_top": [int(i) for i in top],
        "chan_mag": chan_mag,
    }
    Path("../viz/reference.json").write_text(json.dumps(ref))
    print("wrote ../viz/reference.json")


if __name__ == "__main__":
    main()
