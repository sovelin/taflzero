"""Dataset health checks for TaflZero self-play data (format v2).

Diagnostics aimed at the "defense is weaker than attack" hypothesis:

  1. VALUE CALIBRATION by side. Bucket samples by predicted value (MCTS root_q)
     and compare to the actual game outcome z, split by side-to-move.
     If defender (stm=1) buckets show root_q systematically ABOVE mean z, the
     search over-estimates defensive positions ("thinks it holds, then loses").

  2. POLICY SHARPNESS by side x game phase (early/mid/late ply). Reveals whether
     defender targets are systematically flatter/noisier than attacker targets.

  3. ROOT_Q vs Z agreement per side + decisive/draw mix per side.

Reads the last --tail full-search samples (policy_valid) for policy stats; uses
all samples for value calibration.  Game phase (ply) is reconstructed from the
FLAG_LAST_OF_GAME bit.

Usage:
    python dataset_health.py --path selfplay-t2.bin --tail 300000
"""
from __future__ import annotations

import argparse
import math
import struct
from collections import deque

BIT_POSITION_BYTES = 50   # 48 planes + stm + rep
LEGAL_MASK_BYTES = 605

FLAG_POLICY_VALID = 1 << 0
FLAG_LAST_OF_GAME = 1 << 1

# ply buckets (avg game length ~150 in the current meta)
PHASES = [("early", 0, 40), ("mid", 40, 100), ("late", 100, 10_000)]


def iter_records(f):
    """Yield (stm, z, root_q, flags, corner, policy_bytes, policy_len) per sample."""
    while True:
        head = f.read(BIT_POSITION_BYTES)
        if len(head) < BIT_POSITION_BYTES:
            return
        stm = head[48]
        f.seek(LEGAL_MASK_BYTES, 1)  # skip legal mask
        raw = f.read(2)
        if len(raw) < 2:
            return
        policy_len = struct.unpack("<H", raw)[0]
        pol = f.read(policy_len * 4)
        if len(pol) < policy_len * 4:
            return
        tail = f.read(4)
        if len(tail) < 4:
            return
        z = struct.unpack("<b", tail[0:1])[0]
        root_q = struct.unpack("<b", tail[1:2])[0]
        flags = tail[2]
        corner = tail[3]
        yield stm, z, root_q, flags, corner, pol, policy_len


def policy_stats(pol: bytes, policy_len: int) -> tuple[float, float, float]:
    """Return (entropy_nats, n_eff, max_prob) of the visit distribution (temp 1.0)."""
    if policy_len == 0:
        return 0.0, 0.0, 0.0
    visits = [struct.unpack_from("<H", pol, i * 4 + 2)[0] for i in range(policy_len)]
    s = float(sum(visits))
    if s <= 0:
        return 0.0, 0.0, 0.0
    h = 0.0
    mx = 0.0
    for v in visits:
        if v > 0:
            p = v / s
            h -= p * math.log(p)
            if p > mx:
                mx = p
    return h, math.exp(h), mx


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset health checks (format v2)")
    ap.add_argument("--path", required=True)
    ap.add_argument("--tail", type=int, default=300000,
                    help="Analyze the last N samples (0 = all)")
    ap.add_argument("--bins", type=int, default=10, help="Value calibration bins")
    args = ap.parse_args()

    # Rolling window of light records; ply reconstructed on the fly.
    keep = deque(maxlen=args.tail if args.tail > 0 else None)
    ply = 0
    total = 0
    with open(args.path, "rb") as f:
        for stm, z, root_q, flags, corner, pol, plen in iter_records(f):
            total += 1
            rec = (stm, z, root_q, flags, corner, ply, pol, plen)
            keep.append(rec)
            if flags & FLAG_LAST_OF_GAME:
                ply = 0
            else:
                ply += 1

    print(f"File: {args.path}")
    print(f"Total samples: {total}   analyzed (tail): {len(keep)}")
    print()

    # ---- 1. Value calibration by side ----
    # bins over root_q in [-1, 1]
    nb = args.bins
    edges = [-1.0 + 2.0 * i / nb for i in range(nb + 1)]

    def bucket(q):
        for i in range(nb):
            if q < edges[i + 1] or i == nb - 1:
                return i
        return nb - 1

    calib = {0: [[0, 0.0, 0.0] for _ in range(nb)],   # [count, sum_q, sum_z]
             1: [[0, 0.0, 0.0] for _ in range(nb)]}
    side_z = {0: [0, 0.0], 1: [0, 0.0]}     # count, sum_z
    side_q = {0: [0, 0.0], 1: [0, 0.0]}     # count, sum_q
    side_outcome = {0: {"win": 0, "draw": 0, "loss": 0},
                    1: {"win": 0, "draw": 0, "loss": 0}}

    for stm, z, root_q, flags, corner, p, pol, plen in keep:
        if stm not in (0, 1):
            continue
        q = root_q / 127.0
        b = bucket(q)
        calib[stm][b][0] += 1
        calib[stm][b][1] += q
        calib[stm][b][2] += z
        side_z[stm][0] += 1
        side_z[stm][1] += z
        side_q[stm][0] += 1
        side_q[stm][1] += q
        side_outcome[stm]["win" if z > 0 else "loss" if z < 0 else "draw"] += 1

    for stm, name in [(0, "ATTACKERS (stm=0)"), (1, "DEFENDERS (stm=1)")]:
        print(f"=== VALUE CALIBRATION — {name} ===")
        print(f"{'root_q bin':>16} | {'n':>7} | {'pred q':>7} | {'actual z':>8} | {'gap(q-z)':>8}")
        for i in range(nb):
            c, sq, sz = calib[stm][i]
            if c == 0:
                continue
            pq, az = sq / c, sz / c
            print(f"{edges[i]:+.2f}..{edges[i+1]:+.2f} | {c:>7} | {pq:>+7.3f} | {az:>+8.3f} | {pq-az:>+8.3f}")
        cq = side_q[stm][1] / max(side_q[stm][0], 1)
        cz = side_z[stm][1] / max(side_z[stm][0], 1)
        o = side_outcome[stm]
        tot = sum(o.values()) or 1
        print(f"  overall: mean root_q={cq:+.3f}  mean z={cz:+.3f}  bias(q-z)={cq-cz:+.3f}")
        print(f"  outcomes: win {o['win']*100//tot}%  draw {o['draw']*100//tot}%  loss {o['loss']*100//tot}%")
        print()

    # ---- 2. Policy sharpness by side x phase (full-search only) ----
    print("=== POLICY SHARPNESS by side x phase (full-search, temp 1.0) ===")
    print(f"{'side':>10} {'phase':>6} | {'n':>7} | {'entropy':>7} | {'n_eff':>7} | {'max_prob':>8}")
    stats = {}  # (stm, phase) -> [count, sum_ent, sum_neff, sum_maxp]
    for stm, z, root_q, flags, corner, ply, pol, plen in keep:
        if not (flags & FLAG_POLICY_VALID) or stm not in (0, 1):
            continue
        ph = next((nm for nm, lo, hi in PHASES if lo <= ply < hi), "late")
        h, ne, mx = policy_stats(pol, plen)
        s = stats.setdefault((stm, ph), [0, 0.0, 0.0, 0.0])
        s[0] += 1; s[1] += h; s[2] += ne; s[3] += mx
    for stm, sname in [(0, "attacker"), (1, "defender")]:
        for ph, _, _ in PHASES:
            s = stats.get((stm, ph))
            if not s or s[0] == 0:
                continue
            print(f"{sname:>10} {ph:>6} | {s[0]:>7} | {s[1]/s[0]:>7.3f} | {s[2]/s[0]:>7.2f} | {s[3]/s[0]:>8.3f}")
        print()

    # ---- 3. Policy sharpness by decidedness (|root_q|) — is flat == decided? ----
    # If flat targets (low max_prob / high n_eff) cluster in decided positions
    # (|q|~1), they are "outcome already settled, moves indistinguishable" noise
    # → candidates for masking/down-weighting in the trainer.
    print("=== POLICY SHARPNESS by decidedness |root_q| (full-search, temp 1.0) ===")
    print(f"{'side':>10} {'|q| band':>10} | {'n':>7} | {'n_eff':>7} | {'max_prob':>8} | {'%late':>6}")
    dec_bands = [("0.0-0.5", 0.0, 0.5), ("0.5-0.8", 0.5, 0.8),
                 ("0.8-0.95", 0.8, 0.95), ("0.95-1.0", 0.95, 1.01)]
    dstats = {}  # (stm, band) -> [count, sum_neff, sum_maxp, late_count]
    for stm, z, root_q, flags, corner, ply, pol, plen in keep:
        if not (flags & FLAG_POLICY_VALID) or stm not in (0, 1):
            continue
        aq = abs(root_q / 127.0)
        band = next((nm for nm, lo, hi in dec_bands if lo <= aq < hi), "0.95-1.0")
        _, ne, mx = policy_stats(pol, plen)
        s = dstats.setdefault((stm, band), [0, 0.0, 0.0, 0])
        s[0] += 1; s[1] += ne; s[2] += mx
        if ply >= 100:
            s[3] += 1
    for stm, sname in [(0, "attacker"), (1, "defender")]:
        for band, _, _ in dec_bands:
            s = dstats.get((stm, band))
            if not s or s[0] == 0:
                continue
            print(f"{sname:>10} {band:>10} | {s[0]:>7} | {s[1]/s[0]:>7.2f} | {s[2]/s[0]:>8.3f} | {s[3]*100//s[0]:>5}%")
        print()


if __name__ == "__main__":
    main()
