# Analyze policy target sharpness/entropy in selfplay.bin
# Usage: python zero-trainer\check_policy_targets.py --path zero-trainer\selfplay.bin
from __future__ import annotations

import argparse
import math
import os
import struct
from statistics import mean, median

BIT_POSITION_BYTES = 49
LEGAL_MASK_BYTES = 605
FIXED_HEADER = BIT_POSITION_BYTES + LEGAL_MASK_BYTES + 2


def percentile(xs, p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def popcount_bytes(b: bytes) -> int:
    return sum(int(x).bit_count() for x in b)


def iter_samples(fh):
    while True:
        planes = fh.read(48)
        if len(planes) < 48:
            return
        stm = fh.read(1)
        if len(stm) < 1:
            return
        legal = fh.read(LEGAL_MASK_BYTES)
        if len(legal) < LEGAL_MASK_BYTES:
            return
        raw = fh.read(2)
        if len(raw) < 2:
            return
        policy_len = struct.unpack("<H", raw)[0]
        policy_raw = fh.read(policy_len * 4)
        if len(policy_raw) < policy_len * 4:
            return
        val = fh.read(1)
        if len(val) < 1:
            return
        yield legal, policy_raw, policy_len


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze policy target sharpness / entropy from selfplay.bin")
    ap.add_argument("--path", default=os.path.join("zero-trainer", "selfplay.bin"))
    ap.add_argument("--policy-temp", type=float, default=0.5, help="Same temp as dataset.py (default 0.5)")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = no limit")
    ap.add_argument("--every", type=int, default=1, help="Only analyze every Nth sample")
    args = ap.parse_args()

    if args.policy_temp <= 0:
        raise SystemExit("policy-temp must be > 0")
    if args.every < 1:
        raise SystemExit("every must be >= 1")

    entropies = []
    entropies_norm = []
    max_probs = []
    top5_probs = []
    n_eff = []
    visited_frac = []
    legal_moves_list = []
    policy_len_list = []

    samples = 0
    used = 0

    with open(args.path, "rb") as f:
        for legal, policy_raw, policy_len in iter_samples(f):
            samples += 1
            if samples % args.every != 0:
                continue
            if args.max_samples and used >= args.max_samples:
                break

            legal_moves = popcount_bytes(legal)
            legal_moves_list.append(legal_moves)
            policy_len_list.append(policy_len)

            if policy_len == 0:
                continue

            # Extract visit counts
            visits = []
            for i in range(policy_len):
                off = i * 4
                v = struct.unpack_from("<H", policy_raw, off + 2)[0]
                if v > 0:
                    visits.append(float(v))

            if not visits:
                continue

            # Apply temperature (same as dataset.py)
            if args.policy_temp != 1.0:
                power = 1.0 / args.policy_temp
                visits = [v ** power for v in visits]

            s = sum(visits)
            if s <= 0:
                continue

            probs = [v / s for v in visits]
            probs.sort(reverse=True)

            p1 = probs[0]
            top5 = sum(probs[:5]) if len(probs) >= 5 else sum(probs)

            h = 0.0
            for p in probs:
                h -= p * math.log(p)

            entropies.append(h)
            if legal_moves > 1:
                entropies_norm.append(h / math.log(legal_moves))
            else:
                entropies_norm.append(0.0)

            max_probs.append(p1)
            top5_probs.append(top5)
            n_eff.append(math.exp(h))
            visited_frac.append(policy_len / legal_moves if legal_moves > 0 else 0.0)

            used += 1

    def fmt_stats(name, xs):
        if not xs:
            print(f"{name}: no data")
            return
        print(
            f"{name}: mean={mean(xs):.4f} median={median(xs):.4f} "
            f"p10={percentile(xs, 0.10):.4f} p50={percentile(xs, 0.50):.4f} p90={percentile(xs, 0.90):.4f}"
        )

    print(f"File: {args.path}")
    print(f"Analyzed samples: {used} (every={args.every}, total seen={samples})")
    print(f"policy_temp={args.policy_temp}")
    print()
    fmt_stats("legal_moves", legal_moves_list)
    fmt_stats("policy_len", policy_len_list)
    fmt_stats("visited_frac", visited_frac)
    fmt_stats("entropy", entropies)
    fmt_stats("entropy_norm", entropies_norm)
    fmt_stats("max_prob", max_probs)
    fmt_stats("top5_prob", top5_probs)
    fmt_stats("n_eff", n_eff)

    # Quick flags
    if max_probs:
        low_max = sum(1 for x in max_probs if x < 0.1) / len(max_probs)
        print(f"max_prob < 0.10: {low_max * 100:.1f}%")
        low_max2 = sum(1 for x in max_probs if x < 0.05) / len(max_probs)
        print(f"max_prob < 0.05: {low_max2 * 100:.1f}%")


if __name__ == "__main__":
    main()
