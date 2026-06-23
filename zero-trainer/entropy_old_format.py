"""Entropy stats for old-format selfplay.bin (no rep byte). Last 1M samples only."""
from __future__ import annotations
import math, struct, sys

# Old format: 48 planes + 1 stm (NO rep byte)
SKIP_FIXED  = 48 + 1 + 605  # planes + stm + legal mask
POLICY_LEN_OFF = SKIP_FIXED   # offset of policy_len u16 within sample

WINDOW = 1_000_000

def count_and_find_start(path: str, target_tail: int) -> tuple[int, int]:
    """Return (total_samples, byte_offset_of_first_sample_in_tail)."""
    total = 0
    offsets_ring = []  # ring buffer of byte offsets, size = target_tail
    ring_size = target_tail

    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            header = f.read(SKIP_FIXED)
            if len(header) < SKIP_FIXED:
                break
            raw = f.read(2)
            if len(raw) < 2:
                break
            policy_len = struct.unpack("<H", raw)[0]
            f.seek(policy_len * 4 + 1, 1)  # skip policy + value

            if total < ring_size:
                offsets_ring.append(pos)
            else:
                offsets_ring[total % ring_size] = pos
            total += 1

            if total % 500_000 == 0:
                pct = total / 200_000_000 * 100  # rough estimate
                print(f"  quick-scan: {total//1_000_000}M samples... (~{pct:.0f}%)", file=sys.stderr, flush=True)

    if total <= ring_size:
        start_offset = offsets_ring[0] if offsets_ring else 0
    else:
        start_offset = offsets_ring[total % ring_size]

    return total, start_offset


def popcount_bytes(b: bytes) -> int:
    return sum(int(x).bit_count() for x in b)


def analyze_from_offset(path: str, start_offset: int, max_samples: int):
    enorm_sum = maxp_sum = neff_sum = vfrac_sum = 0.0
    count = 0

    with open(path, "rb") as f:
        f.seek(start_offset)
        done = 0
        while done < max_samples:
            planes = f.read(48)
            if len(planes) < 48: break
            stm = f.read(1)
            if not stm: break
            legal = f.read(605)
            if len(legal) < 605: break
            raw = f.read(2)
            if len(raw) < 2: break
            policy_len = struct.unpack("<H", raw)[0]
            policy_raw = f.read(policy_len * 4)
            if len(policy_raw) < policy_len * 4: break
            f.read(1)  # value

            done += 1
            if done % 200_000 == 0:
                print(f"  parsing: {done//1000}k / {max_samples//1000}k", file=sys.stderr, flush=True)

            if policy_len == 0:
                continue
            legal_moves = popcount_bytes(legal)

            visits = []
            for i in range(policy_len):
                v = struct.unpack_from("<H", policy_raw, i * 4 + 2)[0]
                if v > 0:
                    visits.append(float(v))
            if not visits:
                continue

            s = sum(visits)
            probs = sorted([v / s for v in visits], reverse=True)
            h = sum(-p * math.log(p) for p in probs)

            enorm_sum  += h / math.log(legal_moves) if legal_moves > 1 else 0.0
            maxp_sum   += probs[0]
            neff_sum   += math.exp(h)
            vfrac_sum  += policy_len / legal_moves if legal_moves > 0 else 0.0
            count += 1

    if count == 0:
        return None
    return {
        "entropy_norm": enorm_sum / count,
        "max_prob":     maxp_sum  / count,
        "n_eff":        neff_sum  / count,
        "visited_frac": vfrac_sum / count,
        "count":        count,
    }


def main():
    path = r"weights\modern-6x64\selfplay.bin"

    print(f"Pass 1: quick-scan {path} ...", flush=True)
    total, start_offset = count_and_find_start(path, WINDOW)
    tail = min(WINDOW, total)
    print(f"Total samples: {total:,}  |  analyzing last {tail:,}  (offset {start_offset:,})")

    print(f"\nPass 2: computing entropy for last {tail:,} samples ...")
    m = analyze_from_offset(path, start_offset, tail)

    if m is None:
        print("No data.")
        return

    print()
    print(f"  entropy_norm : {m['entropy_norm']:.4f}")
    print(f"  max_prob     : {m['max_prob']:.4f}")
    print(f"  n_eff        : {m['n_eff']:.2f}")
    print(f"  visited_frac : {m['visited_frac']:.4f}")
    print(f"  samples used : {m['count']:,}")
    print()
    print("Ref (current run gen36):  enorm=0.646  max_p=0.329  n_eff=29.4  vis=0.720")
    print("Ref (mature target):      enorm~0.20   max_p~0.60   n_eff~3-5   vis~0.35")


if __name__ == "__main__":
    main()
