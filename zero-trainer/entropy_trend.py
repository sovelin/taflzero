"""One-pass entropy trend: streaming, no RAM storage of samples."""
from __future__ import annotations
import math, struct

BIT_POSITION_BYTES = 50
LEGAL_MASK_BYTES   = 605
WINDOW             = 1_000_000


def popcount_bytes(b: bytes) -> int:
    return sum(int(x).bit_count() for x in b)


class WindowAccum:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum_enorm = 0.0
        self.sum_maxp  = 0.0
        self.sum_neff  = 0.0
        self.sum_vfrac = 0.0

    def add(self, legal: bytes, policy_raw: bytes, policy_len: int):
        legal_moves = popcount_bytes(legal)
        if policy_len == 0:
            return
        visits = []
        for i in range(policy_len):
            v = struct.unpack_from("<H", policy_raw, i * 4 + 2)[0]
            if v > 0:
                visits.append(float(v))
        if not visits:
            return
        s = sum(visits)
        probs = sorted([v / s for v in visits], reverse=True)
        h = sum(-p * math.log(p) for p in probs)
        self.sum_enorm += h / math.log(legal_moves) if legal_moves > 1 else 0.0
        self.sum_maxp  += probs[0]
        self.sum_neff  += math.exp(h)
        self.sum_vfrac += policy_len / legal_moves if legal_moves > 0 else 0.0
        self.count += 1

    def stats(self):
        if self.count == 0:
            return None
        n = self.count
        return {
            "entropy_norm": self.sum_enorm / n,
            "max_prob":     self.sum_maxp  / n,
            "n_eff":        self.sum_neff  / n,
            "visited_frac": self.sum_vfrac / n,
        }


def main():
    path = "selfplay.bin"

    hdr = f"{'Window':>14}  {'Samples':>9}  {'enorm':>7}  {'max_p':>7}  {'n_eff':>7}  {'vis_fr':>7}"
    sep = "-" * len(hdr)

    acc = WindowAccum()
    win_idx = 0
    total_idx = 0
    win_start = 0

    print(hdr)
    print(sep)

    with open(path, "rb") as f:
        while True:
            planes = f.read(48)
            if len(planes) < 48: break
            stm = f.read(1)
            if not stm: break
            f.read(1)  # rep
            legal = f.read(LEGAL_MASK_BYTES)
            if len(legal) < LEGAL_MASK_BYTES: break
            raw = f.read(2)
            if len(raw) < 2: break
            policy_len = struct.unpack("<H", raw)[0]
            policy_raw = f.read(policy_len * 4)
            if len(policy_raw) < policy_len * 4: break
            f.read(2)  # value + root_q

            acc.add(legal, policy_raw, policy_len)
            total_idx += 1

            if total_idx % 200_000 == 0:
                import sys
                pct = total_idx / 18_447_388 * 100
                print(f"  ... {total_idx//1_000_000}M samples ({pct:.0f}%)", file=sys.stderr, flush=True)

            if total_idx % WINDOW == 0:
                m = acc.stats()
                label = f"{win_start//1000}k-{total_idx//1000}k"
                if m:
                    print(f"{label:>14}  {acc.count:>9,}  "
                          f"{m['entropy_norm']:>7.4f}  "
                          f"{m['max_prob']:>7.4f}  "
                          f"{m['n_eff']:>7.2f}  "
                          f"{m['visited_frac']:>7.4f}")
                acc.reset()
                win_start = total_idx
                win_idx += 1

    # last partial window
    m = acc.stats()
    if m and acc.count > 0:
        label = f"{win_start//1000}k-{total_idx//1000}k"
        print(f"{label:>14}  {acc.count:>9,}  "
              f"{m['entropy_norm']:>7.4f}  "
              f"{m['max_prob']:>7.4f}  "
              f"{m['n_eff']:>7.2f}  "
              f"{m['visited_frac']:>7.4f}")

    print(sep)
    print(f"Total samples: {total_idx:,}")
    print()
    print("enorm:  0.15-0.20 = mature | >0.40 = lots to learn")
    print("n_eff:  2-5 = mature       | >10   = active exploration")
    print("vis_fr: 0.3-0.4 = mature   | >0.6  = MCTS visits most moves")


if __name__ == "__main__":
    main()
