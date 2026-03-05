"""Compare FP32 vs FP16 ONNX model outputs on realistic binary game positions."""

import numpy as np
import onnxruntime as ort
from pathlib import Path
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def spearmanr(a, b):
    n = len(a)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    d = ra - rb
    rho = 1 - 6 * (d**2).sum() / (n * (n**2 - 1))
    return rho, None

# Board constants
SQS = 121  # 11x11
NUM_PLANES = 6
SAMPLE_SIZE = NUM_PLANES * SQS

FP32_PATH = Path("weights/gen0361.onnx")
FP16_PATH = Path("weights/gen0361_fp16.onnx")


def make_starting_position() -> np.ndarray:
    """Create starting position for Hnefatafl 11x11 (Copenhagen rules)."""
    inp = np.zeros((NUM_PLANES, 11, 11), dtype=np.float32)

    # Attackers (plane 0) - standard 11x11 Copenhagen starting positions
    attacker_coords = [
        (0,3),(0,4),(0,5),(0,6),(0,7),
        (1,5),
        (3,0),(4,0),(5,0),(6,0),(7,0),
        (3,10),(4,10),(5,10),(6,10),(7,10),
        (10,3),(10,4),(10,5),(10,6),(10,7),
        (9,5),
        (5,1),(5,9),
    ]
    for r, c in attacker_coords:
        inp[0, r, c] = 1.0

    # Defenders (plane 1)
    defender_coords = [
        (3,5),(4,4),(4,5),(4,6),
        (5,3),(5,4),(5,6),(5,7),
        (6,4),(6,5),(6,6),(7,5),
    ]
    for r, c in defender_coords:
        inp[1, r, c] = 1.0

    # King (plane 2)
    inp[2, 5, 5] = 1.0

    # Side to move: attackers first (plane 3 = all 1s)
    inp[3, :, :] = 1.0

    # Throne (plane 4)
    inp[4, 5, 5] = 1.0

    # Corners (plane 5)
    for r, c in [(0,0),(0,10),(10,0),(10,10)]:
        inp[5, r, c] = 1.0

    return inp.reshape(1, NUM_PLANES, 11, 11)


def make_random_binary_position(rng: np.random.Generator) -> np.ndarray:
    """Create a random but structurally valid binary position."""
    inp = np.zeros((1, NUM_PLANES, 11, 11), dtype=np.float32)

    all_squares = [(r, c) for r in range(11) for c in range(11)]
    rng.shuffle(all_squares)

    # Place pieces randomly
    n_attackers = rng.integers(8, 25)
    n_defenders = rng.integers(5, 15)
    king_sq = all_squares[0]

    used = {king_sq}
    attackers = []
    defenders = []

    for sq in all_squares[1:]:
        if len(attackers) < n_attackers:
            attackers.append(sq)
            used.add(sq)
        elif len(defenders) < n_defenders:
            if sq not in used:
                defenders.append(sq)
                used.add(sq)
        else:
            break

    for r, c in attackers:
        inp[0, 0, r, c] = 1.0
    for r, c in defenders:
        inp[0, 1, r, c] = 1.0
    inp[0, 2, king_sq[0], king_sq[1]] = 1.0

    # Side to move
    stm = rng.integers(0, 2)
    inp[0, 3, :, :] = float(stm)

    # Throne and corners (fixed)
    inp[0, 4, 5, 5] = 1.0
    for r, c in [(0,0),(0,10),(10,0),(10,10)]:
        inp[0, 5, r, c] = 1.0

    return inp


def policy_stats(p32: np.ndarray, p16: np.ndarray) -> dict:
    """Compute various comparison stats between two raw policy logit arrays."""
    p32 = p32.flatten()
    p16 = p16.flatten()

    sm32 = softmax(p32)
    sm16 = softmax(p16)

    # KL divergence (FP32 || FP16)
    kl = np.sum(sm32 * np.log((sm32 + 1e-12) / (sm16 + 1e-12)))

    # Top-1 agreement
    top1_match = int(np.argmax(p32) == np.argmax(p16))

    # Top-5 overlap
    top5_32 = set(np.argsort(p32)[-5:])
    top5_16 = set(np.argsort(p16)[-5:])
    top5_overlap = len(top5_32 & top5_16)

    # L1 diff on softmax
    l1 = np.abs(sm32 - sm16).sum()

    # Max logit diff
    max_diff = np.abs(p32 - p16).max()

    # Spearman rank correlation
    rho, _ = spearmanr(p32, p16)

    return {
        "kl": kl,
        "top1_match": top1_match,
        "top5_overlap": top5_overlap,
        "l1_softmax": l1,
        "max_logit_diff": max_diff,
        "spearman_rho": rho,
    }


def main():
    print(f"Loading FP32: {FP32_PATH}")
    s32 = ort.InferenceSession(str(FP32_PATH), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"Loading FP16: {FP16_PATH}")
    s16 = ort.InferenceSession(str(FP16_PATH), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    print(f"\nFP32 provider: {s32.get_providers()[0]}")
    print(f"FP16 provider: {s16.get_providers()[0]}")

    # --- Starting position test ---
    print("\n=== Starting position ===")
    x_start = make_starting_position()
    p32_s, v32_s = s32.run(None, {"input": x_start})
    p16_s, v16_s = s16.run(None, {"input": x_start})
    stats = policy_stats(p32_s, p16_s)
    print(f"  value  FP32={v32_s[0,0]:.6f}  FP16={v16_s[0,0]:.6f}  diff={abs(v32_s[0,0]-v16_s[0,0]):.6f}")
    print(f"  policy max_logit_diff={stats['max_logit_diff']:.6f}")
    print(f"  policy KL(fp32||fp16)={stats['kl']:.6f}")
    print(f"  policy L1 softmax diff={stats['l1_softmax']:.6f}")
    print(f"  top-1 match: {stats['top1_match']}  top-5 overlap: {stats['top5_overlap']}/5")
    print(f"  Spearman rho={stats['spearman_rho']:.6f}")

    # --- Random binary positions ---
    print("\n=== 200 random binary positions ===")
    rng = np.random.default_rng(123)

    kls, l1s, top1s, top5s, rhos = [], [], [], [], []
    val_diffs = []
    max_logit_diffs = []

    for i in range(200):
        x = make_random_binary_position(rng)
        p32, v32 = s32.run(None, {"input": x})
        p16, v16 = s16.run(None, {"input": x})
        s = policy_stats(p32, p16)
        kls.append(s["kl"])
        l1s.append(s["l1_softmax"])
        top1s.append(s["top1_match"])
        top5s.append(s["top5_overlap"])
        rhos.append(s["spearman_rho"])
        val_diffs.append(abs(v32[0,0] - v16[0,0]))
        max_logit_diffs.append(s["max_logit_diff"])

    print(f"  value diff      mean={np.mean(val_diffs):.6f}  max={max(val_diffs):.6f}")
    print(f"  max_logit_diff  mean={np.mean(max_logit_diffs):.6f}  max={max(max_logit_diffs):.6f}")
    print(f"  KL(fp32||fp16)  mean={np.mean(kls):.6f}  max={max(kls):.6f}")
    print(f"  L1 softmax      mean={np.mean(l1s):.6f}  max={max(l1s):.6f}")
    print(f"  top-1 match     {sum(top1s)/len(top1s)*100:.1f}%  ({sum(top1s)}/{len(top1s)})")
    print(f"  top-5 overlap   mean={np.mean(top5s):.2f}/5")
    print(f"  Spearman rho    mean={np.mean(rhos):.6f}  min={min(rhos):.6f}")

    # --- Worst cases ---
    print("\n=== Top 5 worst KL positions ===")
    sorted_idx = np.argsort(kls)[::-1]
    for rank, idx in enumerate(sorted_idx[:5]):
        print(f"  #{rank+1}: KL={kls[idx]:.6f}  L1={l1s[idx]:.6f}  top1={top1s[idx]}  rho={rhos[idx]:.4f}  val_diff={val_diffs[idx]:.6f}")

    # --- Random Gaussian (original validation) ---
    print("\n=== 200 Gaussian random inputs (original validate method) ===")
    gkls, gl1s, gtop1s = [], [], []
    for _ in range(200):
        x = rng.standard_normal((1, 6, 11, 11)).astype(np.float32)
        p32, v32 = s32.run(None, {"input": x})
        p16, v16 = s16.run(None, {"input": x})
        s = policy_stats(p32, p16)
        gkls.append(s["kl"])
        gl1s.append(s["l1_softmax"])
        gtop1s.append(s["top1_match"])

    print(f"  KL(fp32||fp16)  mean={np.mean(gkls):.6f}  max={max(gkls):.6f}")
    print(f"  L1 softmax      mean={np.mean(gl1s):.6f}  max={max(gl1s):.6f}")
    print(f"  top-1 match     {sum(gtop1s)/len(gtop1s)*100:.1f}%")

    print("\n=== Summary ===")
    print(f"Binary positions top-1 match: {sum(top1s)/len(top1s)*100:.1f}%")
    print(f"Gaussian inputs  top-1 match: {sum(gtop1s)/len(gtop1s)*100:.1f}%")
    print(f"Binary KL mean={np.mean(kls):.4f}  Gaussian KL mean={np.mean(gkls):.4f}")
    print()
    if sum(top1s)/len(top1s) < 0.95:
        print("CONCLUSION: FP16 disagrees on top-1 move in >5% of real positions — this explains the Elo drop.")
    else:
        print("CONCLUSION: top-1 move mostly matches. Elo drop may be from accumulated small policy errors or provider mismatch.")


if __name__ == "__main__":
    main()
