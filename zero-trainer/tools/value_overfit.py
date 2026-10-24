"""Value overfit check for a continuous run (docs/plan-T9.md §8).

Loads a checkpoint trained up to `generated` samples and measures its value loss
on sample ranges relative to that point: positions it has not seen yet (fresh),
positions it trained on recently, earlier in the window, and ones already out of
the window. Memorisation shows as seen-loss well below fresh-loss; plain
distribution drift makes older ranges worse, not better.

    python tools/value_overfit.py runs/T8-10x128 step044000 [--n 30000]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import geometry, open_features, parse_samples  # noqa: E402
from qnxx_io import load_qnxx  # noqa: E402
from replay import Chunk, ReplayBuffer  # noqa: E402


def load_range(data: Path, games: list, lo: int, hi: int, feat, geom):
    chunks = []
    with open(data / "worker0.bin", "rb") as f:
        for off, n, first, nbytes, _res in games:
            if first < lo or first + n > hi:
                continue
            f.seek(off)
            samples, _ = parse_samples(f.read(nbytes), geom)
            c = Chunk.from_samples(samples, source="worker0.bin", offset=off)
            c.attach_feat(np.asarray(feat[first:first + n]))
            chunks.append(c)
    return chunks


def evaluate(model, buf, chunks, n, rng, device):
    located = [(c, i) for c in chunks for i in range(len(c))]
    pick = rng.choice(len(located), size=min(n, len(located)), replace=False)
    out = {"mix_full": [], "z_full": [], "z_cheap": [], "pol": []}
    for s in range(0, len(pick), 1024):
        part = [located[k] for k in pick[s:s + 1024]]
        b = buf.collate(part, np.zeros(len(part), dtype=np.int64))
        planes, legal, pi, vt, pvalid = b[0], b[1], b[2], b[3], b[4]
        known = b[11].numpy() > 0
        z = np.array([c.value[i] for c, i in part], dtype=np.float32)
        with torch.no_grad():
            o = model(planes.to(device))
            v = o[1].float().squeeze(1).cpu().numpy()
            logits = o[0].float().cpu()
        logits = logits.masked_fill(legal <= 0, -1e9)
        ce = -(pi * torch.log_softmax(logits, dim=1)).sum(1).numpy()
        full = pvalid.numpy() > 0
        vt = vt.numpy()
        out["mix_full"] += list(((v - vt) ** 2)[known & full])
        out["z_full"] += list(((v - z) ** 2)[known & full])
        out["z_cheap"] += list(((v - z) ** 2)[known & ~full])
        out["pol"] += list(ce[full])
    return {k: (float(np.mean(x)), len(x)) for k, x in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=30000)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8-sig"))
    G = json.loads((run / "ckpt" / f"{a.ckpt}.json").read_text())["generated"]
    geom = geometry(11)
    idx = json.loads((run / "data" / "games0.idx.json").read_text())
    feat = open_features(run / "data" / "worker0.feat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_qnxx(run / "ckpt" / f"{a.ckpt}.qnxx").to(device).eval()
    buf = ReplayBuffer(geom, z_lambda=cfg["value_lambda"], augment=False, planes=cfg["planes"] or "base")
    rng = np.random.default_rng(1)
    K = 1000
    ranges = [("fresh (not trained yet)", G, G + 200 * K),
              ("seen, last 200k", G - 200 * K, G),
              ("seen, 1.0-1.2M back", G - 1200 * K, G - 1000 * K),
              ("seen, 2.2-2.4M back", G - 2400 * K, G - 2200 * K),
              ("out of window, 4.0-4.2M back", G - 4200 * K, G - 4000 * K)]
    print(f"{a.ckpt}: trained up to {G/1e6:.2f}M samples, value_lambda {cfg['value_lambda']}")
    print(f"{'range':32s} {'value mix (full)':>17s} {'value z (full)':>15s} {'value z (cheap)':>16s} {'policy CE':>10s}")
    for name, lo, hi in ranges:
        chunks = load_range(run / "data", idx["games"], lo, hi, feat, geom)
        r = evaluate(model, buf, chunks, a.n, rng, device)
        print(f"{name:32s} {r['mix_full'][0]:17.4f} {r['z_full'][0]:15.4f} {r['z_cheap'][0]:16.4f} {r['pol'][0]:10.4f}"
              f"   (n full {r['z_full'][1]}, cheap {r['z_cheap'][1]})")


if __name__ == "__main__":
    main()
