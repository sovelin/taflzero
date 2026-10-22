"""Supervised probe: which input planes earn their place (docs/probe-input-planes.md).

    python zero-trainer/tools/probe_planes.py --data <worker0.bin> --features <worker0.feat> \
        --out zero-trainer/probe_planes/ [--variants A,I,G] [--resume]
    python zero-trainer/tools/probe_planes.py --data ... --features ... --out ... --stage 2
    python zero-trainer/tools/probe_planes.py --out zero-trainer/probe_planes/ --report

The same 6x64 net is trained on the same positions with different input stacks (and,
for J, without the value loss of cheap-search samples) and scored on a holdout that is
cut by game and by time: the last `--samples` samples of the file, cut at a game
boundary, of which the last 5% of the *games* are held out. Positions of one game never
land on both sides, so a plane that only helps a net remember a game cannot win.

The side file comes from `taflzero --features <data> --out <feat>`; if the data file
has grown since (a live worker file), only its first `count(feat)` samples are used.

Each run writes `<out>/<id>-s<seed>.json` (metrics per holdout subset at the end, the
curve every --eval-every steps); `--report` reads them all and writes `<out>/report.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from az_micro_net import TaflAlphaZeroNet  # noqa: E402
from dataset import (  # noqa: E402
    BIT_POSITION_BYTES,
    FEAT_FLAG_KING_MOVE,
    FEAT_FLAG_THREEFOLD,
    FEAT_FLAGS,
    FEAT_PLIES_TO_END,
    FEAT_PLY,
    FLAG_POLICY_VALID,
    FLAG_RESULT_UNKNOWN,
    SelfPlayDataset,
    feature_count,
    open_features,
    parse_planes,
)
from train import compute_batch_loss  # noqa: E402

# id -> (plane list, value-loss weight of cheap-search samples)
VARIANTS: dict[str, tuple[str, float]] = {
    "A": ("base", 1.0),
    "B": ("base,-throne,-corners,-edges", 1.0),
    "C": ("base,last2", 1.0),
    "D": ("base,rep_dst", 1.0),
    "E": ("base,king12", 1.0),
    "F": ("base,last2,rep_dst,king12,ply", 1.0),
    "G": ("base,-throne,-corners,-edges,last2,rep_dst,king12,ply", 1.0),
    "H": ("base,cap,mob", 1.0),
    "I": ("base,last2,rep_dst,king12,ply,cap,mob", 1.0),
    "J": ("base", 0.0),
    "K": ("base,-throne,-corners,-edges,last2,rep_dst,king12,ply,cap,mob", 1.0),
}

SUBSETS = ("all", "tactical", "king_move", "repeat_game", "cheap")
METRICS = ("policy_ce", "top1", "value_mse", "value_mse_z")
# A stage-1 variant gets a second seed when its `all` policy loss is this far from A's.
STAGE2_POLICY_GAP = 0.01


# ─── data ────────────────────────────────────────────────────────────────────


class ProbeData:
    """The window, the split and the holdout's per-sample facts, shared by every run."""

    def __init__(self, data: Path, features: Path, samples: int, holdout_frac: float,
                 z_lambda: float, curve_samples: int) -> None:
        count = feature_count(features)
        feat = open_features(features)
        ply = feat[:, FEAT_PLY].astype(np.int64) | (feat[:, FEAT_PLY + 1].astype(np.int64) << 8)
        starts = np.flatnonzero(ply == 0)
        if len(starts) == 0 or starts[0] != 0:
            raise SystemExit(f"{features}: the first record does not start a game")
        # The last `samples` samples, cut at a game boundary.
        cut = int(np.searchsorted(starts, max(0, count - samples)))
        if cut == len(starts):
            raise SystemExit(f"--samples {samples} ends inside the last game; raise it")
        first = int(starts[cut])
        window = count - first
        game_starts = starts[starts >= first] - first
        games = len(game_starts)
        if games < 2:
            raise SystemExit(f"only {games} game(s) in the window; raise --samples")
        hold_game = min(games - 1, max(1, int(round(games * (1.0 - holdout_frac)))))
        self.split = int(game_starts[hold_game])
        self.window = window
        self.count = count
        self.games = games
        self.holdout_games = games - hold_game

        t0 = time.time()
        print(f"[data] {count} records in {features}; window = last {window} samples "
              f"({games} games), holdout = last {self.holdout_games} games "
              f"({window - self.split} samples)")
        # need_moves_left: the value-loss mask is then exactly "result known".
        self.base = SelfPlayDataset(data, window_size=window, z_lambda=z_lambda, augment=True,
                                    need_moves_left=True, planes="base", features=features,
                                    limit=count)
        print(f"[data] index built in {time.time() - t0:.0f} s")

        # Holdout facts: from the samples (policy_valid, result known, z) and the side
        # file (plies to end, king move, threefold game).
        hold = range(self.split, window)
        self.hold_indices = np.arange(self.split, window)
        pv = np.zeros(len(hold), dtype=bool)
        known = np.zeros(len(hold), dtype=bool)
        z = np.zeros(len(hold), dtype=np.float32)
        fh = self.base._get_fh()
        mask_bytes = self.base._geom.legal_mask_bytes
        for j, i in enumerate(hold):
            off = int(self.base._offsets[i])
            fh.seek(off + BIT_POSITION_BYTES + mask_bytes)
            plen = int.from_bytes(fh.read(2), "little")
            fh.seek(off + self.base._geom.fixed_header + plen * 4)
            value, _root_q, flags = np.frombuffer(fh.read(3), dtype=np.int8).tolist()
            flags &= 0xFF
            pv[j] = bool(flags & FLAG_POLICY_VALID)
            known[j] = not (flags & FLAG_RESULT_UNKNOWN)
            z[j] = float(value)
        rec = feat[first + self.split:first + window]
        to_end = rec[:, FEAT_PLIES_TO_END].astype(np.int64) | (rec[:, FEAT_PLIES_TO_END + 1].astype(np.int64) << 8)
        fflags = rec[:, FEAT_FLAGS]
        self.hold_policy_valid = pv
        self.hold_known = known
        self.hold_z = z
        self.subsets = {
            "all": np.ones(len(hold), dtype=bool),
            "tactical": (to_end <= 6) & known,
            "king_move": (fflags & FEAT_FLAG_KING_MOVE) != 0,
            "repeat_game": (fflags & FEAT_FLAG_THREEFOLD) != 0,
            "cheap": ~pv,
        }
        rng = np.random.default_rng(12345)
        n_curve = min(curve_samples, len(hold)) if curve_samples > 0 else len(hold)
        self.curve_pick = np.sort(rng.choice(len(hold), size=n_curve, replace=False))
        print("[data] holdout subsets: " + ", ".join(f"{k} {int(v.sum())}" for k, v in self.subsets.items()))

    def summary(self) -> dict:
        return {
            "records": self.count, "window": self.window, "games": self.games,
            "train_samples": self.split, "holdout_samples": self.window - self.split,
            "holdout_games": self.holdout_games,
            "subset_sizes": {k: int(v.sum()) for k, v in self.subsets.items()},
        }


def _worker_init(_worker_id: int) -> None:
    # The D4 draw uses numpy; give every worker its own stream derived from the
    # loader's seed so a run is repeatable.
    np.random.seed(torch.initial_seed() % 2**32)


# ─── evaluation ──────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, ds, pick: np.ndarray, data: ProbeData, device, batch: int, workers: int,
             amp_dtype) -> dict:
    """Per-subset metrics on the holdout samples `pick` (indices into the holdout)."""
    model.eval()
    loader = DataLoader(Subset(ds, (data.hold_indices[pick]).tolist()), batch_size=batch,
                        shuffle=False, num_workers=workers, worker_init_fn=_worker_init)
    n = len(pick)
    ce = np.full(n, np.nan)
    top1 = np.full(n, np.nan)
    v_pred = np.zeros(n, dtype=np.float64)
    v_target = np.zeros(n, dtype=np.float64)
    at = 0
    for b in loader:
        planes, legal, pi, value_target, policy_valid = b[0], b[1], b[2], b[3], b[4]
        with torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else _null():
            out = model(planes.to(device))
        logits, value = out[0].float(), out[1].float()
        k = planes.shape[0]
        v_pred[at:at + k] = value.squeeze(1).cpu().numpy()
        v_target[at:at + k] = value_target.numpy()
        legal = legal.to(device)
        pi = pi.to(device)
        valid = (policy_valid.to(device) > 0.5) & legal.any(dim=1) & (pi.sum(dim=1) > 0)
        if bool(valid.any()):
            lg = logits[valid].masked_fill(~legal[valid], float("-inf"))
            logp = F.log_softmax(lg, dim=1)
            p = pi[valid]
            row_ce = -(p * torch.where(p > 0, logp, torch.zeros_like(logp))).sum(dim=1)
            hit = (lg.argmax(dim=1) == p.argmax(dim=1)).float()
            idx = at + torch.nonzero(valid).squeeze(1).cpu().numpy()
            ce[idx] = row_ce.cpu().numpy()
            top1[idx] = hit.cpu().numpy()
        at += k
    model.train()

    known = data.hold_known[pick]
    z = data.hold_z[pick]
    out: dict = {}
    for name, mask in data.subsets.items():
        m = mask[pick]
        pol = m & ~np.isnan(ce)
        val = m & known
        out[name] = {
            "n": int(m.sum()),
            "n_policy": int(pol.sum()),
            "n_value": int(val.sum()),
            "policy_ce": float(ce[pol].mean()) if pol.any() else None,
            "top1": float(top1[pol].mean()) if pol.any() else None,
            "value_mse": float(((v_pred - v_target)[val] ** 2).mean()) if val.any() else None,
            "value_mse_z": float(((v_pred - z)[val] ** 2).mean()) if val.any() else None,
        }
    return out


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ─── one run ─────────────────────────────────────────────────────────────────


def run_variant(vid: str, seed: int, data: ProbeData, args, out_path: Path) -> dict:
    planes, cheap_w = VARIANTS[vid]
    spec = parse_planes(planes)
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and not args.no_amp) else None

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = data.base.with_planes(planes, augment=True)
    hold_ds = data.base.with_planes(planes, augment=False)
    train_part = Subset(train_ds, range(data.split))

    model = TaflAlphaZeroNet(in_channels=spec.num_planes, board_size=11, trunk_channels=args.channels,
                             num_blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.steps if args.steps > 0 else max(1, int(args.epochs * data.split) // args.batch)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_part, batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, generator=gen, worker_init_fn=_worker_init,
                        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    if len(loader) == 0:
        raise SystemExit(f"{data.split} training samples do not fill one batch of {args.batch}")
    print(f"\n[{vid}-s{seed}] planes {planes} ({spec.num_planes}), cheap value weight {cheap_w}, "
          f"{steps} steps x {args.batch} on {device}")

    curve = []
    run_loss = run_p = run_v = 0.0
    run_n = 0
    t0 = time.time()
    t_train = 0.0
    step = 0
    while step < steps:
        for batch in loader:
            step += 1
            ts = time.time()
            total, p_loss, v_loss, *_ = compute_batch_loss(
                model, batch, device, 1.0, 0.0, 0.0, False, amp_dtype=amp_dtype,
                cheap_value_weight=cheap_w)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            t_train += time.time() - ts
            run_loss += total.item()
            run_p += p_loss.item()
            run_v += v_loss.item()
            run_n += 1
            if step % args.eval_every == 0 or step == steps:
                point = {
                    "step": step,
                    "train_loss": run_loss / run_n, "train_policy": run_p / run_n,
                    "train_value": run_v / run_n,
                    "elapsed_s": time.time() - t0,
                }
                if step < steps:
                    point["holdout"] = evaluate(model, hold_ds, data.curve_pick, data, device,
                                                args.eval_batch, args.workers, amp_dtype)
                    a = point["holdout"]["all"]
                    print(f"[{vid}-s{seed}] step {step}/{steps}  train p={point['train_policy']:.4f} "
                          f"v={point['train_value']:.4f}  holdout p={a['policy_ce']} top1={a['top1']} "
                          f"v={a['value_mse']}  {args.batch * step / (time.time() - t0):.0f} samples/s")
                curve.append(point)
                run_loss = run_p = run_v = 0.0
                run_n = 0
            if step >= steps:
                break

    final = evaluate(model, hold_ds, np.arange(len(data.hold_indices)), data, device,
                     args.eval_batch, args.workers, amp_dtype)
    curve[-1]["holdout"] = final
    a = final["all"]
    print(f"[{vid}-s{seed}] final  holdout p={a['policy_ce']} top1={a['top1']} v={a['value_mse']} "
          f"vz={a['value_mse_z']}  ({time.time() - t0:.0f} s)")
    result = {
        "variant": vid, "seed": seed, "planes": planes, "in_channels": spec.num_planes,
        "cheap_value_weight": cheap_w,
        "model": {"trunk_channels": args.channels, "num_blocks": args.blocks},
        "train": {"steps": steps, "batch": args.batch, "lr": args.lr, "weight_decay": args.weight_decay,
                  "z_lambda": args.z_lambda, "epochs": args.epochs, "amp": str(amp_dtype),
                  "device": str(device)},
        "data": data.summary() | {"data": str(args.data), "features": str(args.features)},
        "final": final,
        "curve": curve,
        "wall_s": time.time() - t0,
        "samples_per_s": args.batch * steps / max(1e-9, time.time() - t0),
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, out_path)
    return result


# ─── report ──────────────────────────────────────────────────────────────────


def _load_results(out_dir: Path) -> dict[str, list[dict]]:
    by_variant: dict[str, list[dict]] = {}
    for p in sorted(out_dir.glob("*-s*.json")):
        try:
            r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "variant" in r and "final" in r:
            by_variant.setdefault(r["variant"], []).append(r)
    return by_variant


def _fmt(x: float | None, digits: int = 4) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def write_report(out_dir: Path) -> Path:
    results = _load_results(out_dir)
    if "A" not in results:
        raise SystemExit(f"no A-s*.json in {out_dir}: the report is relative to A")

    def stat(rs: list[dict], subset: str, metric: str) -> tuple[float | None, float | None]:
        vals = [r["final"][subset][metric] for r in rs if r["final"][subset][metric] is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), (max(vals) - min(vals)) / 2.0

    cols = [(s, m) for s in SUBSETS for m in METRICS if stat(results["A"], s, m)[0] is not None]
    lines = ["# Input-plane probe", ""]
    a0 = results["A"][0]
    d = a0["data"]
    lines += [
        f"Data: `{d['data']}`, last {d['window']} samples ({d['games']} games); holdout "
        f"{d['holdout_samples']} samples from the last {d['holdout_games']} games. "
        f"Model {a0['model']['num_blocks']}x{a0['model']['trunk_channels']}, "
        f"{a0['train']['steps']} steps x {a0['train']['batch']}, lr {a0['train']['lr']}.",
        "",
        "Holdout subset sizes: " + ", ".join(f"{k} {v}" for k, v in d["subset_sizes"].items()) + ".",
        "",
        "Row A is absolute; other rows are % relative to A, `(x - A) / A`. For `policy_ce`, "
        "`value_mse`, `value_mse_z` negative is better; for `top1` positive is better. With "
        "several seeds a cell is mean ± half-range over seeds.",
        "",
    ]
    header = ["variant", "planes", "seeds"] + [f"{s} {m}" for s, m in cols]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for vid in VARIANTS:
        if vid not in results:
            continue
        rs = results[vid]
        planes, cheap_w = VARIANTS[vid]
        label = planes + ("" if cheap_w == 1.0 else f", cheap value weight {cheap_w:g}")
        cells = [vid, f"`{label}`", ",".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))]
        for s, m in cols:
            mean, half = stat(rs, s, m)
            if vid == "A":
                cell = _fmt(mean) + (f" ± {_fmt(half)}" if len(rs) > 1 and half else "")
            else:
                base, _ = stat(results["A"], s, m)
                if mean is None or not base:
                    cell = "–"
                else:
                    cell = f"{100.0 * (mean - base) / base:+.2f}%"
                    if len(rs) > 1 and half is not None:
                        cell += f" ± {100.0 * half / abs(base):.2f}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def stage2_variants(out_dir: Path) -> list[str]:
    """A, plus every stage-1 variant whose `all` policy loss is more than 1% from A's."""
    results = _load_results(out_dir)
    s1 = {v: [r for r in rs if r["seed"] == 1] for v, rs in results.items()}
    if not s1.get("A"):
        raise SystemExit(f"--stage 2 needs A-s1.json in {out_dir}")
    base = s1["A"][0]["final"]["all"]["policy_ce"]
    out = ["A"]
    for vid in VARIANTS:
        if vid == "A" or not s1.get(vid):
            continue
        p = s1[vid][0]["final"]["all"]["policy_ce"]
        if p is not None and abs(p - base) / base > STAGE2_POLICY_GAP:
            out.append(vid)
    return out


# ─── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="self-play sample file (e.g. runs/T6-10x128/data/worker0.bin)")
    ap.add_argument("--features", type=Path, help="side file from `taflzero --features` for --data")
    ap.add_argument("--out", type=Path, required=True, help="output directory for the json files and report.md")
    ap.add_argument("--report", action="store_true", help="only read the json files in --out and write report.md")
    ap.add_argument("--variants", type=str, default="A,I,G,B,C,D,E,F,H,J",
                    help="comma list of variant ids (default: all, A, I, G first)")
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1,
                    help="2 = second seed for A and every variant whose `all` policy loss is >1%% from A's")
    ap.add_argument("--seed", type=int, default=None, help="seed of the runs (default: the stage number)")
    ap.add_argument("--resume", action="store_true", help="skip runs whose json already exists")
    ap.add_argument("--samples", type=int, default=6_000_000, help="last N samples of the file, cut at a game boundary")
    ap.add_argument("--holdout-frac", type=float, default=0.05, help="share of the window's games held out (the last ones)")
    ap.add_argument("--epochs", type=float, default=2.0, help="passes over the training part (sets the step count)")
    ap.add_argument("--steps", type=int, default=0, help="training steps; 0 = epochs x train samples / batch")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--z-lambda", type=float, default=0.5)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=2000, help="holdout curve point every N steps")
    ap.add_argument("--curve-samples", type=int, default=50_000,
                    help="holdout samples scored at curve points (fixed random subset; 0 = all); "
                         "the final evaluation always uses the whole holdout")
    ap.add_argument("--eval-batch", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true", help="fp32 forward on CUDA (default bf16 autocast)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.report:
        print(f"wrote {write_report(args.out)}")
        return
    if args.data is None or args.features is None:
        ap.error("--data and --features are needed unless --report")

    if args.stage == 2:
        variants = stage2_variants(args.out)
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        ap.error(f"unknown variant(s) {unknown}; known: {', '.join(VARIANTS)}")
    seed = args.seed if args.seed is not None else args.stage

    todo = []
    for vid in variants:
        path = args.out / f"{vid}-s{seed}.json"
        if args.resume and path.exists():
            print(f"[{vid}-s{seed}] {path} exists, skipped")
            continue
        todo.append((vid, path))
    if not todo:
        print("nothing to run")
        return

    data = ProbeData(args.data, args.features, args.samples, args.holdout_frac, args.z_lambda,
                     args.curve_samples)
    for vid, path in todo:
        run_variant(vid, seed, data, args, path)
        print(f"[{vid}-s{seed}] wrote {path}")
    if any(args.out.glob("A-s*.json")):
        print(f"wrote {write_report(args.out)}")


if __name__ == "__main__":
    main()
