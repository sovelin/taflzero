#!/usr/bin/env python3
"""
Policy Diversity Logger & Plotter
Tracks n_eff, entropy_norm, max_prob over training to detect mode collapse.

Usage:
  python diversity_logger.py --window 100000 --step 500000 --plot
  python diversity_logger.py --tail 5  # analyze last 5 windows
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

# ANSI color codes
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

BIT_POSITION_BYTES = 49
LEGAL_MASK_BYTES = 605


def popcount_bytes(b: bytes) -> int:
    return sum(int(x).bit_count() for x in b)


def iter_samples(fh):
    """Iterate over samples in selfplay.bin"""
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
        yield stm[0], legal, policy_raw, policy_len


def analyze_window(samples_iter, window_size: int, policy_temp: float) -> dict[str, Any] | None:
    """Analyze a window of samples and return metrics"""
    entropies = []
    entropies_norm = []
    max_probs = []
    n_effs = []
    legal_counts = []

    side_stats = {
        0: {"entropies": [], "entropies_norm": [], "max_probs": [], "n_effs": []},
        1: {"entropies": [], "entropies_norm": [], "max_probs": [], "n_effs": []},
    }

    count = 0
    for stm, legal, policy_raw, policy_len in samples_iter:
        if count >= window_size:
            break
        count += 1

        legal_moves = popcount_bytes(legal)
        legal_counts.append(legal_moves)

        if policy_len == 0:
            continue

        # Extract visits
        visits = []
        for i in range(policy_len):
            off = i * 4
            v = struct.unpack_from("<H", policy_raw, off + 2)[0]
            if v > 0:
                visits.append(float(v))

        if not visits:
            continue

        # Apply temperature
        if policy_temp != 1.0:
            power = 1.0 / policy_temp
            visits = [v**power for v in visits]

        s = sum(visits)
        if s <= 0:
            continue

        probs = [v / s for v in visits]
        probs.sort(reverse=True)

        p1 = probs[0]
        h = sum(-p * math.log(p) for p in probs)

        entropies.append(h)
        if legal_moves > 1:
            entropies_norm.append(h / math.log(legal_moves))
        else:
            entropies_norm.append(0.0)
        max_probs.append(p1)
        n_effs.append(math.exp(h))

        # By side
        if stm in side_stats:
            sstats = side_stats[stm]
            sstats["entropies"].append(h)
            if legal_moves > 1:
                sstats["entropies_norm"].append(h / math.log(legal_moves))
            else:
                sstats["entropies_norm"].append(0.0)
            sstats["max_probs"].append(p1)
            sstats["n_effs"].append(math.exp(h))

    if not n_effs:
        return None

    return {
        "samples": count,
        "n_eff": mean(n_effs),
        "entropy_norm": mean(entropies_norm),
        "max_prob": mean(max_probs),
        "legal_moves": mean(legal_counts),
        "side0_n_eff": mean(side_stats[0]["n_effs"]) if side_stats[0]["n_effs"] else 0,
        "side1_n_eff": mean(side_stats[1]["n_effs"]) if side_stats[1]["n_effs"] else 0,
    }


def get_status_indicator(metric_name: str, value: float) -> tuple[str, str]:
    """Return (emoji, color) based on metric health"""
    if metric_name == "n_eff":
        if value >= 2.7:
            return "✅", GREEN
        elif value >= 2.3:
            return "⚠️ ", YELLOW
        else:
            return "❗", RED
    elif metric_name == "entropy_norm":
        if value >= 0.15:
            return "✅", GREEN
        elif value >= 0.12:
            return "⚠️ ", YELLOW
        else:
            return "❗", RED
    elif metric_name == "max_prob":
        if value <= 0.75:
            return "✅", GREEN
        elif value <= 0.85:
            return "⚠️ ", YELLOW
        else:
            return "❗", RED
    return "  ", RESET


def print_metrics(window_idx: int, start_sample: int, end_sample: int, metrics: dict):
    """Print metrics with colored indicators"""
    n_eff_emoji, n_eff_color = get_status_indicator("n_eff", metrics["n_eff"])
    enorm_emoji, enorm_color = get_status_indicator("entropy_norm", metrics["entropy_norm"])
    maxp_emoji, maxp_color = get_status_indicator("max_prob", metrics["max_prob"])

    print(f"\n{BOLD}Window #{window_idx}{RESET} [{start_sample:,} - {end_sample:,}]")
    print(
        f"  {n_eff_emoji} n_eff:        {n_eff_color}{metrics['n_eff']:.3f}{RESET} "
        f"(side0: {metrics['side0_n_eff']:.2f}, side1: {metrics['side1_n_eff']:.2f})"
    )
    print(f"  {enorm_emoji} entropy_norm: {enorm_color}{metrics['entropy_norm']:.3f}{RESET}")
    print(f"  {maxp_emoji} max_prob:     {maxp_color}{metrics['max_prob']:.3f}{RESET}")


def save_log_entry(log_file: Path, window_idx: int, position: int, metrics: dict):
    """Append metrics to JSON log file"""
    entry = {
        "window": window_idx,
        "position": position,
        "n_eff": metrics["n_eff"],
        "entropy_norm": metrics["entropy_norm"],
        "max_prob": metrics["max_prob"],
        "side0_n_eff": metrics["side0_n_eff"],
        "side1_n_eff": metrics["side1_n_eff"],
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def plot_metrics(log_file: Path, output_file: Path):
    """Generate matplotlib plot from log file"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"{RED}matplotlib not installed. Run: pip install matplotlib{RESET}")
        return

    # Load data
    data = []
    with open(log_file) as f:
        for line in f:
            data.append(json.loads(line))

    if not data:
        print("No data to plot")
        return

    positions = [d["position"] / 1e6 for d in data]  # Convert to millions
    n_eff = [d["n_eff"] for d in data]
    side0_n_eff = [d["side0_n_eff"] for d in data]
    side1_n_eff = [d["side1_n_eff"] for d in data]
    entropy_norm = [d["entropy_norm"] for d in data]
    max_prob = [d["max_prob"] for d in data]

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("Policy Diversity Over Training", fontsize=16, fontweight="bold")

    # Plot 1: n_eff
    ax = axes[0]
    ax.plot(positions, n_eff, label="Overall", linewidth=2.5, color="black", marker="o")
    ax.plot(positions, side0_n_eff, label="Attackers (side=0)", linewidth=2, alpha=0.7, color="blue", marker="^")
    ax.plot(positions, side1_n_eff, label="Defenders (side=1)", linewidth=2, alpha=0.7, color="red", marker="v")
    ax.axhline(3.0, color="green", linestyle=":", alpha=0.6, linewidth=1.5, label="Healthy (3.0)")
    ax.axhline(2.7, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, label="Monitor (2.7)")
    ax.axhline(2.3, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="Warning (2.3)")
    ax.set_ylabel("n_eff (effective moves)", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0, min(n_eff) - 0.5))

    # Plot 2: entropy_norm
    ax = axes[1]
    ax.plot(positions, entropy_norm, linewidth=2.5, color="purple", marker="o")
    ax.axhline(0.20, color="green", linestyle=":", alpha=0.6, linewidth=1.5, label="Healthy (0.20)")
    ax.axhline(0.15, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, label="Monitor (0.15)")
    ax.axhline(0.12, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="Warning (0.12)")
    ax.set_ylabel("Entropy (normalized)", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: max_prob
    ax = axes[2]
    ax.plot(positions, max_prob, linewidth=2.5, color="darkgreen", marker="o")
    ax.axhline(0.70, color="green", linestyle=":", alpha=0.6, linewidth=1.5, label="Healthy (0.70)")
    ax.axhline(0.75, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, label="Monitor (0.75)")
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="Warning (0.85)")
    ax.set_ylabel("Max Probability", fontsize=11, fontweight="bold")
    ax.set_xlabel("Training Samples (millions)", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n{GREEN}✓{RESET} Plot saved to: {output_file}")


def main():
    ap = argparse.ArgumentParser(description="Log and plot policy diversity over training")
    ap.add_argument("--path", default="zero-trainer/selfplay.bin", help="Path to selfplay.bin")
    ap.add_argument("--window", type=int, default=100_000, help="Samples per window (default: 100k)")
    ap.add_argument("--step", type=int, default=500_000, help="Step between windows (default: 500k)")
    ap.add_argument("--policy-temp", type=float, default=0.5, help="Policy temperature (default: 0.5)")
    ap.add_argument("--log-file", default="zero-trainer/diversity_log.jsonl", help="Output log file")
    ap.add_argument("--plot", action="store_true", help="Generate plot after logging")
    ap.add_argument("--plot-only", action="store_true", help="Only generate plot from existing log")
    ap.add_argument("--plot-output", default="zero-trainer/diversity_plot.png", help="Plot output file")
    ap.add_argument("--tail", type=int, default=0, help="Only analyze last N windows (0 = all)")
    ap.add_argument("--clear-log", action="store_true", help="Clear existing log before running")
    args = ap.parse_args()

    log_file = Path(args.log_file)
    plot_file = Path(args.plot_output)

    # Plot only mode
    if args.plot_only:
        if not log_file.exists():
            print(f"{RED}Error: Log file not found: {log_file}{RESET}")
            sys.exit(1)
        plot_metrics(log_file, plot_file)
        return

    # Clear log if requested
    if args.clear_log and log_file.exists():
        log_file.unlink()
        print(f"Cleared existing log: {log_file}")

    # Count total samples
    print(f"Scanning {args.path}...")
    with open(args.path, "rb") as f:
        total_samples = sum(1 for _ in iter_samples(f))
    print(f"Total samples: {total_samples:,}")

    # Determine windows to analyze
    if args.tail and args.tail > 0:
        # Analyze last N windows
        last_position = total_samples
        start_position = max(0, last_position - (args.tail * args.step))
        print(f"Analyzing last {args.tail} windows (from sample {start_position:,})")
    else:
        start_position = 0

    # Analyze windows
    window_idx = 0
    position = start_position

    print(f"\n{BOLD}=== Policy Diversity Analysis ==={RESET}")
    print(f"Window size: {args.window:,} samples")
    print(f"Step: {args.step:,} samples")
    print(f"Policy temp: {args.policy_temp}")

    while position < total_samples:
        # Open file and seek to position
        with open(args.path, "rb") as f:
            # Skip to position
            skip_count = 0
            samples_iter = iter_samples(f)
            for _ in range(position):
                try:
                    next(samples_iter)
                    skip_count += 1
                except StopIteration:
                    break

            if skip_count < position:
                break

            # Analyze window
            metrics = analyze_window(samples_iter, args.window, args.policy_temp)

            if metrics:
                end_sample = min(position + args.window, total_samples)
                print_metrics(window_idx, position, end_sample, metrics)
                save_log_entry(log_file, window_idx, position + args.window // 2, metrics)
                window_idx += 1

        position += args.step

    print(f"\n{GREEN}✓{RESET} Logged {window_idx} windows to: {log_file}")

    # Generate plot
    if args.plot:
        plot_metrics(log_file, plot_file)

    # Summary
    print(f"\n{BOLD}Thresholds:{RESET}")
    print(f"  n_eff:        {GREEN}> 2.7 ✅{RESET}  |  {YELLOW}2.3-2.7 ⚠️ {RESET}  |  {RED}< 2.3 ❗{RESET}")
    print(f"  entropy_norm: {GREEN}> 0.15 ✅{RESET} |  {YELLOW}0.12-0.15 ⚠️ {RESET} |  {RED}< 0.12 ❗{RESET}")
    print(f"  max_prob:     {GREEN}< 0.75 ✅{RESET} |  {YELLOW}0.75-0.85 ⚠️ {RESET} |  {RED}> 0.85 ❗{RESET}")


if __name__ == "__main__":
    main()
