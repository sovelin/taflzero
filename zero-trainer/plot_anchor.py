#!/usr/bin/env python3
"""Plot anchor Elo results from anchor-results.csv — auto-refreshes every 5s"""
import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    print("pip install matplotlib")
    sys.exit(1)

csv_path = Path(__file__).parent / "weights" / "anchor-results.csv"
if len(sys.argv) > 1:
    csv_path = Path(sys.argv[1])


def load_data():
    rows = []
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rows.append(row)
    except FileNotFoundError:
        return [], [], [], [], [], []
    gens = [r["generation"] for r in rows]
    elos = [float(r["elo_vs_anchor"]) for r in rows]
    scores = [float(r["score_pct"]) for r in rows]
    wins = [int(r["wins"]) for r in rows]
    losses = [int(r["losses"]) for r in rows]
    draws = [int(r["draws"]) for r in rows]
    return gens, elos, scores, wins, losses, draws


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Anchor Elo Progress", fontsize=14, fontweight="bold")


def draw(frame):
    gens, elos, scores, wins, losses, draws = load_data()
    if not gens:
        return

    x = list(range(len(gens)))

    ax1.cla()
    ax1.plot(x, elos, marker="o", linewidth=2, color="blue", label="Elo vs anchor")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    for xi, e in zip(x, elos):
        ax1.annotate(f"{e:+.0f}", (xi, e), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax1.set_ylabel("Elo vs anchor")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    bottoms_l = [w + d for w, d in zip(wins, draws)]
    ax2.cla()
    ax2.bar(x, wins, label="Wins", color="green", alpha=0.8)
    ax2.bar(x, draws, bottom=wins, label="Draws", color="gray", alpha=0.8)
    ax2.bar(x, losses, bottom=bottoms_l, label="Losses", color="red", alpha=0.8)
    for xi, w, d, l in zip(x, wins, draws, losses):
        # Label inside each segment, centered vertically
        if w > 20:
            ax2.text(xi, w / 2, f"W:{w}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if d > 20:
            ax2.text(xi, w + d / 2, f"D:{d}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if l > 20:
            ax2.text(xi, w + d + l / 2, f"L:{l}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax2.set_ylabel("Game count")
    ax2.set_xticks(x)
    ax2.set_xticklabels(gens, rotation=30, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()


ani = animation.FuncAnimation(fig, draw, interval=5000, cache_frame_data=False)
draw(0)
plt.show()
