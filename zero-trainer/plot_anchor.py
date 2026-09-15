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

# A single generation carries roughly +-30 Elo of match noise, wide enough to hide a
# trend of a few Elo per generation. The rolling mean is the line to read; the raw
# points stay faint behind it so an outlier is still visible.
args = list(sys.argv[1:])
window = 5
if "--window" in args:
    i = args.index("--window")
    window = max(1, int(args[i + 1]))
    del args[i : i + 2]

csv_path = Path(args[0]) if args else Path(__file__).parent / "weights" / "anchor-results.csv"


def rolling_mean(values, size):
    """Trailing mean; early points average over whatever is available."""
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - size + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


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

    smooth = rolling_mean(elos, window)

    ax1.cla()
    ax1.plot(x, elos, marker="o", markersize=4, linewidth=1, color="blue",
             alpha=0.30, label="Elo vs anchor (per generation)")
    ax1.plot(x, smooth, linewidth=2.5, color="darkblue",
             label=f"rolling mean over {window}")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    if x:
        ax1.annotate(f"{smooth[-1]:+.0f}", (x[-1], smooth[-1]),
                     textcoords="offset points", xytext=(6, 0), ha="left",
                     fontsize=11, fontweight="bold", color="darkblue")
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
