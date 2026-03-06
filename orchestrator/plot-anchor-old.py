"""Live plot of absolute Elo from anchor-results.csv"""

import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ---------- CLI arguments ----------
parser = argparse.ArgumentParser(description="Live Elo plotter")
parser.add_argument(
    "-f", "--file",
    type=Path,
    default=Path(__file__).parent.parent / "zero-trainer" / "weights" / "anchor-results.csv",
    help="Path to anchor-results.csv"
)

parser.add_argument(
    "--interval",
    type=int,
    default=5000,
    help="Refresh interval in milliseconds (default: 5000)"
)

args = parser.parse_args()
CSV_PATH = args.file

print(f"Reading data from: {CSV_PATH.resolve()}")


# ---------- Data reading ----------
def read_data():
    if not CSV_PATH.exists():
        return [], []

    gens = []
    elos = []

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gens.append(row["generation"])
                elos.append(float(row["elo_vs_anchor"]))
            except (KeyError, ValueError):
                # файл может писаться в этот момент
                continue

    return gens, elos


# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(10, 5))
fig.canvas.manager.set_window_title("Elo vs Anchor")


def update(_frame):
    gens, elos = read_data()
    ax.clear()

    if gens:
        ax.plot(range(len(gens)), elos, "o-", markersize=4)
        ax.set_xticks(range(len(gens)))
        ax.set_xticklabels(gens, rotation=45, ha="right", fontsize=7)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Elo vs Anchor")
    ax.set_title(f"Elo vs Anchor ({len(gens)} generations)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


ani = animation.FuncAnimation(fig, update, interval=args.interval)
plt.show()