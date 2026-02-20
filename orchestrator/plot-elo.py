"""Live plot of cumulative Elo from sprt-results.csv"""

import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "zero-trainer" / "weights" / "sprt-results.csv"


def read_data():
    if not CSV_PATH.exists():
        return [], []
    gens = []
    cum_elo = 0
    elos = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gens.append(row["generation"])
            cum_elo += float(row["elo"])
            elos.append(cum_elo)
    return gens, elos


fig, ax = plt.subplots(figsize=(10, 5))
fig.canvas.manager.set_window_title("Cumulative Elo")


def update(_frame):
    gens, elos = read_data()
    ax.clear()
    if gens:
        ax.plot(range(len(gens)), elos, "o-", markersize=4)
        ax.set_xticks(range(len(gens)))
        ax.set_xticklabels(gens, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cumulative Elo")
    ax.set_title(f"Cumulative Elo ({len(gens)} generations)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


ani = animation.FuncAnimation(fig, update, interval=5000)
plt.show()
