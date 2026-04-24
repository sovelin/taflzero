#!/usr/bin/env python3
"""Plot final_val_loss from generation JSON files — auto-refreshes every 5s"""
import json
import re
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    print("pip install matplotlib")
    sys.exit(1)

weights_dir = Path(__file__).parent / "weights"
if len(sys.argv) > 1:
    weights_dir = Path(sys.argv[1])


def load_data():
    entries = []
    for f in weights_dir.glob("gen*.json"):
        m = re.search(r"gen(\d+)", f.stem)
        if not m:
            continue
        try:
            data = json.loads(f.read_text())
            loss = data.get("final_val_loss")
            if loss is not None and loss != float("inf"):
                entries.append((int(m.group(1)), loss))
        except Exception:
            continue
    entries.sort()
    if not entries:
        return [], []
    gens, losses = zip(*entries)
    return list(gens), list(losses)


fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("Val Loss Progress")


def update(_):
    gens, losses = load_data()
    ax.clear()
    if not gens:
        ax.set_title("No data")
        return
    ax.plot(gens, losses, "b-o", markersize=3, linewidth=1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("final_val_loss")
    ax.set_title(f"Val Loss — latest gen{gens[-1]:04d}: {losses[-1]:.4f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


update(None)
ani = animation.FuncAnimation(fig, update, interval=5000)
plt.show()
