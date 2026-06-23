import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

ROOT = os.path.dirname(__file__)

def parse_gen(s):
    return int(s.replace("gen", "").lstrip("0") or "0")

def read_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["elo_vs_anchor"])
    df["gen"] = df["generation"].apply(parse_gen)
    return df

# ── Current run ───────────────────────────────────────────────────────────────
cur_phases = []

df = read_csv(os.path.join(ROOT, "weights/anchor-results-random.csv"))
base = 0.0
cur_phases.append(("vs random", df["gen"].tolist(), (base + df["elo_vs_anchor"]).tolist()))
base += df["elo_vs_anchor"].iloc[-1]  # gen23 = 851

df = read_csv(os.path.join(ROOT, "weights/anchor-results.csv"))
if not df.empty:
    cur_phases.append(("vs gen23", df["gen"].tolist(), (base + df["elo_vs_anchor"]).tolist()))

# ── modern-6x64 ───────────────────────────────────────────────────────────────
mod_phases = []

df = read_csv(os.path.join(ROOT, "weights/modern-6x64/anchor-results-random.csv"))
base = 0.0
mod_phases.append(("vs random", df["gen"].tolist(), (base + df["elo_vs_anchor"]).tolist()))
base += df["elo_vs_anchor"].iloc[-1]  # gen21 = 921.5

df21 = read_csv(os.path.join(ROOT, "weights/modern-6x64/anchor-results-gen0021.csv"))
mod_phases.append(("vs gen21", df21["gen"].tolist(), (base + df21["elo_vs_anchor"]).tolist()))
base56 = base + df21[df21["gen"] == 56]["elo_vs_anchor"].values[0]

df56 = read_csv(os.path.join(ROOT, "weights/modern-6x64/anchor-results.csv"))
mod_phases.append(("vs gen56", df56["gen"].tolist(), (base56 + df56["elo_vs_anchor"]).tolist()))

# ── Loss data ─────────────────────────────────────────────────────────────────
def read_loss(folder, max_gen=None):
    gens, losses = [], []
    i = 1
    while True:
        path = os.path.join(ROOT, folder, f"gen{i:04d}.candidate.json")
        if not os.path.exists(path):
            break
        import json
        with open(path) as f:
            d = json.load(f)
        gens.append(i)
        losses.append(d["final_val_loss"])
        i += 1
        if max_gen and i > max_gen:
            break
    return gens, losses

cur_loss_gens, cur_loss = read_loss("weights")
mod_loss_gens, mod_loss = read_loss("weights/modern-6x64")

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS_CUR = ["#1565C0", "#0288D1"]
COLORS_MOD = ["#B71C1C", "#E53935", "#FF7043"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 11))

# ── Graph 1: Elo ──────────────────────────────────────────────────────────────
for i, (label, gens, elos) in enumerate(cur_phases):
    ax1.plot(gens, elos, "o-", color=COLORS_CUR[i], label=f"Current: {label}",
             markersize=5, linewidth=2.2)

for i, (label, gens, elos) in enumerate(mod_phases):
    ax1.plot(gens, elos, "s--", color=COLORS_MOD[i], label=f"modern-6x64: {label}",
             markersize=4, linewidth=1.8, alpha=0.85)

ax1.set_title("Абсолютный Elo (цепочка якорей)", fontsize=14, fontweight="bold")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Elo vs random (chained)")
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax1.set_ylim(bottom=0)

# Annotate last known points
last_cur_gen = cur_phases[-1][1][-1]
last_cur_elo = cur_phases[-1][2][-1]
ax1.annotate(f"gen{last_cur_gen}\n{last_cur_elo:.0f} Elo",
             xy=(last_cur_gen, last_cur_elo), xytext=(last_cur_gen + 3, last_cur_elo - 80),
             arrowprops=dict(arrowstyle="->", color=COLORS_CUR[-1]), fontsize=9,
             color=COLORS_CUR[-1])

last_mod_gen = mod_phases[1][1][-1]
last_mod_elo = mod_phases[1][2][-1]

# ── Graph 2: Loss ─────────────────────────────────────────────────────────────
ax2.plot(cur_loss_gens, cur_loss, "o-", color=COLORS_CUR[0], label="Current (11 planes)",
         markersize=4, linewidth=2)
ax2.plot(mod_loss_gens, mod_loss, "s--", color=COLORS_MOD[0], label="modern-6x64 (6 planes)",
         markersize=4, linewidth=1.8, alpha=0.85)

ax2.set_title("Val Loss по генерациям", fontsize=14, fontweight="bold")
ax2.set_xlabel("Generation")
ax2.set_ylabel("Final val loss")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(3.0, 5.5)
ax2.axhline(y=3.5, color="gray", linestyle=":", alpha=0.6, label="~3.5 (старый потолок)")

plt.suptitle("Current run (11 planes, BFS fix) vs modern-6x64 (6 planes)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()

out = os.path.join(ROOT, "weights/elo_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
