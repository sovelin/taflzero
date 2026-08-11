import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

ROOT = os.path.dirname(__file__)

def parse_gen(s):
    return int(s.replace("gen", "").lstrip("0") or "0")

def read_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["elo_vs_anchor"])
    df["gen"] = df["generation"].apply(parse_gen)
    return df

# ── Chain all phases ──────────────────────────────────────────────────────────
phases = []

# Phase 1: vs random (gen1-23)
df1 = read_csv(os.path.join(ROOT, "weights/anchor-results-random.csv"))
base1 = 0.0
phases.append(("vs random", df1["gen"].tolist(), (base1 + df1["elo_vs_anchor"]).tolist()))

# Phase 2: vs gen23 (gen24-32)
base2 = df1["elo_vs_anchor"].iloc[-1]  # 851
df2 = read_csv(os.path.join(ROOT, "weights/anchor-results-gen0023.csv"))
phases.append(("vs gen23", df2["gen"].tolist(), (base2 + df2["elo_vs_anchor"]).tolist()))

# Phase 3: vs gen32 (gen33+)
base3 = base2 + df2[df2["gen"] == 32]["elo_vs_anchor"].values[0]  # 851 + 768
df3 = read_csv(os.path.join(ROOT, "weights/anchor-results.csv"))
if not df3.empty:
    phases.append(("vs gen32", df3["gen"].tolist(), (base3 + df3["elo_vs_anchor"]).tolist()))

# ── Flatten all phases into one line ─────────────────────────────────────────
all_gens, all_elos = [], []
for _, gens, elos in phases:
    all_gens.extend(gens)
    all_elos.extend(elos)

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS = ["#1565C0", "#0288D1", "#00838F"]

fig, ax = plt.subplots(figsize=(13, 7))

ax.plot(all_gens, all_elos, "o-", color="#1565C0", markersize=5, linewidth=2.2)

# Phase boundary lines with labels
phase_colors = ["#1565C0", "#0288D1", "#00838F"]
for i, (label, gens, elos) in enumerate(phases):
    mid = gens[len(gens) // 2]
    mid_elo = elos[len(elos) // 2]
    ax.axvline(x=gens[0], color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(gens[0] + 0.4, 80, label, color=phase_colors[i], fontsize=9, alpha=0.8)

# Annotate last point
last_gen = all_gens[-1]
last_elo = all_elos[-1]
ax.annotate(f"gen{last_gen}\n{last_elo:.0f} Elo",
            xy=(last_gen, last_elo),
            xytext=(last_gen + 1.5, last_elo - 120),
            arrowprops=dict(arrowstyle="->", color="#00838F"),
            fontsize=10, color="#00838F", fontweight="bold")

# Annotate peak
peak_gen = all_gens[all_elos.index(max(all_elos))]
peak_elo = max(all_elos)
if peak_gen != last_gen:
    ax.annotate(f"peak: {peak_elo:.0f}",
                xy=(peak_gen, peak_elo),
                xytext=(peak_gen + 1.5, peak_elo + 50),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9, color="gray")

ax.set_title("Текущий ран — накопительный Elo (11 плоскостей, BFS fix)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Generation", fontsize=12)
ax.set_ylabel("Elo vs random (chained)", fontsize=12)
ax.legend([], [], frameon=False)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

plt.tight_layout()
out = os.path.join(ROOT, "weights/current_elo.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
