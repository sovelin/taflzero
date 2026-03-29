#!/usr/bin/env python3
"""
Reads selfplay.bin.gamelog and shows win/draw stats in chunks.

Usage:
  python plot_gamelog.py [--path PATH] [--step N] [--tail N] [--plot]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

ANSI = {
    "green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m",
    "blue": "\033[94m", "gray": "\033[90m", "bold": "\033[1m", "reset": "\033[0m",
}

TERMINAL_COLORS = {
    "atk_capture":    ("ATK", "red"),
    "atk_threefold":  ("ATK", "red"),
    "atk_surrounded": ("ATK", "red"),
    "atk_no_moves":   ("ATK", "red"),
    "def_corner":     ("DEF", "green"),
    "def_fort":       ("DEF", "green"),
    "def_no_moves":   ("DEF", "green"),
    "draw_nocapture": ("DRAW", "yellow"),
    "draw_limit":     ("DRAW", "yellow"),
    "draw_threefold": ("DRAW", "yellow"),
}

ALL_TYPES = list(TERMINAL_COLORS.keys())


def load_gamelog(path: Path):
    games = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            terminal, length = parts[0], int(parts[1])
            games.append((terminal, length))
    return games


def print_chunk(chunk_idx: int, start: int, end: int, games: list):
    counts = defaultdict(int)
    total_len = 0
    for t, l in games:
        counts[t] += 1
        total_len += l

    n = len(games)
    atk = sum(v for k, v in counts.items() if TERMINAL_COLORS.get(k, ("?",))[0] == "ATK")
    def_ = sum(v for k, v in counts.items() if TERMINAL_COLORS.get(k, ("?",))[0] == "DEF")
    draw = sum(v for k, v in counts.items() if TERMINAL_COLORS.get(k, ("?",))[0] == "DRAW")
    avg_len = total_len / n if n else 0

    b, r = ANSI["bold"], ANSI["reset"]
    print(f"\n{b}Chunk #{chunk_idx}{r} [games {start+1}–{end}]  avg_len={avg_len:.0f}")

    atk_pct = atk / n * 100 if n else 0
    def_pct = def_ / n * 100 if n else 0
    draw_pct = draw / n * 100 if n else 0

    print(f"  {ANSI['red']}ATK {atk:4d} ({atk_pct:5.1f}%){r}  "
          f"{ANSI['green']}DEF {def_:4d} ({def_pct:5.1f}%){r}  "
          f"{ANSI['yellow']}DRAW {draw:4d} ({draw_pct:5.1f}%){r}")

    # Detail per terminal type
    for t in ALL_TYPES:
        c = counts.get(t, 0)
        if c == 0:
            continue
        _, color = TERMINAL_COLORS[t]
        pct = c / n * 100
        print(f"    {ANSI[color]}{t:<20}{r} {c:4d}  ({pct:5.1f}%)")


def plot_chunks(all_chunks, step):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("pip install matplotlib")
        return

    xs = [i * step for i in range(len(all_chunks))]
    atk_pcts, def_pcts, draw_pcts = [], [], []
    fort_pcts = []

    for games in all_chunks:
        n = len(games)
        counts = defaultdict(int)
        for t, _ in games:
            counts[t] += 1
        atk = sum(v for k, v in counts.items() if TERMINAL_COLORS.get(k, ("?",))[0] == "ATK")
        def_ = sum(v for k, v in counts.items() if TERMINAL_COLORS.get(k, ("?",))[0] == "DEF")
        draw = sum(v for k, v in counts.items() if TERMINAL_COLORS.get(k, ("?",))[0] == "DRAW")
        atk_pcts.append(atk / n * 100 if n else 0)
        def_pcts.append(def_ / n * 100 if n else 0)
        draw_pcts.append(draw / n * 100 if n else 0)
        fort_pcts.append(counts["def_fort"] / n * 100 if n else 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Self-play game outcomes over training", fontsize=13, fontweight="bold")
    ax.plot(xs, atk_pcts, color="red", linewidth=2, label="ATK wins %")
    ax.plot(xs, def_pcts, color="green", linewidth=2, label="DEF wins %")
    ax.plot(xs, draw_pcts, color="gray", linewidth=2, label="Draws %")
    ax.plot(xs, fort_pcts, color="blue", linewidth=1.5, linestyle="--", label="Fort wins %")
    ax.set_xlabel(f"Games (step={step})")
    ax.set_ylabel("Percentage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="zero-trainer/selfplay.bin.gamelog")
    ap.add_argument("--step", type=int, default=10, help="Games per chunk")
    ap.add_argument("--tail", type=int, default=0, help="Show only last N chunks (0=all)")
    ap.add_argument("--plot", action="store_true", help="Show matplotlib plot")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    games = load_gamelog(path)
    print(f"Total games: {len(games)}")

    # Split into chunks
    all_chunks = []
    for i in range(0, len(games), args.step):
        all_chunks.append(games[i:i + args.step])

    chunks_to_show = all_chunks
    offset = 0
    print(args.tail)
    if args.tail > 0:
        chunks_to_show = all_chunks[-args.tail:]
        offset = len(all_chunks) - len(chunks_to_show)

    for i, chunk in enumerate(chunks_to_show):
        idx = offset + i
        start = idx * args.step
        end = min(start + len(chunk), len(games))
        print_chunk(idx, start, end, chunk)

    if args.plot:
        plot_chunks_live(path, args.step)


SEGMENTS = [
    ("atk_capture",    "#e74c3c", "atk_capture"),
    ("atk_surrounded", "#e67e22", "atk_surround"),
    ("atk_threefold",  "#c0392b", "atk_3fold"),
    ("def_corner",     "#27ae60", "def_corner"),
    ("def_fort",       "#2980b9", "def_fort"),
    ("draw_nocapture", "#95a5a6", "draw_nocap"),
    ("draw_limit",     "#7f8c8d", "draw_limit"),
    ("draw_threefold", "#bdc3c7", "draw_3fold"),
]


def plot_chunks_live(path: Path, step: int):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np
    except ImportError:
        print("pip install matplotlib numpy")
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle("Self-play game outcomes over training", fontsize=13, fontweight="bold")

    def draw(frame):
        games = load_gamelog(path)
        if not games:
            return

        all_chunks = [games[i:i + step] for i in range(0, len(games), step)]
        xs = list(range(len(all_chunks)))
        xlabels = [str(i * step) for i in xs]

        # counts per segment per chunk
        seg_counts = {key: [] for key, _, _ in SEGMENTS}
        totals = []
        for chunk in all_chunks:
            n = len(chunk)
            totals.append(n)
            counts = defaultdict(int)
            for t, _ in chunk:
                counts[t] += 1
            for key, _, _ in SEGMENTS:
                seg_counts[key].append(counts.get(key, 0))

        # avg game length per chunk
        avg_lens = []
        for chunk in all_chunks:
            lens = [l for _, l in chunk]
            avg_lens.append(sum(lens) / len(lens) if lens else 0)

        ax.cla()
        bottoms = [0] * len(xs)
        for key, color, label in SEGMENTS:
            vals = seg_counts[key]
            ax.bar(xs, vals, bottom=bottoms, color=color, label=label, alpha=0.85)
            for xi, v, b in zip(xs, vals, bottoms):
                if v > 0:
                    ax.text(xi, b + v / 2, str(v), ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        # text above each bar: ATK% / DEF% / avg_len
        for xi, total, avg_l in zip(xs, totals, avg_lens):
            if total == 0:
                continue
            atk_n = sum(seg_counts[k][xi] for k, _, _ in SEGMENTS if k.startswith("atk"))
            def_n = sum(seg_counts[k][xi] for k, _, _ in SEGMENTS if k.startswith("def"))
            atk_pct = atk_n / total * 100
            def_pct = def_n / total * 100
            info = f"A:{atk_pct:.0f}% D:{def_pct:.0f}%\nlen:{avg_l:.0f}"
            ax.text(xi, bottoms[xi] + 0.3, info, ha="center", va="bottom",
                    fontsize=6.5, color="black", linespacing=1.3)

        max_total = max(totals) if totals else 1
        ax.set_ylim(0, max_total * 1.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel(f"Game # (step={step})")
        ax.set_ylabel("Count")
        ax.set_title(f"Total games: {len(games)}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()

    ani = animation.FuncAnimation(fig, draw, interval=5000, cache_frame_data=False)
    draw(0)
    plt.show()


if __name__ == "__main__":
    main()
