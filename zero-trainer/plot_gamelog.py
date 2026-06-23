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


def print_forts(all_chunks, step):
    g, y, r = ANSI["green"], ANSI["yellow"], ANSI["reset"]
    rows = []
    mx = 0.0
    for i, games in enumerate(all_chunks):
        n = len(games)
        c = sum(1 for t, _ in games if t == "def_fort")
        pct = c / n * 100 if n else 0.0
        rows.append((i, i * step, c, pct))
        mx = max(mx, pct)
    scale = 50 / mx if mx else 1.0
    print(f"\n{ANSI['bold']}def_fort %% per chunk (step={step}, max={mx:.2f}%){r}\n")
    for i, start, c, pct in rows:
        col = g if pct >= mx * 0.66 else (y if pct >= mx * 0.33 else ANSI["gray"])
        bar = "█" * int(round(pct * scale))
        print(f"  #{i:3d} [{start:>7}] {c:4d}  {pct:5.2f}%  {col}{bar}{r}")


def _roll(v, w=5):
    return [sum(v[max(0, i - w + 1):i + 1]) / (i - max(0, i - w + 1) + 1) for i in range(len(v))]


_SIDE = lambda side: [k for k, v in TERMINAL_COLORS.items() if v[0] == side]

# key -> (label, color, is_percent, fn(counts, n, avg_len) -> value)
METRICS = {
    "fort":      ("def_fort %  (defense builds)", "#2980b9", True,
                  lambda c, n, l: c["def_fort"] / n * 100 if n else 0),
    "threefold": ("atk_threefold %  (cheap repetition)", "#c0392b", True,
                  lambda c, n, l: c["atk_threefold"] / n * 100 if n else 0),
    "surround":  ("atk_surrounded %  (full encirclement)", "#e67e22", True,
                  lambda c, n, l: c["atk_surrounded"] / n * 100 if n else 0),
    "capture":   ("atk_capture %  (king taken)", "#e74c3c", True,
                  lambda c, n, l: c["atk_capture"] / n * 100 if n else 0),
    "corner":    ("def_corner %  (king escapes)", "#16a085", True,
                  lambda c, n, l: c["def_corner"] / n * 100 if n else 0),
    "atk":       ("ATK win %", "#c0392b", True,
                  lambda c, n, l: sum(c[k] for k in _SIDE("ATK")) / n * 100 if n else 0),
    "def":       ("DEF win %", "#27ae60", True,
                  lambda c, n, l: sum(c[k] for k in _SIDE("DEF")) / n * 100 if n else 0),
    "draw":      ("draw %", "#7f8c8d", True,
                  lambda c, n, l: sum(c[k] for k in _SIDE("DRAW")) / n * 100 if n else 0),
    "len":       ("avg game length", "#8e44ad", False,
                  lambda c, n, l: l),
}


def _chunk_stats(games):
    counts = defaultdict(int)
    tl = 0
    for t, ln in games:
        counts[t] += 1
        tl += ln
    n = len(games)
    return counts, n, (tl / n if n else 0)


def plot_metrics_mpl(all_chunks, step, keys, out="forts.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        return
    keys = [k for k in keys if k in METRICS]
    if not keys:
        print("no valid metrics. available:", ", ".join(METRICS))
        return
    stats = [_chunk_stats(g) for g in all_chunks]
    xs = [i * step for i in range(len(all_chunks))]

    fig, axes = plt.subplots(len(keys), 1, sharex=True,
                             figsize=(13, 2.4 * len(keys) + 1), squeeze=False)
    axes = axes[:, 0]
    fig.suptitle("Self-play metrics over training", fontsize=13, fontweight="bold")
    for ax, key in zip(axes, keys):
        label, color, is_pct, fn = METRICS[key]
        vals = [fn(c, n, l) for (c, n, l) in stats]
        ax.plot(xs, vals, color=color, lw=1, marker="o", ms=2.5, alpha=0.35)
        ax.plot(xs, _roll(vals), color=color, lw=2.6, label="5-chunk mean")
        suffix = "%" if is_pct else ""
        ax.annotate(f"{vals[-1]:.2f}{suffix}", (xs[-1], vals[-1]),
                    textcoords="offset points", xytext=(-6, 9), ha="right",
                    fontsize=10, fontweight="bold", color=color)
        ax.set_ylabel(label, fontsize=9)
        if is_pct:
            ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel(f"games (step={step})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}  ({', '.join(keys)})")
    plt.show()


def plot_metrics_interactive(all_chunks, step, initial):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons
    except ImportError:
        print("pip install matplotlib")
        return
    keys = list(METRICS)
    stats = [_chunk_stats(g) for g in all_chunks]
    xs = [i * step for i in range(len(all_chunks))]
    series = {k: [METRICS[k][3](c, n, l) for (c, n, l) in stats] for k in keys}

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Self-play metrics over training", fontsize=13, fontweight="bold")
    ax_check = fig.add_axes([0.005, 0.30, 0.12, 0.45])
    ax_check.set_title("metrics", fontsize=9)
    check = CheckButtons(ax_check, keys, [k in initial for k in keys])

    data_axes = []

    def positions(n):
        x0, x1, y0, y1, gap = 0.22, 0.97, 0.07, 0.92, 0.03
        h = (y1 - y0 - gap * (n - 1)) / n
        return [[x0, y1 - h - i * (h + gap), x1 - x0, h] for i in range(n)]

    def redraw(_=None):
        for ax in data_axes:
            ax.remove()
        data_axes.clear()
        checked = [k for k, st in zip(keys, check.get_status()) if st]
        if checked:
            pos = positions(len(checked))
            for i, (p, key) in enumerate(zip(pos, checked)):
                label, color, is_pct, _ = METRICS[key]
                vals = series[key]
                ax = fig.add_axes(p)
                ax.plot(xs, vals, color=color, lw=1, marker="o", ms=2, alpha=0.3)
                ax.plot(xs, _roll(vals), color=color, lw=2.4)
                ax.annotate(f"{vals[-1]:.2f}{'%' if is_pct else ''}", (xs[-1], vals[-1]),
                            textcoords="offset points", xytext=(-6, 8), ha="right",
                            fontsize=9, fontweight="bold", color=color)
                ax.set_ylabel(label, fontsize=8)
                if is_pct:
                    ax.set_ylim(bottom=0)
                ax.grid(True, alpha=0.25)
                if i < len(checked) - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(f"games (step={step})")
                data_axes.append(ax)
        fig.canvas.draw_idle()

    check.on_clicked(redraw)
    redraw()
    fig.savefig("forts.png", dpi=110)   # snapshot of initial state
    plt.show()


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
    ap.add_argument("--print-forts", action="store_true", help="Terminal ASCII graph of def_fort %% per chunk only")
    ap.add_argument("--plot-forts", action="store_true", help="matplotlib graph, one panel per metric (saves forts.png + shows window)")
    ap.add_argument("--metrics", default="fort,threefold",
                    help="comma list; available: fort,threefold,surround,capture,corner,atk,def,draw,len")
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="live window with checkboxes to toggle metrics (needs a GUI backend)")
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

    if args.print_forts:
        print_forts(all_chunks, args.step)
        return

    initial = [m.strip() for m in args.metrics.split(",")]
    if args.interactive:
        plot_metrics_interactive(all_chunks, args.step, initial)
        return
    if args.plot_forts:
        plot_metrics_mpl(all_chunks, args.step, initial)
        return

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
