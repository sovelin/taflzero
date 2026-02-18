"""
Scan selfplay.bin and report attacker/defender win ratios per chunk of samples.

Value encoding:
  value = +1 => side-to-move won
  value = -1 => side-to-move lost
  value =  0 => draw
stm byte:
  0 => attackers to move
  1 => defenders to move
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

BIT_POSITION_BYTES = 49  # 48 planes + 1 stm
LEGAL_MASK_BYTES = 605   # ceil(4840 / 8)


def classify_outcome(stm: int, value: int) -> str:
    if value == 0:
        return "draw"
    # value is from side-to-move perspective
    if stm == 0:  # attackers to move
        return "atk" if value > 0 else "def"
    # defenders to move
    return "def" if value > 0 else "atk"


def print_chunk(chunk_idx: int, start_idx: int, end_idx: int, atk: int, defn: int, draw: int) -> tuple[float, float, float] | None:
    total = atk + defn + draw
    if total == 0:
        return None
    atk_pct = 100.0 * atk / total
    def_pct = 100.0 * defn / total
    draw_pct = 100.0 * draw / total
    print(
        f"Chunk {chunk_idx} [{start_idx}-{end_idx}]  "
        f"atk={atk} def={defn} draw={draw}  "
        f"atk%={atk_pct:.1f}% def%={def_pct:.1f}% draw%={draw_pct:.1f}%"
    )
    return atk_pct, def_pct, draw_pct


def plot_series(chunks: list[int], atk: list[float], defn: list[float], draw: list[float], out_path: Path | None, show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"matplotlib is required for --plot/--plot-out ({exc})")

    plt.figure(figsize=(10, 5))
    plt.plot(chunks, atk, label="attack %")
    plt.plot(chunks, defn, label="defense %")
    plt.plot(chunks, draw, label="draw %")
    plt.xlabel("Chunk index")
    plt.ylabel("Percent")
    plt.title("Self-play outcome ratios by chunk")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {out_path}")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute attacker/defender win ratios from selfplay.bin")
    parser.add_argument("--data", type=Path, default=Path("zero-trainer/selfplay.bin"), help="Path to selfplay.bin")
    parser.add_argument("--chunk", type=int, default=100000, help="Samples per chunk (default: 100000)")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plot of atk/def/draw percentages")
    parser.add_argument("--plot-out", type=Path, default=None, help="Save plot to file (png/svg)")
    args = parser.parse_args()

    if args.chunk <= 0:
        raise SystemExit("--chunk must be > 0")
    if not args.data.exists():
        raise SystemExit(f"File not found: {args.data}")

    atk = defn = draw = 0
    total = 0
    chunk_idx = 1
    chunk_start = 1

    chunks: list[int] = []
    atk_pct_series: list[float] = []
    def_pct_series: list[float] = []
    draw_pct_series: list[float] = []

    with open(args.data, "rb") as f:
        while True:
            planes = f.read(48)
            if len(planes) < 48:
                break
            stm_b = f.read(1)
            if len(stm_b) < 1:
                break
            stm = stm_b[0]

            legal = f.read(LEGAL_MASK_BYTES)
            if len(legal) < LEGAL_MASK_BYTES:
                break

            policy_len_bytes = f.read(2)
            if len(policy_len_bytes) < 2:
                break
            policy_len = struct.unpack("<H", policy_len_bytes)[0]

            # Skip policy entries
            skip_bytes = policy_len * 4
            if skip_bytes:
                f.seek(skip_bytes, 1)

            value_b = f.read(1)
            if len(value_b) < 1:
                break
            value = struct.unpack("<b", value_b)[0]

            outcome = classify_outcome(stm, value)
            if outcome == "atk":
                atk += 1
            elif outcome == "def":
                defn += 1
            else:
                draw += 1

            total += 1
            if total % args.chunk == 0:
                stats = print_chunk(chunk_idx, chunk_start, total, atk, defn, draw)
                if stats is not None:
                    a, d, dr = stats
                    chunks.append(chunk_idx)
                    atk_pct_series.append(a)
                    def_pct_series.append(d)
                    draw_pct_series.append(dr)
                chunk_idx += 1
                chunk_start = total + 1
                atk = defn = draw = 0

    # остаток
    if atk + defn + draw > 0:
        stats = print_chunk(chunk_idx, chunk_start, total, atk, defn, draw)
        if stats is not None:
            a, d, dr = stats
            chunks.append(chunk_idx)
            atk_pct_series.append(a)
            def_pct_series.append(d)
            draw_pct_series.append(dr)

    print(f"Total samples: {total}")

    if args.plot or args.plot_out is not None:
        plot_series(chunks, atk_pct_series, def_pct_series, draw_pct_series, args.plot_out, args.plot)


if __name__ == "__main__":
    main()
