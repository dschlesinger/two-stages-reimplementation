#!/usr/bin/env python3
"""Output distances against the scale of z and of s: our version of the paper's Fig. 6c.

Each thin line is one protein's change in mean pairwise CA distance from its own
scale 1.0 fold; the thick line is the mean over proteins. Scaling z should move
the distances monotonically and scaling s should leave them where they are.

Usage:
    python scripts/plot_scaling.py
    python scripts/plot_scaling.py --results outputs/scaling/scaling.csv --out outputs/scaling
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {"z": "Scaling z (pairwise)", "s": "Scaling s (sequence)"}
COLORS = {"z": "#2a78d6", "s": "#eb6834"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e6e5e1", "#fcfcfb"


def plot(table, path):
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    for target, color in COLORS.items():
        rows = table[table.scaled == target]
        for _, protein in rows.groupby("protein"):
            ax.plot(protein.scale, protein.distance_deviation, color=color, alpha=0.3, linewidth=1)
        mean = rows.groupby("scale").distance_deviation.mean()
        ax.plot(mean.index, mean, "o-", color=color, linewidth=2.5, markersize=5, label=LABELS[target])

    ax.set_xlabel("scale factor", color=INK)
    ax.set_ylabel("change in mean pairwise CA distance (Å)", color=INK)
    ax.set_title(f"Output distances against the scale of z and s (ESMFold, {table.protein.nunique()} proteins)",
                 color=INK, fontsize=12, loc="left")
    ax.set_xticks(sorted(table.scale.unique()))
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("outputs/scaling/scaling.csv"),
                   help="scaling CSV from scale_pairwise.py")
    p.add_argument("--out", type=Path, default=Path("outputs/scaling"), help="directory for the figure")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.results.exists():
        sys.exit(f"{args.results} not found: python scripts/scale_pairwise.py 0 0.5 1 1.5 2 --target both")
    table = pd.read_csv(args.results)
    if set(table.scaled) != set(COLORS):
        sys.exit("the results need both targets: run scale_pairwise.py with --target both")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "scaling_distance.png"
    plot(table, path)
    print(table.pivot_table(index="scale", columns="scaled", values="distance_deviation", aggfunc="mean").round(2).to_string())
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
