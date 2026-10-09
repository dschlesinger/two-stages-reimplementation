#!/usr/bin/env python3
"""Hairpin transfer success by patched module: the paper's Fig. 12 next to our reproduction.

The paper's bars are the percentages printed on Fig. 12 (App. I.1). The authors'
released results/module_patching/ only holds a one-case test run, so the figure
is the only source for them. Our bars come from the full-patching CSV written by
patch_esmfold.py; a module we have not run gets no bar.

Usage:
    python scripts/plot_module_patching.py
    python scripts/plot_module_patching.py --results full_patching.csv --out outputs/patching
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# % of outputs with a hairpin, read off Fig. 12 of arXiv:2602.06020.
PAPER = {
    "Folding Trunk\n(Touch Mask)": 62.5,
    "Folding Trunk\n(Standard Intra Mask)": 38.3,
    "Structure\nModule": 4.6,
    "Input\nIntervention": 2.6,
    "ESM\nEncoder": 0.6,
}
UNPATCHED = "Unpatched\nTarget"
PAPER_COLOR, OURS_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e6e5e1", "#fcfcfb"


def our_rates(path):
    """{module label: (successes, cases)} for the modules covered by the full-patching CSV."""
    if not path.exists():
        sys.exit(f"{path} not found: python scripts/patch_esmfold.py full")
    df = pd.read_csv(path, dtype={"block": str}).dropna(subset=["hairpin_found"])
    df = df.astype({"hairpin_found": bool})
    patched = df[(df.patch_mode == "both") & (df.block == "all")]
    clean = df[df.patch_mode == "none"]
    return {
        "Folding Trunk\n(Standard Intra Mask)": (patched.hairpin_found.sum(), len(patched)),
        UNPATCHED: (clean.hairpin_found.sum(), len(clean)),
    }


def plot(ours, path):
    labels = list(PAPER) + [UNPATCHED]
    height = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for y, label in enumerate(labels):
        if label in PAPER:
            ax.barh(y - height / 2, PAPER[label], height, color=PAPER_COLOR, edgecolor=SURFACE, linewidth=1)
            ax.text(PAPER[label] + 1.2, y - height / 2, f"{PAPER[label]:.1f}%", va="center", color=INK, fontsize=9)
        if label in ours:
            hits, total = ours[label]
            rate = 100 * hits / total
            ax.barh(y + height / 2, rate, height, color=OURS_COLOR, edgecolor=SURFACE, linewidth=1)
            ax.text(rate + 1.2, y + height / 2, f"{rate:.1f}%  ({hits:,} of {total:,})",
                    va="center", color=INK, fontsize=9)
        else:
            ax.text(1.2, y + height / 2, "not run", va="center", color=MUTED, fontsize=8, style="italic")
    # The paper has no bar for the unpatched control.
    ax.text(1.2, len(labels) - 1 - height / 2, "not reported", va="center", color=MUTED, fontsize=8, style="italic")

    ax.set_yticks(range(len(labels)), labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of outputs with hairpin", color=INK)
    ax.set_title("Hairpin transfer success rate by module (ESMFold)", color=INK, fontsize=12, loc="left")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)

    handles = [plt.Rectangle((0, 0), 1, 1, color=PAPER_COLOR), plt.Rectangle((0, 0), 1, 1, color=OURS_COLOR)]
    ax.legend(handles, ["Lu et al. 2026 (Fig. 12)", "Our reproduction"], loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("outputs/patching/full_patching.csv"),
                   help="full-patching CSV from patch_esmfold.py")
    p.add_argument("--out", type=Path, default=Path("outputs/patching"), help="directory for the figure")
    return p.parse_args()


def main():
    args = parse_args()
    ours = our_rates(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "module_patching_success.png"
    plot(ours, path)
    for label, (hits, total) in ours.items():
        print(f"{label.replace(chr(10), ' ')}: {hits} of {total} ({hits / total:.1%})")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
