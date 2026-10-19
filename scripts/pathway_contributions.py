#!/usr/bin/env python3
"""Seq2pair and pair2seq update norms across the ESMFold trunk (section 4, Fig. 4b).

Measures the relative contribution of each pathway at every block, without any
intervention: the norm of the update that pathway writes at that block,
normalized within the pathway across the trunk. The paper's result is that
seq2pair flow into z peaks in the first ten blocks and declines through the
middle, while pair2seq flow into s is small early and rises sharply late.

Per block, one fold without recycling records
    seq2pair    the update added to z: the output of block.sequence_to_pair
    pair2seq    the update added to s: the output of block.seq_attention, the
                only place z reaches s (it carries the pair bias)
    pair_bias   the output of block.pair_to_sequence, the bias alone; a weaker
                proxy for pair2seq, recorded but not plotted
Each is the Frobenius norm over all positions. The blocks are pre-norm: each
sublayer LayerNorms its input and adds its raw output to the residual stream, so
these are raw updates and their size depends on how large the stream is at that
block. --measure relative divides each update by the norm of the state entering
the block (s for pair2seq, z for seq2pair), which is the size the next LayerNorm
sees it against. Within a protein, a pathway's 48 values are divided by their
maximum; the plot averages that over proteins.

Usage:
    python scripts/pathway_contributions.py
    python scripts/pathway_contributions.py --set both --limit 20
    python scripts/pathway_contributions.py --plot-only          # redraw from the CSV
    python scripts/pathway_contributions.py --plot-only --measure relative

--set picks the sequences: the patching dataset's helical targets, its hairpin
donors, or both. --sequence folds one given sequence instead. Writes
pathways.csv (raw and relative norms, each also scaled by its maximum, per protein per block) and
pathway_contributions.png to --out. Needs a GPU for the folds.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

from patch_esmfold import HUB_ID, fold

PATHWAYS = ["seq2pair", "pair2seq", "pair_bias"]
PLOTTED = ["seq2pair", "pair2seq"]
LABELS = {
    "seq2pair": "seq2pair: update to z",
    "pair2seq": "pair2seq: update to s",
    "pair_bias": "pair2seq: attention bias only",
}
COLORS = {"seq2pair": "#2a78d6", "pair2seq": "#eb6834", "pair_bias": "#898781"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e6e5e1", "#fcfcfb"


# The state each pathway's update is written into: seq2pair adds to z, the pair2seq paths add to s.
STREAM = {"seq2pair": "z", "pair2seq": "s", "pair_bias": "s"}


def recording(blocks):
    """Hook every block; returns the handles, {pathway: [norm per call]} and {"s"/"z": [input norm per call]}."""
    norms = {name: [] for name in PATHWAYS}
    streams = {"s": [], "z": []}

    def stream_hook(_module, inputs):
        streams["s"].append(inputs[0].float().norm().item())
        streams["z"].append(inputs[1].float().norm().item())

    def recorder(name, pick=lambda output: output):
        def hook(_module, _inputs, output):
            norms[name].append(pick(output).float().norm().item())
        return hook

    handles = []
    for block in blocks:
        handles.append(block.register_forward_pre_hook(stream_hook))
        handles.append(block.sequence_to_pair.register_forward_hook(recorder("seq2pair")))
        # EsmFoldSelfAttention returns (output, attention weights); the update to s is the output.
        handles.append(block.seq_attention.register_forward_hook(recorder("pair2seq", lambda output: output[0])))
        handles.append(block.pair_to_sequence.register_forward_hook(recorder("pair_bias")))
    return handles, norms, streams


def sequences(args):
    """(name, sequence) pairs to fold."""
    if args.sequence:
        return [("given", args.sequence)]
    if not args.data.exists():
        sys.exit(f"{args.data} not found: python scripts/download_data.py, or pass --sequence")
    cases = pd.read_csv(args.data)
    found = []
    if args.set in ("targets", "both"):
        found.append(cases.drop_duplicates("target_name").rename(
            columns={"target_name": "name", "target_sequence": "sequence"}))
    if args.set in ("donors", "both"):
        found.append(cases.drop_duplicates("donor_pdb").rename(
            columns={"donor_pdb": "name", "donor_sequence": "sequence"}))
    # --limit applies to each set, so "both" folds up to 2 x limit proteins.
    return [pair for table in found for pair in table.head(args.limit)[["name", "sequence"]].itertuples(index=False, name=None)]


def measure(args):
    proteins = sequences(args)
    if not Path(args.model).exists():
        args.model = HUB_ID
    print(f"Loading ESMFold from {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmForProteinFolding.from_pretrained(args.model).to(args.device).eval()
    blocks = model.trunk.blocks

    rows = []
    for name, sequence in tqdm(proteins, desc="proteins"):
        handles, norms, streams = recording(blocks)
        try:
            fold(model, tokenizer, sequence)
        finally:
            for handle in handles:
                handle.remove()
        for pathway, values in norms.items():
            if len(values) != len(blocks):
                sys.exit(f"{pathway} ran {len(values)} times for {len(blocks)} blocks; recycling must be off")
        for pathway, values in norms.items():
            relative = [value / state for value, state in zip(values, streams[STREAM[pathway]])]
            rows.extend({
                "protein": name, "length": len(sequence), "pathway": pathway, "block": k,
                "norm": round(value, 4), "normalized": round(value / max(values), 4),
                "relative": round(rel, 5), "relative_normalized": round(rel / max(relative), 4),
            } for k, (value, rel) in enumerate(zip(values, relative)))

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "pathways.csv", index=False)
    return table


def plot(table, path, measure):
    column = "relative_normalized" if measure == "relative" else "normalized"
    mean = table.pivot_table(index="block", columns="pathway", values=column, aggfunc="mean")
    spread = table.pivot_table(index="block", columns="pathway", values=column, aggfunc="std").fillna(0)
    proteins = table.protein.nunique()

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for pathway in PLOTTED:
        if pathway not in mean:
            continue
        ax.plot(mean.index, mean[pathway], "-", color=COLORS[pathway], linewidth=2, label=LABELS[pathway])
        ax.fill_between(mean.index, mean[pathway] - spread[pathway], mean[pathway] + spread[pathway],
                        color=COLORS[pathway], alpha=0.12, linewidth=0)

    ax.set_xlim(0, mean.index.max())
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("trunk block", color=INK)
    ax.set_ylabel(("update norm / state norm" if measure == "relative" else "update norm") + ", over its maximum", color=INK)
    ax.set_title(f"Pathway contributions across the ESMFold trunk ({proteins} proteins, mean ± sd)",
                 color=INK, fontsize=12, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.legend(loc="upper center", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=["targets", "donors", "both"], default="targets",
                   help="which dataset sequences to fold")
    p.add_argument("--sequence", help="fold this sequence instead of the dataset")
    p.add_argument("--data", type=Path, default=Path("reference/data/patching_dataset.csv"))
    p.add_argument("--limit", type=int, default=20, help="number of proteins per set")
    p.add_argument("--out", type=Path, default=Path("outputs/pathways"), help="results directory")
    p.add_argument("--measure", choices=["raw", "relative"], default="raw",
                   help="raw update norms, or each divided by the norm of the state entering the block")
    p.add_argument("--plot-only", action="store_true", help="redraw the figure from an existing pathways.csv")
    p.add_argument("--model", default="weights/esmfold", help=f"local ESMFold directory; falls back to {HUB_ID}")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        csv = args.out / "pathways.csv"
        if not csv.exists():
            sys.exit(f"{csv} not found: run without --plot-only first")
        table = pd.read_csv(csv)
    else:
        table = measure(args)

    column = "relative_normalized" if args.measure == "relative" else "normalized"
    if column not in table:
        sys.exit(f"{column} is not in the CSV: rerun without --plot-only to record it")
    path = args.out / ("pathway_contributions_relative.png" if args.measure == "relative" else "pathway_contributions.png")
    plot(table, path, args.measure)
    mean = table.pivot_table(index="block", columns="pathway", values=column, aggfunc="mean")
    for pathway in PATHWAYS:
        print(f"{pathway}: peak at block {mean[pathway].idxmax()}, "
              f"mean of blocks 0-9 {mean[pathway][:10].mean():.2f}, last 10 {mean[pathway][-10:].mean():.2f}")
    print(f"Saved {args.out / 'pathways.csv'} and {path}")


if __name__ == "__main__":
    main()
