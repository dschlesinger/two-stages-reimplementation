#!/usr/bin/env python3
"""Steer the charge direction on a neutral sequence and see what shape comes out.

A demo built on section 4.2: fold a chain with no charged residues (poly-alanine
by default), then add +/- alpha * v_charge to the sequence state of chosen
segments during an early window of trunk blocks, and compare the geometry with
the unsteered fold. Opposite charges on two segments should pull them together,
the same charge should push them apart.

This goes past what the paper validates. Its steering acts on helix-turn-helix
regions of real helical proteins, where the surrounding sequence already favours
the motif; a uniform sequence has none of that context, so treat low pLDDT as
the expected answer and the geometry numbers, not the DSSP string, as the signal.

Presets, with the default 40-residue sequence:

    clamp    + on residues 4-13, - on 26-35: the ends attract, so the chain
             should fold back on itself (a horseshoe or U)
    splay    + on both of those segments: they repel, so the chain should
             straighten out
    zigzag   + on 2-9, - on 15-23, + on 30-37: alternating, for a meander
    all      every preset above, against one shared baseline
    none     no steering, baseline only

Segments can also be given directly, 0-based with the end excluded, as in the
patch regions elsewhere in the repo:

    python scripts/charge_shapes.py --segments 4-14:+ 26-36:-

Usage:
    python scripts/charge_shapes.py --preset clamp
    python scripts/charge_shapes.py --preset all --strength 1 2 3 5 --axis signed
    python scripts/charge_shapes.py --preset splay --sequence $(python -c "print('A'*60)")
    python scripts/charge_shapes.py --preset clamp --blocks 0-14 --axis signed

--axis picks what gets added: signed uses +/- the one charge axis, as the paper
does; poles uses the separate positive and negative directions, each measured
from the neutral mean, which suits a sequence that starts out neutral.

The unsteered baseline is always folded, in the same batch as the steered
copies, so nothing but the intervention differs between them.

Writes one PDB per fold, plus shapes.csv with the geometry, to --out. Needs a
GPU and the directions from scripts/charge_directions.py; mkdssp is optional.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

from patch_esmfold import HUB_ID, find_hairpins, fold, run_dssp

PRESETS = {
    "none": [],
    "clamp": [(4, 14, +1), (26, 36, -1)],
    "splay": [(4, 14, +1), (26, 36, +1)],
    "zigzag": [(2, 10, +1), (15, 24, -1), (30, 38, +1)],
}


def parse_segments(specs):
    """'4-14:+' -> (4, 14, +1), with the end excluded."""
    segments = []
    for spec in specs:
        m = re.fullmatch(r"(\d+)-(\d+):([+-])", spec)
        if not m:
            sys.exit(f"bad segment {spec!r}: expected START-END:+ or START-END:-")
        start, end, sign = int(m[1]), int(m[2]), 1 if m[3] == "+" else -1
        if start >= end:
            sys.exit(f"bad segment {spec!r}: start must be below end")
        segments.append((start, end, sign))
    return segments


def parse_blocks(spec, num_blocks):
    """'0-14' -> range(0, 15), the inclusive window the paper's sliding ablation uses."""
    m = re.fullmatch(r"(\d+)-(\d+)", spec)
    if not m or int(m[1]) > int(m[2]) or int(m[2]) >= num_blocks:
        sys.exit(f"bad block window {spec!r}: expected FIRST-LAST within 0-{num_blocks - 1}")
    return range(int(m[1]), int(m[2]) + 1)


def load_directions(path, axis, num_blocks):
    """Per-depth deltas of unit length: what a +1 segment and a -1 segment get, in units of sigma."""
    if not path.exists():
        sys.exit(f"{path} not found: python scripts/charge_directions.py")
    saved = torch.load(path, weights_only=False)
    depths = saved["charge"].shape[0]
    if depths != num_blocks + 1:
        sys.exit(f"{path} has {depths} depths but the trunk has {num_blocks} blocks")
    if axis == "signed":
        # The paper's construction: one axis, added on one segment and subtracted on the other.
        return {+1: saved["charge"], -1: -saved["charge"]}, saved["charge_std"]
    # Each pole measured from the neutral mean, which is where a poly-alanine chain starts.
    return {+1: saved["positive"], -1: saved["negative"]}, (saved["positive_std"] + saved["negative_std"]) / 2


def steering(model, batch_delta):
    """Add batch_delta[k] to the whole batch's sequence state before block k runs."""
    def make_hook(k):
        def hook(_module, inputs):
            return (inputs[0] + batch_delta[k].to(inputs[0]), *inputs[1:])
        return hook

    return [block.register_forward_pre_hook(make_hook(k))
            for k, block in enumerate(model.trunk.blocks) if batch_delta[k].any()]


def segment_deltas(segments, directions, sigma, strength, length, dim, device):
    """[depths, L, d_s] of what to add at every position, zero outside the steered segments."""
    depths = sigma.shape[0]
    delta = torch.zeros(depths, length, dim, device=device)
    for start, end, sign in segments:
        if end > length:
            sys.exit(f"segment {start}-{end} runs past the {length}-residue sequence")
        delta[:, start:end] = (strength * sigma[:, None] * directions[sign]).to(device)[:, None, :]
    return delta


def geometry(ca, segments):
    """Radius of gyration, end-to-end distance, and the mean distance between each steered pair."""
    out = {
        "radius_of_gyration": (ca - ca.mean(0)).norm(dim=-1).square().mean().sqrt().item(),
        "end_to_end": (ca[-1] - ca[0]).norm().item(),
    }
    for i, (a0, a1, _) in enumerate(segments):
        for b0, b1, _ in segments[i + 1:]:
            key = f"dist_{a0}_{a1}__{b0}_{b1}"
            out[key] = torch.cdist(ca[a0:a1], ca[b0:b1]).mean().item()
    return out


def describe(segments):
    return "_".join(f"{start}-{end}{'+' if sign > 0 else '-'}" for start, end, sign in segments)


def conditions(args):
    """The (name, segments) pairs to steer; empty means the baseline fold is the whole run."""
    if args.segments:
        return [("custom", parse_segments(args.segments))]
    if args.preset == "all":
        return [(name, PRESETS[name]) for name in PRESETS if PRESETS[name]]
    return [(args.preset, PRESETS[args.preset])] if PRESETS[args.preset] else []


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS) + ["all"], default="clamp",
                   help="all runs every steered preset against one shared baseline")
    p.add_argument("--segments", nargs="+", help="steered segments as START-END:+ or START-END:-, overriding --preset")
    p.add_argument("--sequence", default="A" * 40, help="the neutral sequence to fold")
    p.add_argument("--strength", type=float, nargs="+", default=[3.0], help="steering strength in sigmas")
    p.add_argument("--blocks", default="0-14", help="inclusive window of trunk blocks to steer")
    p.add_argument("--axis", choices=["poles", "signed"], default="poles", help="which directions to add")
    p.add_argument("--directions", type=Path, default=Path("outputs/charge/charge_directions.pt"))
    p.add_argument("--out", type=Path, default=Path("outputs/charge/shapes"))
    p.add_argument("--model", default="weights/esmfold", help=f"local ESMFold directory; falls back to {HUB_ID}")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    steered = conditions(args)
    charged = set(args.sequence) & set("KRHDE")
    if charged:
        print(f"warning: the sequence already carries charge at {sorted(charged)}")
    if not Path(args.model).exists():
        args.model = HUB_ID
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading ESMFold from {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmForProteinFolding.from_pretrained(args.model).to(args.device).eval()
    num_blocks = len(model.trunk.blocks)
    window = parse_blocks(args.blocks, num_blocks)
    directions, sigma = load_directions(args.directions, args.axis, num_blocks)
    dim = directions[+1].shape[-1]

    length = len(args.sequence)
    # One unsteered baseline, then every condition at every strength, all in one batch.
    jobs = [(name, segments, strength) for name, segments in steered for strength in args.strength]
    print(f"{len(args.sequence)} residues, baseline + {len(jobs)} steered fold{'' if len(jobs) == 1 else 's'}, "
          f"blocks {args.blocks}, axis {args.axis}")

    batch_delta = torch.zeros(num_blocks, len(jobs) + 1, length, dim, device=args.device)
    for b, (_, segments, strength) in enumerate(jobs, start=1):
        delta = segment_deltas(segments, directions, sigma, strength, length, dim, args.device)
        for k in window:
            batch_delta[k, b] = delta[k]

    handles = steering(model, batch_delta)
    try:
        outputs = fold(model, tokenizer, args.sequence, len(jobs) + 1)
    finally:
        for h in handles:
            h.remove()

    have_dssp = shutil.which("mkdssp") is not None
    if not have_dssp:
        print("mkdssp not on PATH; skipping the secondary structure check")
    pdbs = model.output_to_pdb(outputs)
    plddt = outputs.plddt[:, :, 1].float().cpu()
    # The baseline is scored against every condition's segment pairs, so each steered row has a reference.
    all_segments = [segments for _, segments in steered]
    labels = ["baseline"] + [f"{name}_a{strength:g}" for name, _, strength in jobs]
    rows = []
    for b, label in enumerate(tqdm(labels, desc="scoring")):
        (args.out / f"{label}.pdb").write_text(pdbs[b])
        ss = run_dssp(pdbs[b], length) if have_dssp else None
        ca = outputs.positions[-1, b, :, 1].float().cpu()
        distances = geometry(ca, [])  # radius of gyration and end to end, whether or not anything is steered
        for segments in (all_segments if b == 0 else [jobs[b - 1][1]]):
            distances.update(geometry(ca, segments))
        rows.append({
            "label": label,
            "preset": "" if b == 0 else jobs[b - 1][0],
            "segments": "" if b == 0 else describe(jobs[b - 1][1]),
            "strength": 0.0 if b == 0 else jobs[b - 1][2],
            "blocks": "" if b == 0 else args.blocks,
            "axis": "" if b == 0 else args.axis,
            "plddt": round(plddt[b].mean().item(), 4),
            **{k: round(v, 2) for k, v in distances.items()},
            "hairpin": "" if ss is None else bool(find_hairpins(ss)),
            "dssp": ss or "",
        })

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "shapes.csv", index=False)
    print(table.drop(columns="dssp").to_string(index=False))
    print(f"Saved {len(labels)} structures and {args.out / 'shapes.csv'}")


if __name__ == "__main__":
    main()
