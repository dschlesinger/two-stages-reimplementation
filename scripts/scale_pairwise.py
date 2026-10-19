#!/usr/bin/env python3
"""Scale the pairwise representation before the structure module and fold (section 5.1, Fig. 6c-d).

Isolates how the structure module reads z. The trunk runs once, untouched; the
pairwise input to the structure module is then multiplied by a scale factor, with
s left alone. The scale is applied after the structure module's input LayerNorm
(layer_norm_z), not to trunk2sm_z: that LayerNorm divides out any uniform scale,
so scaling its input leaves the fold unchanged for every nonzero factor. The paper's result is that output distances
follow the scale monotonically: scaling up expands the structure, scaling down
contracts it, while the same treatment of s does nothing.

Every scale is a separate element of one batch, so the trunk computation behind
them is identical and the scale is the only difference. Scale 1.0 is the
baseline and is always included.

Usage:
    python scripts/scale_pairwise.py 0 1 2
    python scripts/scale_pairwise.py 0 0.5 1 1.5 2 --sequence $(python -c "print('A'*40)")
    python scripts/scale_pairwise.py 0 0.5 1 1.5 2 --target both --limit 5

With no --sequence, the target proteins of the patching dataset are used, the
first --limit of them. --target s scales the sequence projection instead, and
both runs each in turn, which is the paper's control.

Writes one PDB per protein per scale, plus scaling.csv with the mean pairwise
CA distance and its deviation from the scale 1.0 baseline, to --out. Needs a
GPU; mkdssp is optional.

The gradient of distance against scale at the operating point (Fig. 6d) is not
computed here; this script only folds at the scales it is given.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

from patch_esmfold import HUB_ID, find_hairpins, fold, run_dssp

BASELINE = 1.0


def scaling(module, factors):
    """Multiply element b of the module's output by factors[b], and count the calls."""
    calls = []

    def hook(_module, _inputs, output):
        calls.append(1)
        shape = (-1, *([1] * (output.dim() - 1)))
        return output * factors.to(output).view(shape)

    return module.register_forward_hook(hook), calls


def mean_ca_distance(ca):
    """Mean distance over all residue pairs, the metric of Fig. 6c."""
    return torch.pdist(ca).mean().item()


def radius_of_gyration(ca):
    return (ca - ca.mean(0)).norm(dim=-1).square().mean().sqrt().item()


def sequences(args):
    """(name, sequence) pairs to fold."""
    if args.sequence:
        return [("given", args.sequence)]
    if not args.data.exists():
        sys.exit(f"{args.data} not found: python scripts/download_data.py, or pass --sequence")
    targets = pd.read_csv(args.data).drop_duplicates("target_name")
    return list(targets.head(args.limit)[["target_name", "target_sequence"]].itertuples(index=False, name=None))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scales", type=float, nargs="+", help="scale factors; 1.0 is added if missing")
    p.add_argument("--target", choices=["z", "s", "both"], default="z", help="which projection to scale")
    p.add_argument("--sequence", help="fold this sequence instead of the dataset targets")
    p.add_argument("--data", type=Path, default=Path("reference/data/patching_dataset.csv"))
    p.add_argument("--limit", type=int, default=3, help="number of dataset targets to fold")
    p.add_argument("--out", type=Path, default=Path("outputs/scaling"), help="results directory")
    p.add_argument("--model", default="weights/esmfold", help=f"local ESMFold directory; falls back to {HUB_ID}")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    scales = sorted(set(args.scales) | {BASELINE})
    if any(s < 0 for s in scales):
        sys.exit("scales must be zero or positive")
    targets = ["z", "s"] if args.target == "both" else [args.target]
    proteins = sequences(args)
    if not Path(args.model).exists():
        args.model = HUB_ID
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading ESMFold from {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmForProteinFolding.from_pretrained(args.model).to(args.device).eval()
    # The structure module's own input LayerNorms; scaling before them would be undone by the normalisation.
    projections = {"z": model.trunk.structure_module.layer_norm_z, "s": model.trunk.structure_module.layer_norm_s}

    have_dssp = shutil.which("mkdssp") is not None
    if not have_dssp:
        print("mkdssp not on PATH; skipping the secondary structure check")
    print(f"{len(proteins)} protein(s) x {len(targets)} target(s) x {len(scales)} scales: {scales}")

    factors = torch.tensor(scales, device=args.device)
    rows = []
    for name, sequence in tqdm(proteins, desc="proteins"):
        for target in targets:
            handle, calls = scaling(projections[target], factors)
            try:
                outputs = fold(model, tokenizer, sequence, len(scales))
            finally:
                handle.remove()
            if len(calls) != 1:
                sys.exit(f"the {target} layer norm ran {len(calls)} times, not once; recycling must be off")

            pdbs = model.output_to_pdb(outputs)
            plddt = outputs.plddt[:, :, 1].float().cpu()
            for b, scale in enumerate(scales):
                (args.out / f"{name}_{target}{scale:g}.pdb").write_text(pdbs[b])
                ss = run_dssp(pdbs[b], len(sequence)) if have_dssp else None
                ca = outputs.positions[-1, b, :, 1].float().cpu()
                rows.append({
                    "protein": name,
                    "length": len(sequence),
                    "scaled": target,
                    "scale": scale,
                    "mean_ca_distance": round(mean_ca_distance(ca), 3),
                    "radius_of_gyration": round(radius_of_gyration(ca), 3),
                    "plddt": round(plddt[b].mean().item(), 4),
                    "hairpin": "" if ss is None else bool(find_hairpins(ss)),
                    "dssp": ss or "",
                })

    table = pd.DataFrame(rows)
    # Deviation from the unscaled fold of the same protein, which is the quantity Fig. 6c plots.
    baseline = table[table.scale == BASELINE].set_index(["protein", "scaled"]).mean_ca_distance
    table["distance_deviation"] = (
        table.mean_ca_distance - table.set_index(["protein", "scaled"]).index.map(baseline)
    ).round(3)
    table.to_csv(args.out / "scaling.csv", index=False)
    print(table.drop(columns="dssp").to_string(index=False))

    summary = table.pivot_table(index="scale", columns="scaled", values="distance_deviation", aggfunc="mean")
    print("\nMean deviation in pairwise CA distance from the scale 1.0 fold (Angstrom):")
    print(summary.round(2).to_string())
    print(f"\nSaved {len(rows)} structures and {args.out / 'scaling.csv'}")


if __name__ == "__main__":
    main()
