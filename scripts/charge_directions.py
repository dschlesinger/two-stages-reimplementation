#!/usr/bin/env python3
"""Charge directions in the ESMFold trunk's sequence representation (section 4.2, Eq. 4).

Folds the alpha-helical training proteins of probing_train_test.csv without
recycling and records the sequence state s at every depth of the trunk. At each
depth the mean s of every amino acid is taken, and the positive (K, R, H),
negative (D, E) and neutral (the other 15) class means are averages of those
per-amino-acid means, as in Eq. 4. Three unit directions are saved per depth:

    charge     mean_P - mean_N, the paper's v_charge: adding it makes a residue
               more positive, subtracting it makes it more negative
    positive   mean_P - mean_neutral
    negative   mean_N - mean_neutral

each with the standard deviation of the charged training residues' projections
onto it, so steering strength can be given in sigmas (the paper uses 3 sigma).
The per-amino-acid means are saved too, so other contrasts, such as alanine as
the neutral reference, need no refolding.

Depth k is the state entering trunk block k, which is where steering at block k
adds the direction; depth 48 is the trunk output. The authors' released
directions index block outputs, so their block k is depth k + 1 here. Their code
also pools all charged residues instead of averaging per amino acid; the
cos_pooled column of the evaluation shows how much that changes the direction.

The test split (helical and hairpin proteins) checks the directions: ROC-AUC of
the charge projection for positive against negative residues (App. K.1, Fig. 15).

Usage:
    python scripts/charge_directions.py
    python scripts/charge_directions.py --limit 10      # quick check

Writes charge_directions.pt and charge_evaluation.csv to --out. Needs a GPU;
about 400 folds.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

from patch_esmfold import HUB_ID, fold

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
POSITIVE, NEGATIVE = "KRH", "DE"
NEUTRAL = "".join(a for a in AMINO_ACIDS if a not in POSITIVE + NEGATIVE)


def trunk_states(model, tokenizer, sequence):
    """s entering each trunk block, then the trunk output: [blocks + 1, L, d_s]."""
    states = []
    blocks = model.trunk.blocks
    handles = [blocks[0].register_forward_pre_hook(lambda _module, inputs: states.append(inputs[0][0].clone()))]
    handles += [block.register_forward_hook(lambda _module, _inputs, output: states.append(output[0][0].clone()))
                for block in blocks]
    try:
        fold(model, tokenizer, sequence)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(states)


def residue_index(sequence, device):
    return torch.tensor([AMINO_ACIDS.index(a) for a in sequence], device=device)


def charged_mask(sequence, device):
    return torch.tensor([a in POSITIVE + NEGATIVE for a in sequence], device=device)


def collect(model, tokenizer, sequences):
    """Per-amino-acid sums of s at every depth, plus first and second moments of the charged residues."""
    stats = {}
    for sequence in tqdm(sequences, desc="train"):
        s = trunk_states(model, tokenizer, sequence).double()
        if not stats:
            depths, _, dim = s.shape
            stats = {
                "sums": s.new_zeros(depths, len(AMINO_ACIDS), dim),
                "counts": torch.zeros(len(AMINO_ACIDS), dtype=torch.long),
                "charged_sum": s.new_zeros(depths, dim),
                "charged_gram": s.new_zeros(depths, dim, dim),
                "n_charged": 0,
            }
        idx = residue_index(sequence, s.device)
        stats["sums"].index_add_(1, idx, s)
        stats["counts"] += torch.bincount(idx.cpu(), minlength=len(AMINO_ACIDS))
        x = s[:, charged_mask(sequence, s.device)]
        stats["charged_sum"] += x.sum(1)
        stats["charged_gram"] += x.transpose(1, 2) @ x
        stats["n_charged"] += x.shape[1]
    return stats


def directions(stats):
    counts = stats["counts"].to(stats["sums"].device)
    means = stats["sums"] / counts.clamp(min=1)[:, None]

    def class_mean(group):
        present = [AMINO_ACIDS.index(a) for a in group if counts[AMINO_ACIDS.index(a)] > 0]
        if not present:
            sys.exit(f"no training residues of {group}; use more sequences")
        return means[:, present].mean(1)

    def pooled_mean(group):
        idx = [AMINO_ACIDS.index(a) for a in group]
        return stats["sums"][:, idx].sum(1) / counts[idx].sum()

    def unit(v):
        return v / v.norm(dim=-1, keepdim=True)

    # Spread of the charged training residues along a direction, from their mean and second moment.
    mu = stats["charged_sum"] / stats["n_charged"]
    cov = stats["charged_gram"] / stats["n_charged"] - mu[:, :, None] * mu[:, None, :]

    def spread(d):
        return torch.einsum("kd,kde,ke->k", d, cov, d).clamp(min=0).sqrt()

    pos, neg, neutral = class_mean(POSITIVE), class_mean(NEGATIVE), class_mean(NEUTRAL)
    out = {
        "charge": unit(pos - neg),
        "positive": unit(pos - neutral),
        "negative": unit(neg - neutral),
    }
    out.update({f"{name}_std": spread(d) for name, d in list(out.items())})
    out["pooled_charge"] = unit(pooled_mean(POSITIVE) - pooled_mean(NEGATIVE))
    out["aa_means"] = means
    return out


def evaluate(model, tokenizer, sequences, charge):
    """ROC-AUC per depth of the charge projection, positive against negative residues."""
    projections, labels = [], []
    for sequence in tqdm(sequences, desc="test"):
        s = trunk_states(model, tokenizer, sequence).double()
        mask = charged_mask(sequence, s.device)
        projections.append(torch.einsum("kld,kd->kl", s[:, mask], charge).cpu())
        labels += [a in POSITIVE for a in sequence if a in POSITIVE + NEGATIVE]
    projections = torch.cat(projections, dim=1)
    return [roc_auc_score(labels, p) for p in projections.numpy()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("reference/data/probing_train_test.csv"),
                   help="probing train/test split")
    p.add_argument("--out", type=Path, default=Path("outputs/charge"), help="results directory")
    p.add_argument("--model", default="weights/esmfold", help=f"local ESMFold directory; falls back to {HUB_ID}")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, help="only use the first N train and N test sequences")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.data.exists():
        sys.exit(f"{args.data} not found: python scripts/download_data.py")
    if not Path(args.model).exists():
        args.model = HUB_ID
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df = df[df.FullChainSequence.map(lambda s: set(s) <= set(AMINO_ACIDS))]
    train = df[(df.split == "train") & (df.structure_type == "alpha_helical")].FullChainSequence.head(args.limit).tolist()
    test = df[df.split == "test"].FullChainSequence.head(args.limit).tolist()
    print(f"{len(train)} helical training proteins, {len(test)} test proteins -> {args.out}")

    print(f"Loading ESMFold from {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmForProteinFolding.from_pretrained(args.model).to(args.device).eval()

    stats = collect(model, tokenizer, train)
    found = directions(stats)
    auc = evaluate(model, tokenizer, test, found["charge"])

    saved = {name: v.float().cpu() for name, v in found.items() if name != "pooled_charge"}
    saved.update({
        "depths": "index k is s entering trunk block k; the last index is the trunk output",
        "amino_acids": AMINO_ACIDS,
        "aa_counts": stats["counts"],
        "train_sequences": len(train),
    })
    torch.save(saved, args.out / "charge_directions.pt")

    evaluation = pd.DataFrame({
        "depth": range(len(auc)),
        "roc_auc": auc,
        "cos_pooled": (found["charge"] * found["pooled_charge"]).sum(-1).cpu(),
        "cos_positive_negative": (found["positive"] * found["negative"]).sum(-1).cpu(),
        "charge_std": found["charge_std"].cpu(),
    })
    evaluation.to_csv(args.out / "charge_evaluation.csv", index=False)
    print(evaluation.round(3).to_string(index=False))
    print(f"Saved {args.out / 'charge_directions.pt'} and {args.out / 'charge_evaluation.csv'}")


if __name__ == "__main__":
    main()
