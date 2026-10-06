#!/usr/bin/env python3
"""Activation patching on the ESMFold trunk: hairpin donors into helical targets.

Reimplements the patching experiment of section 3 (Fig. 1) on the authors'
donor-target pairs. A donor containing a beta hairpin and a helical target are
both folded without recycling; the target's sequence state s and/or pairwise
state z in the patch region are overwritten with the donor's hairpin region at
the output of a trunk block, and DSSP decides whether the result has a hairpin.

Stages:
    full     patch s and z at all 48 blocks, for every case in patching_dataset.csv
    single   for the cases where full patching produced a hairpin, patch s or z
             alone at one block at a time (96 folds per case)
    summary  recompute the success-rate table and plot from the single-block results

Usage:
    python scripts/patch_esmfold.py full
    python scripts/patch_esmfold.py single
    python scripts/patch_esmfold.py full --limit 20      # quick check

Both stages append to a CSV in --out after every case and resume from it when
rerun. Needs the mkdssp binary on PATH (see requirements.txt) and a GPU: the
paper reports about 1.5 days on 2x A6000 for the single-block stage.

Pairwise patches use the intra-region mask of the main text (z_ij with both i
and j in the patch region); the "touch" mask of App. I.2 is not implemented.
"""

import argparse
import csv
import re
import shutil
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch
from Bio.PDB import DSSP, PDBParser
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, EsmForProteinFolding

HUB_ID = "facebook/esmfold_v1"
FULL_CSV = "full_patching.csv"
SINGLE_CSV = "single_block_patching.csv"
FIELDS = [
    "case_id", "target_name", "donor_pdb", "patch_mode", "block",
    "hairpin_found", "hairpin_in_patch", "plddt", "patch_plddt", "dssp",
]
# ESMFold writes neither record, and mkdssp 4.x refuses PDB files without them.
PDB_HEADER = (
    "HEADER    ESMFOLD PREDICTION\n"
    "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
)


def fold(model, tokenizer, sequence, copies=1):
    inputs = tokenizer([sequence] * copies, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        return model(**inputs, num_recycles=0)


def collect_donor(model, tokenizer, sequence, start, end):
    """Donor s [blocks, P, d_s] and z [blocks, P, P, d_z] in the hairpin region, at every block output."""
    s_blocks, z_blocks = [], []

    def save(_module, _inputs, output):
        s, z = output
        s_blocks.append(s[0, start:end].clone())
        z_blocks.append(z[0, start:end, start:end].clone())

    handles = [block.register_forward_hook(save) for block in model.trunk.blocks]
    try:
        fold(model, tokenizer, sequence)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(s_blocks), torch.stack(z_blocks)


@contextmanager
def patching(model, donor_s, donor_z, region, jobs):
    """Patch the batch so that element b gets jobs[b] = (mode, block); block None means every block."""

    def make_hook(k):
        def hook(_module, _inputs, output):
            s, z = output
            for b, (mode, block) in enumerate(jobs):
                if mode == "none" or block not in (None, k):
                    continue
                if mode in ("sequence", "both"):
                    s[b, region] = donor_s[k]
                if mode in ("pairwise", "both"):
                    z[b, region, region] = donor_z[k]
        return hook

    handles = [block.register_forward_hook(make_hook(k)) for k, block in enumerate(model.trunk.blocks)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def run_dssp(pdb, length):
    """8-state DSSP string for a predicted structure, or None if mkdssp fails."""
    # mkdssp also rejects the PARENT record that ESMFold does write.
    body = "\n".join(line for line in pdb.splitlines() if not line.startswith(("PARENT", "REMARK 220")))
    with tempfile.NamedTemporaryFile("w", suffix=".pdb") as f:
        f.write(PDB_HEADER + body + "\n")
        f.flush()
        structure = PDBParser(QUIET=True).get_structure("pred", f.name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dssp = DSSP(structure[0], f.name, dssp="mkdssp")
        except Exception as e:
            warnings.warn(f"DSSP failed: {e}")
            return None
    ss = ["-"] * length
    for key in dssp.keys():
        ss[key[1][1] - 1] = dssp[key][2]
    return "".join(ss)


def find_hairpins(ss, min_strand=2, max_loop=5):
    """(start, end) of each hairpin: two consecutive strands (E or B) joined by a short loop (App. E.1)."""
    strands = [m.span() for m in re.finditer("[EB]{%d,}" % min_strand, ss)]
    return [(a0, b1) for (a0, a1), (b0, b1) in zip(strands, strands[1:]) if b0 - a1 <= max_loop]


def score(model, outputs, case, jobs, pool):
    start, end = int(case.target_patch_start), int(case.target_patch_end)
    length = len(case.target_sequence)
    plddt = outputs.plddt[:, :, 1].float().cpu()  # CA atoms
    rows = []
    for b, ss in enumerate(pool.map(lambda pdb: run_dssp(pdb, length), model.output_to_pdb(outputs))):
        mode, block = jobs[b]
        hairpins = find_hairpins(ss) if ss is not None else []
        rows.append({
            "case_id": case.Index,
            "target_name": case.target_name,
            "donor_pdb": case.donor_pdb,
            "patch_mode": mode,
            "block": "" if mode == "none" else "all" if block is None else block,
            # The authors count a hairpin anywhere in the chain; hairpin_in_patch requires it to overlap the patch.
            "hairpin_found": "" if ss is None else bool(hairpins),
            "hairpin_in_patch": "" if ss is None else any(h0 < end and h1 > start for h0, h1 in hairpins),
            "plddt": round(plddt[b].mean().item(), 4),
            "patch_plddt": round(plddt[b, start:end].mean().item(), 4),
            "dssp": ss or "",
        })
    return rows


def finished_cases(path, rows_per_case):
    """Case ids already in the results file. A case cut off mid-write is dropped so it gets rerun."""
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    counts = df.groupby("case_id", sort=False).size()
    partial = counts[counts != rows_per_case]
    if len(partial) > 1 or (len(partial) == 1 and partial.index[0] != df.case_id.iloc[-1]):
        sys.exit(f"{path} was written with different settings; move it or pick another --out.")
    if len(partial):
        df[df.case_id != partial.index[0]].to_csv(path, index=False)
    return set(counts.index) - set(partial.index)


def run(args, cases, jobs, path):
    done = finished_cases(path, len(jobs))
    todo = cases[~cases.index.isin(done)]
    print(f"{len(cases)} cases, {len(done & set(cases.index))} already done, {len(jobs)} folds per case -> {path}")
    if todo.empty:
        return

    print(f"Loading ESMFold from {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmForProteinFolding.from_pretrained(args.model).to(args.device).eval()

    new_file = not path.exists()
    with open(path, "a", newline="") as f, ThreadPoolExecutor(args.dssp_workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for n, case in enumerate(tqdm(todo.itertuples(), total=len(todo), desc="cases")):
            d0, d1 = int(case.donor_hairpin_start), int(case.donor_hairpin_end)
            region = slice(int(case.target_patch_start), int(case.target_patch_end))
            if d1 - d0 != region.stop - region.start:
                sys.exit(f"case {case.Index}: donor hairpin and target patch region differ in length")
            donor_s, donor_z = collect_donor(model, tokenizer, case.donor_sequence, d0, d1)

            rows = []
            for i in range(0, len(jobs), args.batch_size):
                batch = jobs[i:i + args.batch_size]
                with patching(model, donor_s, donor_z, region, batch):
                    outputs = fold(model, tokenizer, case.target_sequence, len(batch))
                rows += score(model, outputs, case, batch, pool)
            if n == 0 and all(r["dssp"] == "" for r in rows):
                sys.exit("DSSP failed on every structure of the first case; check the mkdssp install.")
            writer.writerows(rows)
            f.flush()


def read_results(path):
    if not path.exists():
        sys.exit(f"{path} not found; run the earlier stage first.")
    df = pd.read_csv(path, dtype={"block": str}).dropna(subset=["hairpin_found"])
    return df.astype({"hairpin_found": bool, "hairpin_in_patch": bool})


def summarize_full(out):
    df = read_results(out / FULL_CSV)
    patched, clean = df[df.patch_mode == "both"], df[df.patch_mode == "none"]
    print(f"Full patching: hairpin in {patched.hairpin_found.sum()} of {len(patched)} cases "
          f"({patched.hairpin_found.mean():.1%}); {patched.hairpin_in_patch.sum()} overlap the patch region.")
    print(f"Unpatched targets that already contain a hairpin: {clean.hairpin_found.sum()} of {len(clean)} cases.")


def summarize_single(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = read_results(out / SINGLE_CSV)
    df["block"] = df.block.astype(int)
    rate = df.pivot_table(index="block", columns="patch_mode", values="hairpin_found", aggfunc="mean")
    rate.to_csv(out / "single_block_success.csv")
    print(f"Single-block success rate over {df.case_id.nunique()} cases:")
    print(rate.round(3).to_string())

    colors = {"sequence": "tab:orange", "pairwise": "tab:green", "both": "tab:gray"}
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for mode in rate.columns:
        ax.plot(rate.index, rate[mode], marker="o", markersize=3, color=colors.get(mode), label=mode)
    ax.set(xlabel="Patched trunk block", ylabel="Hairpin success rate", title="ESMFold single-block patching")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "single_block_success.png", dpi=200)
    print(f"Saved {out / 'single_block_success.csv'} and .png")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["full", "single", "summary"])
    p.add_argument("--data", type=Path, default=Path("reference/data/patching_dataset.csv"), help="donor-target pairs")
    p.add_argument("--out", type=Path, default=Path("outputs/patching"), help="results directory")
    p.add_argument("--model", default="weights/esmfold", help=f"local ESMFold directory; falls back to {HUB_ID}")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--modes", nargs="+", choices=["sequence", "pairwise", "both"], default=["sequence", "pairwise"],
                   help="what to patch in the single stage")
    p.add_argument("--batch-size", type=int, default=8, help="patched copies of the target folded together")
    p.add_argument("--limit", type=int, help="only run the first N cases")
    p.add_argument("--dssp-workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "summary":
        return summarize_single(args.out)
    if shutil.which("mkdssp") is None:
        sys.exit("mkdssp not found on PATH: conda install -c conda-forge dssp, or apt-get install dssp")
    if not args.data.exists():
        sys.exit(f"{args.data} not found: python scripts/download_data.py")
    if not Path(args.model).exists():
        args.model = HUB_ID
    args.out.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(args.data)

    if args.stage == "full":
        run(args, cases.head(args.limit), [("none", None), ("both", None)], args.out / FULL_CSV)
        summarize_full(args.out)
    else:
        full = read_results(args.out / FULL_CSV)
        success = full[(full.patch_mode == "both") & full.hairpin_found].case_id
        cases = cases.loc[success].head(args.limit)
        num_blocks = AutoConfig.from_pretrained(args.model).esmfold_config.trunk.num_blocks
        jobs = [(mode, k) for mode in args.modes for k in range(num_blocks)]
        run(args, cases, jobs, args.out / SINGLE_CSV)
        summarize_single(args.out)


if __name__ == "__main__":
    main()
