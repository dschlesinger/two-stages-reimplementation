#!/usr/bin/env python3
"""Regenerate structures for cases where full patching produced a hairpin.

The patching stages keep only DSSP strings and pLDDT. This refolds a sample of
the successful cases from full_patching.csv exactly as the full stage did
(unpatched and patched target in one batch, no recycling) and writes both
predictions as PDB files:

    <case_id>_<target>_original.pdb           unpatched target
    <case_id>_<target>_<donor>_patched.pdb    s and z patched at all 48 blocks

index.csv lists the files with the patch region (1-based, inclusive, as numbered
in the PDB files) and whether the refolded structure still has the hairpin.

Usage:
    python scripts/save_patched_structures.py                  # 20 random successes
    python scripts/save_patched_structures.py -n 50 --seed 1
    python scripts/save_patched_structures.py --cases 12 340 2718

Needs a GPU. mkdssp is only used for the hairpin check and is skipped if missing.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

from patch_esmfold import FULL_CSV, HUB_ID, collect_donor, find_hairpins, fold, patching, read_results, run_dssp

JOBS = [("none", None), ("both", None)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=20, help="number of successful cases to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cases", type=int, nargs="+", help="save these case ids instead of sampling")
    p.add_argument("--data", type=Path, default=Path("reference/data/patching_dataset.csv"), help="donor-target pairs")
    p.add_argument("--out", type=Path, default=Path("outputs/patching"), help="results directory")
    p.add_argument("--results", type=Path, help=f"full-stage results; defaults to <out>/{FULL_CSV}")
    p.add_argument("--structures", type=Path, help="where to write the PDB files; defaults to <out>/structures")
    p.add_argument("--model", default="weights/esmfold", help=f"local ESMFold directory; falls back to {HUB_ID}")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def pick_cases(args):
    full = read_results(args.results or args.out / FULL_CSV)
    success = full[(full.patch_mode == "both") & full.hairpin_found].case_id
    if args.cases:
        missing = sorted(set(args.cases) - set(success))
        if missing:
            sys.exit(f"not successful full-patching cases: {missing}")
        return args.cases
    return sorted(success.sample(min(args.n, len(success)), random_state=args.seed))


def main():
    args = parse_args()
    if not args.data.exists():
        sys.exit(f"{args.data} not found: python scripts/download_data.py")
    if not Path(args.model).exists():
        args.model = HUB_ID
    check = shutil.which("mkdssp") is not None
    if not check:
        print("mkdssp not found on PATH; skipping the hairpin check.")
    structures = args.structures or args.out / "structures"
    structures.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(args.data).loc[pick_cases(args)]

    print(f"Loading ESMFold from {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmForProteinFolding.from_pretrained(args.model).to(args.device).eval()

    index = []
    for case in tqdm(cases.itertuples(), total=len(cases), desc="cases"):
        d0, d1 = int(case.donor_hairpin_start), int(case.donor_hairpin_end)
        start, end = int(case.target_patch_start), int(case.target_patch_end)
        donor_s, donor_z = collect_donor(model, tokenizer, case.donor_sequence, d0, d1)
        with patching(model, donor_s, donor_z, slice(start, end), JOBS):
            outputs = fold(model, tokenizer, case.target_sequence, len(JOBS))
        original, patched = model.output_to_pdb(outputs)

        names = (f"{case.Index}_{case.target_name}_original.pdb",
                 f"{case.Index}_{case.target_name}_{case.donor_pdb}_patched.pdb")
        for name, pdb in zip(names, (original, patched)):
            (structures / name).write_text(pdb)

        ss = run_dssp(patched, len(case.target_sequence)) if check else None
        index.append({
            "case_id": case.Index,
            "target_name": case.target_name,
            "donor_pdb": case.donor_pdb,
            "patch_first_residue": start + 1,
            "patch_last_residue": end,
            "hairpin_found": "" if ss is None else bool(find_hairpins(ss)),
            "dssp": ss or "",
            "original": names[0],
            "patched": names[1],
        })

    index = pd.DataFrame(index)
    index.to_csv(structures / "index.csv", index=False)
    print(f"Saved {2 * len(index)} structures and index.csv to {structures}")
    if check and not index.hairpin_found.eq(True).all():
        print(f"No hairpin on refolding: cases {index.case_id[index.hairpin_found.ne(True)].tolist()}")


if __name__ == "__main__":
    main()
