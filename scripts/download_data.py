#!/usr/bin/env python3
"""Download the paper authors' datasets from the Hugging Face Hub.

Source: https://huggingface.co/datasets/kevinlu4588/ProteinFolding

By default this fetches the small dataset files (~50 MB) into reference/data/:

    patching_dataset.csv                  donor-target pairs with sequences and patch regions
    single_block_patching_successes.csv   cases used for the stage-1 analyses
    target_loops_dataset.csv              helix-loop-helix regions for charge steering
    probing_train_test.csv                probing train/test split
    alpha_helical_train.csv               helical proteins for the charge direction
    donor_hairpins.csv, bb_motifs_local.csv, cath_all_alpha.*   donor and source pools

Usage:
    pip install huggingface_hub
    python scripts/download_data.py                 # dataset CSVs only
    python scripts/download_data.py --full          # also the 1 GB block-patching parquet
    python scripts/download_data.py --results       # also the authors' result tables and figures

The data covers ESMFold with hairpin donors and helical targets only. It has no
MSAs and nothing for OpenFold, Boltz-1, or the cross-model experiments.
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("huggingface_hub is not installed: pip install huggingface_hub")

REPO_ID = "kevinlu4588/ProteinFolding"
# The authors' own block-patching output; large and not needed to build inputs.
LARGE_FILES = ["data/all_block_patching_results.parquet"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("reference"), help="destination directory (default: ./reference)")
    p.add_argument("--full", action="store_true", help="include the 1 GB all_block_patching_results.parquet")
    p.add_argument("--results", action="store_true", help="include the authors' results/ folder (~75 MB)")
    return p.parse_args()


def main():
    args = parse_args()
    allow = ["data/*"] + (["results/*"] if args.results else [])
    ignore = None if args.full else LARGE_FILES

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{REPO_ID} -> {args.out.resolve()}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=allow,
        ignore_patterns=ignore,
        local_dir=args.out,
    )

    files = sorted(f for f in (args.out / "data").glob("*") if f.is_file())
    if not files:
        sys.exit("No dataset files were downloaded; the repo layout may have changed.")
    for f in files:
        print(f"    {f.stat().st_size / 1e6:8.2f} MB  {f.relative_to(args.out)}")
    print(f"\nDone: {len(files)} dataset files in {args.out / 'data'}")


if __name__ == "__main__":
    main()
