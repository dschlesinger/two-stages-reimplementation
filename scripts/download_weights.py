#!/usr/bin/env python3
"""Download pretrained weights for ESMFold, OpenFold, and Boltz-1.

All three are pulled from the Hugging Face Hub (~12.8 GB total by default):

    esmfold   facebook/esmfold_v1       8.4 GB  (ESM-2 3B encoder + trunk + structure module)
    openfold  nz/OpenFold               0.4 GB  (one checkpoint; ~3.3 GB with --openfold-all)
    boltz1    boltz-community/boltz-1   4.0 GB  (boltz1_conf.ckpt + ccd.pkl)

Usage:
    pip install huggingface_hub
    python scripts/download_weights.py --out /path/to/weights
    python scripts/download_weights.py --out /path/to/weights --models esmfold boltz1

Downloads resume if interrupted. Set HF_TOKEN if you hit anonymous rate limits.
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("huggingface_hub is not installed: pip install huggingface_hub")

# OpenFold publishes several fine-tuned checkpoints. The paper does not say
# which one it uses; finetuning_ptm_2.pt is OpenFold's own default (model_1_ptm).
OPENFOLD_DEFAULT_CKPT = "finetuning_ptm_2.pt"

MODELS = {
    "esmfold": {
        "repo": "facebook/esmfold_v1",
        "patterns": ["config.json", "pytorch_model.bin", "*.txt", "*token*.json"],
        "gb": 8.5,
    },
    "openfold": {
        "repo": "nz/OpenFold",
        "patterns": [OPENFOLD_DEFAULT_CKPT, "LICENSE"],
        "gb": 0.4,
    },
    "boltz1": {
        "repo": "boltz-community/boltz-1",
        # boltz1.ckpt (6.9 GB) is the full training checkpoint; inference
        # only needs the confidence checkpoint and the CCD dictionary.
        "patterns": ["boltz1_conf.ckpt", "ccd.pkl"],
        "gb": 4.0,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("weights"), help="destination directory (default: ./weights)")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS), help="which models to fetch")
    p.add_argument("--openfold-ckpt", default=OPENFOLD_DEFAULT_CKPT, help="OpenFold checkpoint file name in nz/OpenFold")
    p.add_argument("--openfold-all", action="store_true", help="fetch every OpenFold checkpoint (~3.3 GB)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.openfold_all:
        MODELS["openfold"]["patterns"] = ["*.pt", "LICENSE"]
        MODELS["openfold"]["gb"] = 3.3
    else:
        MODELS["openfold"]["patterns"] = [args.openfold_ckpt, "LICENSE"]

    args.out.mkdir(parents=True, exist_ok=True)
    need_gb = sum(MODELS[m]["gb"] for m in args.models)
    free_gb = shutil.disk_usage(args.out).free / 1e9
    print(f"Destination: {args.out.resolve()}  (need ~{need_gb:.1f} GB, {free_gb:.1f} GB free)")
    if free_gb < need_gb * 1.05:
        sys.exit("Not enough free disk space; pick another --out or fewer --models.")

    for name in args.models:
        spec = MODELS[name]
        dest = args.out / name
        print(f"\n[{name}] {spec['repo']} -> {dest}")
        snapshot_download(repo_id=spec["repo"], allow_patterns=spec["patterns"], local_dir=dest)
        files = sorted(f for f in dest.rglob("*") if f.is_file() and ".cache" not in f.parts)
        if not any(f.stat().st_size > 1e8 for f in files):
            sys.exit(f"[{name}] no weight file was downloaded; check patterns {spec['patterns']}")
        for f in files:
            print(f"    {f.stat().st_size / 1e9:6.2f} GB  {f.relative_to(args.out)}")

    print("\nDone. Load with:")
    print(f"  ESMFold : EsmForProteinFolding.from_pretrained('{args.out / 'esmfold'}')")
    print(f"  OpenFold: --openfold_checkpoint_path {args.out / 'openfold' / args.openfold_ckpt}")
    print(f"  Boltz-1 : boltz predict ... --cache {args.out / 'boltz1'}")


if __name__ == "__main__":
    main()
