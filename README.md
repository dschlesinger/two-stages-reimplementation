# two-stages-reimplementation

Reimplementation of [Two Stages of Folding: Convergent Mechanisms in AI Protein Folding Trunks](https://arxiv.org/abs/2602.06020) (Lu et al., 2026).

Work in progress. So far the repo contains setup, the ESMFold patching experiment and the charge directions:

```bash
pip install -r requirements.txt
python scripts/download_weights.py --out weights   # ~12.8 GB: ESMFold, OpenFold, Boltz-1
python scripts/download_data.py                    # ~50 MB: the authors' datasets, into reference/data/
```

See the comments in `requirements.txt` for OpenFold and DSSP, which are installed separately.

## ESMFold patching

```bash
python scripts/patch_esmfold.py full      # patch s and z at all 48 blocks, 4,870 donor-target pairs
python scripts/patch_esmfold.py single    # one block at a time, for the pairs where full patching made a hairpin
```

Results go to `outputs/patching/`: one CSV row per fold, plus the per-block success rates and plot (Fig. 1). Both stages resume when rerun, and `--limit N` runs the first N cases. Needs a GPU and `mkdssp` on the PATH.

## Charge directions

```bash
python scripts/charge_directions.py       # difference of means over the 200 helical training proteins (Eq. 4)
```

Saves `outputs/charge/charge_directions.pt` with unit charge (K/R/H minus D/E), positive and negative (each minus the neutral mean) directions, their projection spreads for steering in sigmas, and the per-amino-acid means. Index k is the sequence state entering trunk block k, and the last index is the trunk output. `charge_evaluation.csv` has the per-depth ROC-AUC on the test split. On the cluster: `qsub scripts/patch_esmfold.qsub scripts/charge_directions.py`.

```bash
python scripts/charge_shapes.py --preset clamp                 # steer a neutral poly-alanine chain
python scripts/charge_shapes.py --preset splay --strength 1 2 3 5
```

Folds a chain with no charged residues and steers chosen segments with those directions during an early block window, to see whether opposite charges pull the segments together and like charges push them apart. `--preset clamp` puts opposite charges near the two ends, `splay` puts the same charge on both, `zigzag` alternates three segments; `--segments 4-14:+ 26-36:-` sets them directly. Writes one PDB per fold plus `shapes.csv` with pLDDT, radius of gyration, end-to-end distance and the mean distance between each steered pair, to `outputs/charge/shapes/`. This goes past what the paper validates: its steering acts on helical motifs in real proteins, so expect low pLDDT on a uniform sequence and read the geometry rather than the DSSP string.

## Data

`scripts/download_data.py` pulls the authors' released datasets, which contain the exact donor-target pairs, patch regions, and probing splits with sequences inline. Add `--full` for their 1 GB block-patching results or `--results` for their result tables and figures.

The released data covers ESMFold with hairpin donors and helical targets. It does not include MSAs, the helix-into-hairpin direction, or anything for OpenFold, Boltz-1, and the cross-model experiments; those have to be rebuilt from the recipe in the paper's appendix.

## Sources

- Paper: [arXiv:2602.06020](https://arxiv.org/abs/2602.06020)
- Project page: [folding.baulab.info](https://folding.baulab.info/)
- Authors' code (ESMFold only, MIT): [kevinlu4588/ProteinFolding](https://github.com/kevinlu4588/ProteinFolding)
- Authors' dataset: [kevinlu4588/ProteinFolding on Hugging Face](https://huggingface.co/datasets/kevinlu4588/ProteinFolding/tree/main)
- Anonymized code release cited in the paper: [anon_protein-2F6C](https://anonymous.4open.science/r/anon_protein-2F6C)
- Model weights: [facebook/esmfold_v1](https://huggingface.co/facebook/esmfold_v1), [nz/OpenFold](https://huggingface.co/nz/OpenFold), [boltz-community/boltz-1](https://huggingface.co/boltz-community/boltz-1)
