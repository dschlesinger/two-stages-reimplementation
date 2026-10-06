# two-stages-reimplementation

Reimplementation of [Two Stages of Folding: Convergent Mechanisms in AI Protein Folding Trunks](https://arxiv.org/abs/2602.06020) (Lu et al., 2026).

Work in progress. So far the repo only contains setup:

```bash
pip install -r requirements.txt
python scripts/download_weights.py --out weights   # ~12.8 GB: ESMFold, OpenFold, Boltz-1
```

See the comments in `requirements.txt` for OpenFold and DSSP, which are installed separately.
