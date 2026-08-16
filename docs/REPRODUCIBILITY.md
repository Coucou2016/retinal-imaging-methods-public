# Reproducibility and trust boundaries

## Demo vs production

| Aspect | Demo (`--demo`, synthetic `.npz`) | Production (paper cohorts) |
|--------|-----------------------------------|----------------------------|
| Data | `scripts/generate_demo_data.py` | UK Biobank + hospital CFPs, author preprocessing |
| Labels | Random Bernoulli (~25% positive) | ICD / adjudicated outcomes |
| Metrics | Sanity checks only (AUROC ∈ [0,1]) | Paper internal-test AUROCs (~0.70–0.83) |
| Training | 3 epochs, CPU, held-out val split | 20 epochs, GPU, full cohort |
| Weights | Random init | Official TorchScript + foundation checkpoints |

**Do not compare demo AUROC to clinical results in the Nature Medicine paper.**

## Seeds and splits

| Item | Default | Where |
|------|---------|--------|
| Demo tensor generation | `seed=42` | `configs/demo.yaml` → `demo.seed` |
| Train/val partition | `seed=42`, `val_fraction=0.25` | `configs/demo.yaml` → `split.*` |
| Balance sampler | `430` | `utils/run.py` |
| Split file | `split.npz` next to run | Written by `scripts/train.py` when `val_fraction > 0` |

Re-evaluate on the same partition:

```powershell
python scripts/evaluate.py --demo --disease t2dm --horizon 0 --split val --ckpt ckpt/<run>/t2dm/y0
```

## Software versions (record for your run)

```powershell
python --version
python -c "import torch, sklearn, ignite; print('torch', torch.__version__); print('sklearn', sklearn.__version__); print('ignite', ignite.__version__)"
```

Recommended: **Python 3.12** on Windows (full `lzma` stdlib). Python 3.13 may break `import torchvision` until liblzma is installed; **fast-mode training does not require torchvision**.

## Expected ranges on synthetic demo data

After `python scripts/train.py --demo --disease t2dm --horizon 0`:

- Validation **AUROC** and **AP** are defined (both classes in stratified val split).
- Typical demo val AUROC after 3 epochs: **~0.45–0.65** (noise features, not clinically meaningful).
- Training loss should decrease; checkpoint files appear under `ckpt/<timestamp>/t2dm/y0/ckpt/`.

## Checksums

Official UKB `.npz` bundles should be verified against author manifests when available. Demo files are regenerated deterministically from `--seed` (no clinical checksum).

## Verification commands

```powershell
cd E:\Projects\20260522-retinal-imaging
python -m unittest discover -s tests -v
python scripts/train.py --demo --disease t2dm --horizon 0
python scripts/evaluate.py --demo --disease t2dm --horizon 0 --split val --ckpt ckpt/<latest>/t2dm/y0
python scripts/train.py --demo --multitask --learnable-q --horizon 0
python scripts/setup_pretrained.py
```
