# Paper reference — Reti-Pioneer

## Citation

Zhang, X., Li, Q., Liang, Y. *et al.* **AI framework for multidisease detection via retinal imaging.** *Nature Medicine* (2026).

- DOI: [10.1038/s41591-026-04359-w](https://doi.org/10.1038/s41591-026-04359-w)
- Official code: [github.com/lyhyl/Reti-Pioneer](https://github.com/lyhyl/Reti-Pioneer)
- Website: [retipioneer.cn](https://www.retipioneer.cn)

## Clinical scope (6 diseases)

| Key | Label in repo | Internal-test AUROC (paper) |
|-----|----------------|------------------------------|
| T2D | `t2dm` | 0.833 |
| Gout | `gout` | 0.832 |
| Osteoporosis | `osteoporosis` | 0.787 |
| Hypertension | `hypertension` | 0.740 |
| Hyperlipidemia | `hyperlipemia` | 0.736 |
| Thyroid | `thyroid` | 0.699 |

Longitudinal horizons: **0** (cross-sectional), **5** and **10** years (UK Biobank incident subset).

## Architecture (Methods summary)

1. **Inputs**: bilateral color fundus photos (CFPs) + structured metadata (age, sex, weight, ethnicity) + image-quality scores (good / usable / bad).
2. **Backbones (frozen)**: RETFound (ViT-L), Swin Transformer V2-B, Vision Mamba-S — each with a learnable head.
3. **Quality-aware fusion**: bilinear fusion of deep features with quality probabilities; SELU + linear classifier per backbone.
4. **Ensemble**: weighted soft voting in training; max pooling at inference across three heads.
5. **Training data scale**: 107,730 CFPs / 53,865 participants (UK Biobank + Chinese hospital cohorts).

## What this repository implements

| Component | Status |
|-----------|--------|
| `ComplexModel` / `FuseBase` / `QualityAware` | Official clone plus K-class head and `learnable_q` |
| Training on pre-extracted `.npz` features | `scripts/train.py` (`--multitask`, `--learnable-q`, `--dataset`) |
| Demo/synthetic UKB tensors | `scripts/generate_demo_data.py` |
| Inference API (images + metadata) | `inference.py` (needs GPU + weights) |
| Public fundus loaders | ODIR / BRSET / RFMiD + APTOS/MESSIDOR; see `docs/PUBLIC_DATA.md` |
| Public feature cache | `scripts/prepare_public_npz.py` (synthetic fallback) |
| Feature extraction pipeline | `scripts/extract_features.py` |
| Calibration / decision-curve / temperature | `utils/calibration.py`, `scripts/evaluate.py --calibrate` |

## What cannot be fully replicated without approved data access

- **UK Biobank** fundus images and disease labels (controlled access).
- **Chinese hospital / SEED** external validation cohorts (institutional approval).
- **EyeQ** quality model weights (third-party download).
- **Vision Mamba** Linux wheels in upstream `pyproject.toml` (optional; training uses cached features).
- **Prospective silent trial / clinical pilot** workflows (deployment + IRB).

Use `scripts/generate_demo_data.py` for local development and CI; replace with real `UKB_*.npz` after preprocessing per official README.
