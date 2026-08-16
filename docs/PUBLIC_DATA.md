# Public fundus datasets (follow-up paper)

These cohorts are independently downloadable. They do **not** reproduce UK Biobank ICD labels or the Nature Medicine internal-test AUROCs. Demo / synthetic caches in this repo are for pipeline tests only.

Gout, osteoporosis, hyperlipidemia, and thyroid disease have **no** public CFP gold standard here — omit them or mark them UKB-only.

## Download

| Dataset | Access | Typical layout |
|---------|--------|----------------|
| **ODIR-5K** | [Grand Challenge](https://odir2019.grand-challenge.org/dataset/), [Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) | `full_df.csv` + left/right images; 5,000 patients |
| **BRSET** | [PhysioNet 1.0.2](https://www.physionet.org/content/brazilian-ophthalmological/1.0.2/) (credentialed) | image-level CSV with `patient_id`, `laterality`, demographics, quality |
| **RFMiD / 2.0** | IEEE DataPort, [Hugging Face](https://huggingface.co/datasets/ctmedtech/RFMID), [MDPI Data 2021](https://www.mdpi.com/2306-5729/6/2/14) | official `RFMiD_*_Labels.csv` train/val/test |
| APTOS 2019 / MESSIDOR | Kaggle / ADCIS | DR grades only — **not** systemic T2DM |

Print the same checklist (and optionally try Kaggle/HF if credentials exist; never auto-downloads BRSET):

```powershell
python scripts/download_public_data.py
python scripts/download_public_data.py --try-download
```

## Label mapping (pre-register these heads)

| Head in this repo | ODIR | BRSET | RFMiD | UKB / Reti-Pioneer |
|-------------------|------|-------|-------|--------------------|
| `N` | Normal | — | — | — |
| `D` / `diabetes_ocular` | Diabetes (ocular findings / DR) | — | `DR` (retinal) | **Not** T2DM ICD |
| `diabetes` | — | Self-reported / clinical diabetes | — | Closer to systemic, still not UKB T2DM |
| `G` | Glaucoma | — | — | — |
| `C` | Cataract | — | — | — |
| `A` / `ARMD` | AMD | AMD if present | `ARMD` | — |
| `H` / `hypertension_ocular` | Hypertension (ocular) | — | — | **Not** UKB hypertension |
| `hypertensive_retinopathy` | — | Specialist HR | hypertensive retinopathy if present | Ocular sign, not systemic HTN |
| `dr_referable` | — | DR grade ≥ 2 when column exists | — | Referable DR, not T2DM |
| `M` / `MYA` | Myopia | — | `MYA` | — |
| `O` / `OTHER` | Other | — | `OTHER` | — |
| `t2dm` `gout` `osteoporosis` `hyperlipemia` `thyroid` | omitted | omitted | omitted | UKB-only |

**Honesty notes (write in Methods):**

- ODIR **D** ≠ UKB T2DM ICD. Call it ocular-evidence diabetes.
- BRSET **diabetes**, **DR grade**, and **hypertensive retinopathy** are distinct heads. Do not pool them without a footnote.
- RFMiD is a long-tail retinal-sign ontology, not a six-disease oculomics replica.
- Public `UKB_y5.npz` / `UKB_y10.npz` written by `prepare_public_npz.py` are **copies of prevalence** for loader compatibility, not 5/10-year incidence.

## Build a feature cache

Images are **not** required for CI. If they are missing, the helper writes deterministic **synthetic** backbone features from labels (see `SYNTHETIC_FEATURES.txt` in the output folder). Do not put those AUROCs in a paper.

### Real images → aligned labels + foundation features

```powershell
# 1) Labels + pairs.csv in the same row order (drop rows without files)
python scripts/prepare_public_npz.py --dataset odir --src E:\data\odir --out data/odir `
  --labels-only --force --write-manifest --require-images

# 2) Dry-run alignment check
python scripts/extract_features.py --cache-dir data/odir --dry-run

# 3) Real extraction (CUDA + RETFound/Swin/Vim weights)
python scripts/extract_features.py --cache-dir data/odir --all-backbones

# CI-only stub (NOT for manuscript):
# python scripts/extract_features.py --cache-dir data/odir --all-backbones --stub-identity
```

If `pairs.csv` N ≠ `UKB_mqd.npz` N, extraction exits with an error — re-run step 1 with `--require-images`.

### Synthetic / CI caches

```powershell
# Tiny synthetic cache (no download)
python scripts/prepare_public_npz.py --dataset odir --synthetic-demo --n-samples 64 --out data/odir --force
python scripts/prepare_public_npz.py --dataset brset --synthetic-demo --out data/brset --force
python scripts/prepare_public_npz.py --dataset rfmid --synthetic-demo --out data/rfmid --force

# From a real CSV (still synthetic features until you extract backbones)
python scripts/prepare_public_npz.py --dataset odir --src E:\data\odir --out data/odir --force
```

Loaders (unit-testable without pixels): `dataset/odir.py`, `dataset/brset.py`, `dataset/rfmid.py`. Re-exports: `dataset/public_fundus.py`.

Patient-level split: both eyes / visits of one `patient_id` stay in one fold (`reti_pioneer/split.py`). RFMiD prefers the official train/val/test CSVs.

Metadata padding into `UKB_mqd.npz`: age, sex, **weight=0**, **ethnicity=0** (7-way one-hot still used in the fusion layer).

## Cross-dataset evaluation

Train on one public cache, score **overlapping heads only** on another. Default tasks: `diabetes_related`, `hypertension_ocular`.

```powershell
python scripts/prepare_public_npz.py --dataset odir --synthetic-demo --out data/odir --force
python scripts/prepare_public_npz.py --dataset brset --synthetic-demo --out data/brset --force
python scripts/train.py --config configs/ablation_full.yaml --dataset odir --data-dir data/odir --multitask --horizon 0
python scripts/evaluate.py --ckpt ckpt/<run>/multitask/y0 --dataset odir --data-dir data/odir --test-data-dir data/brset --test-dataset brset
```

| Canonical task | ODIR | BRSET | RFMiD | Honesty |
|----------------|------|-------|-------|---------|
| `diabetes_related` | `D` | `diabetes` | `DR` | Ocular or self-report vs retinal DR — not UKB T2DM |
| `diabetes_ocular` | `D` | `dr_referable` | `DR` | Closer ocular pairing |
| `hypertension_ocular` | `H` | `hypertensive_retinopathy` | `HR` if present | Ocular sign, not systemic HTN |

Do not put synthetic-cache AUROCs in a manuscript. Real ODIR→BRSET numbers require downloaded images and backbone features.
