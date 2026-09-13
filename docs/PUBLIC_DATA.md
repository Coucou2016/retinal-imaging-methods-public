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

### Workspace download status (2026-08-16 iter-5)

| Dataset | Labels | Images | Foundation features | Notes |
|---------|--------|--------|---------------------|-------|
| **ODIR-5K** | not downloaded | not downloaded | SYNTHETIC demo cache only | No `kaggle` CLI / `~/.kaggle/kaggle.json` |
| **BRSET** | blocked | blocked | SYNTHETIC demo cache only | PhysioNet credentialing / DUA required; script never auto-downloads |
| **RFMiD** | **real** official CSVs (n=3200, K=46) under `data/raw/rfmid/` | sample PNGs only (3 training images for path check); full ~3200 images not bulk-pulled | **SYNTHETIC** in `data/rfmid/` (`SYNTHETIC_FEATURES.txt`) | CUDA unavailable (`torch.cuda.is_available()==False`) → RETFound/Swin extraction **待补充** |

Honesty: real RFMiD *labels* do not authorize manuscript AUROCs until real pixels + foundation features replace the synthetic backbone caches.

## Label mapping / endpoint harmonization (pre-register these heads)

Endpoints carry kind (`systemic` | `ocular_manifestation`) and alignment
(`direct` | `partial` | `related_not_equivalent`). Clinical tables require
`alignment=direct` (`scripts/evaluate.py --clinical-tables` / `--paper-mode`).

| Endpoint | Kind | ODIR | BRSET | RFMiD | UKB / Reti-Pioneer | Alignment |
|----------|------|------|-------|-------|--------------------|-----------|
| `diabetes_ocular` | ocular_manifestation | D | `dr_referable` | DR | **Not** T2DM ICD | direct / partial |
| `diabetes_systemic` | systemic | — | diabetes | — | t2dm (still not identical ICD coding) | direct on BRSET/UKB |
| `diabetes_related` | exploratory only | D | diabetes | DR | t2dm | **related_not_equivalent** |
| `hypertension_ocular` | ocular_manifestation | H | hypertensive_retinopathy | — | hypertension (related) | direct ocular |
| `N` | ocular | Normal | — | — | — | — |
| `G` / `C` / `A` / `M` / `O` | ocular | as labeled | — | ARMD/MYA/OTHER when present | — | — |
| gout / osteoporosis / hyperlipemia / thyroid | systemic | omitted | omitted | omitted | UKB-only | — |

**Honesty notes (write in Methods):**

- ODIR **D** ≠ UKB T2DM ICD. Prefer endpoint `diabetes_ocular`.
- Do **not** use `diabetes_related` in clinical-claim tables.
- BRSET **diabetes**, **DR grade / dr_referable**, and **hypertensive retinopathy** are distinct heads.
- RFMiD is a long-tail retinal-sign ontology, not a six-disease oculomics replica.
- Public `UKB_y5.npz` / `UKB_y10.npz` written by `prepare_public_npz.py` are **copies of prevalence** for loader compatibility, not 5/10-year incidence.
- Synthetic exports set `clinical_claim_allowed: false` in `label_map.json` / `SYNTHETIC_FEATURES.txt`.

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

## Endpoint-aware cross-cohort evaluation

Train on one public cache, score **overlapping endpoints only** on another.
Default endpoints: `hypertension_ocular`, `diabetes_ocular` (not `diabetes_related`).
Pass `--clinical-tables` / `--paper-mode` to refuse non-`direct` alignments.

```powershell
python scripts/prepare_public_npz.py --dataset odir --synthetic-demo --out data/odir --force
python scripts/prepare_public_npz.py --dataset brset --synthetic-demo --out data/brset --force
python scripts/train.py --config configs/ablation_e4.yaml --dataset odir --data-dir data/odir --multitask --horizon 0
python scripts/evaluate.py --ckpt ckpt/<run>/multitask/y0 --dataset odir --data-dir data/odir `
  --test-data-dir data/brset --test-dataset brset --clinical-tables
```

| Canonical endpoint | ODIR | BRSET | RFMiD | Alignment |
|--------------------|------|-------|-------|-----------|
| `diabetes_ocular` | `D` | `dr_referable` | `DR` | direct/partial — preferred default |
| `hypertension_ocular` | `H` | `hypertensive_retinopathy` | `HR` if present | direct ocular |
| `diabetes_related` | `D` | `diabetes` | `DR` | related_not_equivalent — exploratory only |

Do not put synthetic-cache AUROCs in a manuscript. Real ODIR→BRSET numbers require downloaded images and backbone features.
