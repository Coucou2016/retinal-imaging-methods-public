# Data preparation

## Official compressed format (training)

Place files under `data/UKBCompressed/` (or set `data_dir` in `configs/default.yaml`):

| File | Contents |
|------|----------|
| `UKB_RETF.npz` | `left`, `right`: N × 1024 RETFound features |
| `UKB_swin.npz` | N × 1024 Swin-B features |
| `UKB_vim.npz` | N × 384 Vision Mamba-S features |
| `UKB_mqd.npz` | `m`, `mn`, `ql`, `qr`, `center` |
| `UKB_y0.npz` / `UKB_y5.npz` / `UKB_y10.npz` | `y`: N × 6 binary labels |

Generate compatible **demo** tensors (skipped if files already exist unless `--force`):

```bash
python scripts/generate_demo_data.py --out-dir data/UKBCompressed --n-samples 256 --seed 42
```

Train/val split metadata is written to `ckpt/<run>/<disease>/y<h>/split.npz` when `split.val_fraction > 0` in config.

## UK Biobank (paper primary cohort)

1. Apply for access: [ukbiobank.ac.uk](https://www.ukbiobank.ac.uk/)
2. Preprocess macula CFPs: center **1400×1400**, resize **224×224** (see `dataset/transforms.py`, `source=ukb`).
3. Run `scripts/extract_features.py` per backbone or use authors' preprocessing scripts from [Reti-Pioneer](https://github.com/lyhyl/Reti-Pioneer).

## Public fundus datasets (pipeline testing only)

These do **not** provide the six endocrine/metabolic labels from the paper. ODIR / BRSET / RFMiD are used for the follow-up methods paper with **explicit label mismatch**; see [docs/PUBLIC_DATA.md](PUBLIC_DATA.md).

### ODIR-5K / BRSET / RFMiD

- Loaders: `dataset/odir.py`, `dataset/brset.py`, `dataset/rfmid.py`
- Cache builder: `scripts/prepare_public_npz.py` (synthetic features if images are absent)
- Patient-level split: `reti_pioneer.split.patient_level_train_val_indices`

### APTOS 2019

- Kaggle: [aptos2019-blindness-detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
- Layout: `train.csv`, `train_images/*.png`
- Loader: `dataset.public_fundus.AptosCsvDataset`

### MESSIDOR

- [messidor.crihan.fr](https://www.adcis.net/en/third-party/messidor/)
- Layout: `images/`, optional `labels.csv` with `image,grade`
- Loader: `dataset.public_fundus.MessidorIndexDataset`

### EyePACS

- Request access via [eyepacs.com](https://www.eyepacs.com/) (DR grading; not systemic disease labels)
- Use same preprocessing as hospital cohort (`source=hospital`: pad-to-square, 224×224)

## Pretrained weights (inference)

| Model | Source |
|-------|--------|
| RETFound | Hugging Face `YukunZhou/RETFound_mae_natureCFP` (auto-download) |
| Swin V2-B | `torchvision` weights |
| EyeQ DenseNet | [EyeQ repo](https://github.com/HzFu/EyeQ) → set `RETI_PIONEER_EYEQ_WEIGHTS` |
| Vim-S | Set `RETI_PIONEER_VIM_WEIGHTS` under `pretrained/` |

```powershell
$env:RETI_PIONEER_PRETRAINED_DIR = "E:\Projects\20260522-retinal-imaging\pretrained"
$env:RETI_PIONEER_EYEQ_WEIGHTS = "$env:RETI_PIONEER_PRETRAINED_DIR\fundus\DenseNet121_v3_v1.tar"
python scripts/setup_pretrained.py
```

`setup_pretrained.py` reports missing EyeQ/Vim paths and whether RETFound is cached on Hugging Face. Fast-mode training on `UKB_*.npz` does not require these files.
