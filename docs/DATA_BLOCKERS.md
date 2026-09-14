# Data / compute blockers (honest inventory)

**Date:** 2026-09-15  
**Host evidence:** `nvidia-smi` shows GeForce GTX 950M (driver 460.89, CUDA 11.2 capability advertised).  
**PyTorch:** `torch 2.12.0+cpu` — `torch.cuda.is_available() == False` (CPU wheel only).

| Asset | Attempt | Result | Blocker |
|-------|---------|--------|---------|
| RFMiD images | Local HF mirror under `data/raw/rfmid` | **3200 images + official CSVs present** | Labels prepared; foundation extract needs GPU-class CUDA torch + RETFound/Vim weights for full 3-backbone clinical cache |
| RFMiD labels→UKB cache | `prepare_public_npz.py --labels-only --write-manifest` | **Done** (`data/rfmid`, N=3200, `pairs.csv`) | Backbone npz still SYNTHETIC until real extract finishes |
| RFMiD CPU Swin subset | `extract_features.py --backbone swin --allow-cpu --limit 64` | In progress / environment-limited | CPU torch + 336MB Swin download; RETFound/Vim not extracted on CPU in this pass |
| ODIR-5K | `kaggle datasets download` | **Impossible here** | No `~/.kaggle/kaggle.json`; `kaggle` CLI not installed |
| BRSET | PhysioNet DUA download | **Impossible here** | No PhysioNet credentials / `~/.physionet`; helper never scrapes PhysioNet |
| CUDA foundation extract | Use GTX 950M | **Blocked** | Installed torch is CPU-only; driver 460 / CUDA 11.2 is too old for current CUDA wheels without a custom rebuild |
| Clinical Results tables | Fill manuscript AUROCs | **待补充** | No invented AUROCs; require real multi-backbone features on full cohorts |

## What software still does without clinical pixels

- E0–E5 + MultiCohort joint vocabulary on synthetic / demo caches
- Patient-level bootstrap + DeLong AUROC CI in `evaluate.py --out-json`
- Per-disease DCA as primary utility (`dca_curves`)
- `scripts/predict_extension.py` for extension checkpoints
- Intervention figures (`scripts/intervention_quality.py`)
- CI: unittest + every YAML builds + E5 1-epoch smoke + `run_ablations.py --quick`
