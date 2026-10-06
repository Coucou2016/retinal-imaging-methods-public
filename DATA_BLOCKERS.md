# Data / compute blockers (honest inventory)

**Date:** 2026-09-17  
**Host evidence:** `nvidia-smi` shows GeForce GTX 950M (driver 460.89, CUDA 11.2).  
**PyTorch:** `torch 2.12.0+cpu` — `torch.cuda.is_available() == False` (CPU wheel only). Maxwell sm_50 is unsupported by current CUDA PyTorch binaries.

| Asset | Attempt | Result | Blocker |
|-------|---------|--------|---------|
| RFMiD images | Local HF mirror under `data/raw/rfmid` | **3200 images + official CSVs present** | None for pixels |
| RFMiD real-pixel features | `build_real_rfmid_cache.py` (CPU) | **Done** — Swin-V2-B + ViT-B→1024 + ViT-S/384; quality proxy; 8-label subset; `SYNTHETIC_FEATURES.txt` removed | RETFound HF gated (401 without token); Vision Mamba / mamba-ssm unavailable on Windows → ImageNet surrogates documented in `FEATURE_PROVENANCE.json` |
| ODIR-5K | `kaggle datasets download` | **Impossible here** | No `~/.kaggle/kaggle.json`; `kaggle` CLI not installed |
| BRSET | PhysioNet DUA download | **Impossible here** | No PhysioNet credentials; helper never scrapes PhysioNet |
| CUDA foundation extract | Use GTX 950M | **Blocked** | CPU-only torch; driver/CUDA too old for modern wheels; Maxwell dropped |
| Clinical UKB tables | Fill manuscript UKB AUROCs | **待补充** | No UKB access; no invented AUROCs |

## What is reportable now

- Real-pixel RFMiD Results (ImageNet foundation surrogates) under ``
- Intuitive figures (fundus quality strata, ROC, reliability, DCA, ablation bars, routing)
- ODIR/BRSET remain **待补充** (no credentials) — do **not** put prior SYNTHETIC AUROCs in main Results tables
