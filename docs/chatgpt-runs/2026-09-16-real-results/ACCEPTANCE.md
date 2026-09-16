# ACCEPTANCE — 2026-09-16 real-results

**Date:** 2026-09-17  
**Workspace:** `E:\Projects\20260522-retinal-imaging`  
**Public:** https://github.com/Coucou2016/retinal-imaging-methods-public  
**Demand focus:** intuitive Results / data / figures first; remove SYNTHETIC from main Results; fill feasible 待补充 with real RFMiD.

---

## Checklist

| Item | Status | Evidence |
|------|--------|----------|
| RFMiD N=3200 real-pixel feature extract | **DONE** | `scripts/build_real_rfmid_cache.py`; `data/rfmid/FEATURE_PROVENANCE.json`; `SYNTHETIC_FEATURES.txt` removed |
| Quality soft-q from pixels | **DONE** | strata good/usable/bad = 179/2793/228 |
| Backbone honesty | **DONE** | Swin-V2-B real; RETFound gated → ViT-B→1024; Vim unavailable → ViT-S/384 |
| ODIR / BRSET | **待补充** | No Kaggle / PhysioNet credentials (`docs/DATA_BLOCKERS.md`) |
| Train E0–E5 on real cache | **DONE** | `results/real_rfmid/ckpts/*`; 12 epochs; official split + nested cal |
| Eval calibrate on cal, metrics on test + bootstrap CI | **DONE** | `metrics_E{0-5}_rfmid_{val,test}.json` |
| Forbidden SYNTHETIC AUROCs in main Results | **DONE** | Manuscript §5 replaced; legacy SYNTHETIC only in old CI paths |
| Intuitive figures before AUROC tables | **DONE** | fig5 fundus, fig6 routing/pie, fig8 ROC, fig3 reliability, fig7 DCA, fig2 bars, fig9 hist |
| Embed figures + rebuild report.html | **DONE** | `docs/paper_assets/embedded_png_uris.json`; `docs/report/report.html` |
| AUTHENTICITY_AUDIT refreshed | **DONE** | `docs/paper/AUTHENTICITY_AUDIT.md` |
| ACCEPTANCE this path | **DONE** | this file |
| Push to GitHub | **DONE** | see PUSH note / `git log -1` after push |

---

## Key real test numbers (not invented)

| Arm | Macro AUROC | AUROC_DR | Bootstrap 95% CI | n |
|-----|-------------|----------|------------------|---|
| E0 | 0.925 | 0.925 | [0.902, 0.947] | 640 |
| E1 | 0.931 | 0.931 | [0.908, 0.949] | 640 |
| E2 | 0.886 | 0.910 | [0.872, 0.902] | 640 |
| E3 | 0.898 | 0.928 | [0.882, 0.914] | 640 |
| E4 | 0.898 | 0.928 | [0.882, 0.914] | 640 |
| E5 | 0.901 | 0.916 | [0.887, 0.914] | 640 |

Source: `results/real_rfmid/ablation_summary.csv`.

---

## Commands (repro)

```text
python scripts/build_real_rfmid_cache.py --cache-dir data/rfmid --batch-size 8
python scripts/run_real_rfmid_experiments.py --out-dir results/real_rfmid --epochs 12 --bootstrap 200
python scripts/plot_real_results.py
python scripts/build_paper_report.py
```

---

## Remaining 待补充 (honest)

- ODIR (Kaggle) / BRSET (PhysioNet) downloads  
- RETFound / Vim-S exact weights  
- UKB systemic clinical tables  
- Multi-seed {42,43,44}  
- CUDA torch on this host (GTX 950M / Maxwell unsupported by current wheels)
