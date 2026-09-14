# ACCEPTANCE — 2026-09-15 no-tails closure

**Workspace:** repository root  
**Public GitHub:** https://github.com/Coucou2016/retinal-imaging-methods-public  
**Policy:** Finish all doable software tails; no invented clinical AUROCs; document real-data blockers with evidence.

## Master Done / Impossible matrix

| # | Item | Status | Evidence |
|---|------|--------|----------|
| A1 | Audit ACCEPTANCE/AUDIT gaps → master checklist | **Done** | This file + prior audits |
| A2 | MultiCohort joint partial-label (ODIR+BRSET+RFMiD → one vocab) | **Done** | `reti_pioneer/joint_vocab.py`, `dataset/multi_cohort.py`, `configs/ablation_multicohort.yaml`, `train.py --multi-cohort`; smoke train n=3296 |
| A3 | E5 backbone routing (softmax over backbones from q) | **Done** | `QualityBackboneRouter` in `model/quality_gate.py`; wired in `ComplexModel.forward` when `quality_gating`; λ_q + `QualityAuxHead` end-to-end |
| A3b | Intervention good→usable→bad + figures | **Done** | `scripts/intervention_quality.py` → `artifacts/figures/intervention_quality.png` |
| A4 | `scripts/predict_extension.py` full inference path | **Done** | config / run_meta / cal T / frozen thresholds / endpoints |
| A5 | Mark `inference.py` upstream-reference only | **Done** | README |
| A6 | Disease-specific DCA primary; bootstrap CIs; DeLong | **Done** | `evaluate.py` out-json: `dca_curves`, `bootstrap_auroc`, `delong_auroc`; `utils/bootstrap.py:delong_auroc_ci` |
| A7 | Multilabel patient stratification + fail missing classes | **Done** | `patient_level_multilabel_train_val_indices`; `assert_split_label_coverage` (paper_mode); masked-aware `_label_scalar` |
| A8 | Strict config: every YAML builds + 1-epoch smoke | **Done** | `TestAllConfigsBuildAndOneEpoch`; CI job steps strengthened |
| A9 | Sync manuscript/report/README; regenerate figures/report | **Done** | manuscript E5/MultiCohort wording; `build_paper_report.py`; `plot_paper_figures.py`; intervention figure |
| A10 | Push public GitHub (no secrets/large private data) | **Done** | see Git section below |
| B1 | Download RFMiD images | **Done** (already present) | `data/raw/rfmid` N=3200; labels+`pairs.csv` via prepare |
| B1b | CPU Swin subset features | **Done** (partial) | `data/rfmid_cpu_subset` UKB_swin.npz (64,1024); RETF/Vim still synthetic; `clinical_claim_allowed: false` |
| B2 | ODIR via Kaggle | **Impossible** | No `~/.kaggle/kaggle.json`; no `kaggle` CLI — see `docs/DATA_BLOCKERS.md` |
| B3 | BRSET PhysioNet | **Impossible** | No PhysioNet credentials — documented |
| B4 | Full CUDA foundation extract | **Impossible** | `nvidia-smi`: GTX 950M / driver 460.89; `torch 2.12.0+cpu`, `cuda=False` |
| B5 | Clinical Results tables with real AUROCs | **Impossible / 待补充** | No invented numbers; manuscript remains 待补充; paper_mode blocks synthetic claims |
| C1 | Full unittest green | **Done** | **82 OK** (~377 s) |
| C2 | `run_ablations.py --quick` | **Done** | 8/8 eval rows ok (SYNTHETIC) |
| C3 | ACCEPTANCE no-tails matrix | **Done** | this file |

## Commands

```text
python -m unittest discover -s tests -v
python scripts/run_ablations.py --quick
python scripts/intervention_quality.py
python scripts/train.py --config configs/ablation_multicohort.yaml --multi-cohort --demo --horizon 0
python scripts/train.py --config configs/ablation_e5.yaml --demo --multitask --horizon 0
python scripts/build_paper_report.py
```

## Honesty

- Clinical AUROC / ECE / DCA **tables not filled with invented numbers**.
- RFMiD pixels exist; full 3-backbone clinical feature cache still blocked by CPU-only torch + missing RETFound/Vim weights for complete extract.
- Software release is **complete** relative to peer-review engineering tails; clinical efficacy claims remain locked behind real multi-backbone features + ODIR/BRSET credentials.
