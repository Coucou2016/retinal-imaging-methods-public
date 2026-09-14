# ACCEPTANCE — 2026-09-15 no-tails (software-complete)

**Public:** https://github.com/Coucou2016/retinal-imaging-methods-public  
**Detail matrix:** `docs/chatgpt-runs/2026-09-15-no-tails/ACCEPTANCE.md`  
**Data blockers:** `docs/DATA_BLOCKERS.md`

## Software (Done)

| Item | Status |
|------|--------|
| E5 quality-conditioned **backbone routing** + λ_q + intervention figures | **Done** |
| MultiCohort joint vocabulary (ODIR+BRSET+RFMiD masks) | **Done** |
| `scripts/predict_extension.py` | **Done** |
| `inference.py` marked upstream-reference only | **Done** |
| Per-disease DCA + bootstrap CI + DeLong in evaluate JSON | **Done** |
| Multilabel stratification + paper_mode class-coverage fail | **Done** |
| Strict YAML build + CI 1-epoch E5 smoke | **Done** |
| Manuscript / README / report sync | **Done** |
| Unittest **82 OK**; `run_ablations.py --quick` 8/8 | **Done** |

## Real data

| Item | Status |
|------|--------|
| RFMiD images + labels + pairs.csv (N=3200) | **Done** |
| CPU Swin subset features (N=64, partial) | **Done** (not clinical 3-backbone) |
| ODIR (Kaggle) | **Impossible** — no credentials |
| BRSET (PhysioNet) | **Impossible** — no credentials |
| CUDA foundation extract | **Impossible** — torch CPU-only despite GTX 950M |
| Clinical Results AUROCs | **待补充** — not invented |

## Unittest

```text
python -m unittest discover -s tests -v
# Ran 82 tests — OK
```
