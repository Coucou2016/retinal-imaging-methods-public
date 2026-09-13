# Gap closure acceptance — 2026-09-14

**Trigger:** Completeness audit (`docs/chatgpt-runs/2026-09-14-completeness-audit/AUDIT.md`) verdict **未严格完整落实**.  
**Scope:** Local engineering / narrative gaps only (no ODIR/BRSET pixel downloads, no GPU).  
**Unittest:** `python -m unittest discover -s tests -v` → **60 tests, OK** (~228 s).

---

## Before / after matrix

| # | Item (from audit) | Before | After | Evidence |
|---|-------------------|--------|-------|----------|
| 1 | Real-data Results | 待补充 (honest) | **Unchanged** — still 待补充 | `docs/paper/manuscript.md` §5 |
| 2 | Narrative drift (`build_paper_report.manuscript_md`, PAPER_PLAN NB@0.10) | Stale title / co-primary NB | **Fixed** | `scripts/build_paper_report.py:manuscript_md` loads/aligns current MS; `docs/PAPER_PLAN.md` §5 item 5: NB@0.10 illustrative only |
| 3 | Published vs released framing in builder | Old single-task text in fallback | **Fixed** (same as #2) | Regenerated HTML/PDF via `build_paper_report.py` |
| 9 | Threshold val-only / test frozen | Fit-on-eval leakage | **Fixed** | `fit_operating_point_thresholds` / `apply_operating_point_thresholds`; `resolve_threshold_fit_ids`; `paper_mode` raises on fit-on-eval; `tests/test_gap_closure.py:TestThresholdFitNotEval` |
| 11 | Patient-level bootstrap 95% CI (+ paired Δ) | 未做 | **Done** | `utils/bootstrap.py`; wired into `evaluate.py --bootstrap` / paper_mode default 1000; `--out-json` key `bootstrap_auroc`; unit tests on synthetic labels |
| 14 | CI / Docker / CITATION | 未做 | **Done** | `.github/workflows/ci.yml`, `Dockerfile`, `CITATION.cff`, `NOTICE` → `THIRD_PARTY_NOTICES.md` |
| 15 | E5 quality-conditioned gating | 未做 | **Done** (optional, default off) | `model/quality_gate.py`, `quality_gating` on `FuseBase`/`get_reti_pioneer`; `configs/ablation_e5.yaml` |
| 16 | BRSET aux quality loss λ_q | 未做 | **Done** (hook) | `soft_quality_ce` + `QualityAuxHead`; `training.lambda_q` in `train.py` / custom step in `utils/run.py` |
| 17 | Intervention good→usable→bad | 未做 | **Done** | `utils/intervention.py`, `scripts/intervention_quality.py`, unit smoke test |
| — | Monotone scalar × q order bug | `(good,usable,bad)` q × `(bad,usable,good)` w | **Fixed** | `QualityAware.quality_scalar` reorders; `TestMonotoneScalarOrder` |

Items already **已完整** in the audit (endpoints, cal ⊥ eval, monotone router, ensemble rename, masked BCE, pyproject, license/disclaimer) were left intact.

---

## Remaining blockers (not local)

1. **Fatal for clinical claims:** real ODIR / BRSET / RFMiD pixels + GPU foundation features + filled Results tables (still 待补充; no invented AUROCs).
2. Optional follow-ups: `val_weighted` ensemble (still deferred); richer BRSET quality-label supervision beyond soft-q CE; DeLong asymptotic CI (bootstrap paired Δ is present).

---

## Commands

```text
python -m unittest discover -s tests -v
python scripts/build_paper_report.py
python scripts/intervention_quality.py
```
