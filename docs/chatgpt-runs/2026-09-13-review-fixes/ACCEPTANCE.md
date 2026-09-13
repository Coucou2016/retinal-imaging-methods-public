# ACCEPTANCE — 2026-09-13 review fixes

Peer-review (Reject & Resubmit) P0/P1 engineering + narrative fixes.
No fabricated clinical AUROCs.

## Fixed (this run)

| Item | Status | Notes |
|------|--------|-------|
| 1. Paper narrative rewrite | **Done** | Title → monotone routing + endpoint-aware MTL; calibration/DCA = evaluation framework; published multitask vs released independent loops; Endpoint-Aware Cross-Cohort Evaluation |
| Manuscript HTML/PDF rebuild | **Done** | `scripts/build_paper_report.py` + `plot_paper_figures.py` |
| Report honesty banners | **Done** | `docs/report/report.*` |
| README / PAPER_PLAN / OUTLINE | **Done** | Claims aligned |
| 2. Endpoint ontology | **Done** | `Endpoint` + alignment flags; `diabetes_related` exploratory only; harmonization in PUBLIC_DATA + label_map |
| Cross-eval clinical guard | **Done** | `--clinical-tables` / `--paper-mode` refuse non-direct |
| 3. Calibration leakage | **Done** | `calibration_idx` disjoint; nested val warn; `paper_mode` error; split hashes |
| 4. Monotone quality router | **Done** | Default learnable path; fixed baseline; free_linear ablation |
| 5. Ensemble naming | **Done** | `released_code` (+ `paper` DeprecationWarning); `published_soft_vote`, `mean`, `temp_mean` |
| 6. Partial-label multitask | **Done** | Masked BCE; E0–E4 configs |
| 7. DCA / metrics | **Done** | Per-disease `dca_curves` in evaluate JSON; NB@0.10 illustrative note; intercept/slope helpers |
| 8. Engineering hygiene | **Done** | Removed local `../../wheel` paths; extras `[core]`/`[cuda]`/`[dev]`; `THIRD_PARTY_NOTICES.md`; README disclaimer vs MIT |
| Synthetic `clinical_claim_allowed: false` | **Done** | `label_map.json` + SYNTHETIC marker |
| Unit tests | **Done** | `tests/test_review_fixes.py` + updated follow-up; `python -m unittest discover -s tests -v` → **49 OK** |
| 9. GitHub push | See below | code+docs only |

## 待补充 (intentionally deferred / blocked)

| Item | Reason |
|------|--------|
| Real ODIR/BRSET pixels + GPU foundation extract | No Kaggle/PhysioNet creds; CUDA unavailable on this host |
| Clinical AUROC / ECE / DCA tables | Require real features — **not invented** |
| E5 quality-conditioned gating | Time-boxed; stubbed as 待补充 |
| Patient-level bootstrap CIs | Minimal/stub deferred |
| Live ChatGPT multi-round critique | Optional; browser may be blocked — local engineering prioritized |
| Val-weighted ensemble | Not required for K=1; deferred |

## Commands run

```powershell
python -m unittest discover -s tests -v   # 49 OK
python scripts/plot_paper_figures.py
python scripts/build_paper_report.py
```

## Public snapshot push

Target: https://github.com/Coucou2016/retinal-imaging-methods-public  
Policy: **code + docs only** (no `data/`, `ckpt/`, secrets, real images).

| Attempt | Result |
|---------|--------|
| Local commit | **Done** `f431a000b10d3f60cf7afbd3fde716a51f60ca79` — *Fix P0 review: monotone router, endpoint ontology, calibration split.* |
| `git push origin HEAD` | **Done** (retry after initial 443 timeout) `5e967a9..f431a00` → `main` |
