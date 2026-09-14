# ACCEPTANCE — 2026-09-15 contract sync (YAML→code)

**Workspace:** repository root (path machine-local)  
**Public GitHub:** https://github.com/Coucou2016/retinal-imaging-methods-public  
**Date:** 2026-09-15

## Mission

Close P0 docs/YAML ↔ code contract gaps so reviewer claims about monotone router, ensembles, cal splits, endpoint ontology, and E5 gating match **actual** `main`. Prefer implementation sync over new paper prose. Do **not** invent clinical AUROCs.

## Checklist

| Item | Status | Evidence |
|------|--------|----------|
| Audit local vs remote HEAD | **Done** | `docs/chatgpt-runs/2026-09-15-contract-sync/AUDIT.md` |
| Strict config schema (unknown keys fail) | **Done** | `reti_pioneer/config.py:validate_config` |
| Every ablation YAML builds a model | **Done** | `build_model_from_config` + `TestStrictConfigSchema` |
| Monotone router `fixed\|free_linear\|monotone` | **Done** (already on remote; reconfirmed) | `QualityAware.py` |
| Ensemble `released_code` / `published_soft_vote` / `mean` / `temp_mean`; `paper` deprecated | **Done** | `RetiPioneer.normalize_ensemble` |
| train/val/cal/test + cal ⊥ eval; paper_mode no val-fit+score | **Done** | `split.py` / `evaluate.resolve_calibration_fit_ids` |
| `EndpointSpec` + diabetes/hypertension ocular vs systemic | **Done** | `label_map.py` |
| paper_mode fail-closed on non-direct **and** synthetic features | **Done** | `assert_clinical_alignment` + `features_clinical_claim_allowed` |
| `masked_bce` / `lambda_q` / `quality_gating` wired | **Done** | `train.py` / `RetiPioneer` / `ablation_e5.yaml` |
| Provenance: `clinical_claim_allowed` only after real features | **Done** | `update_label_map_provenance`; extract updates JSON |
| `prepare_public_npz` never destroys official test | **Done** | carve val from train only |
| README no local Windows path | **Done** | `cd <repo-root>` |
| `pyproject.toml` no local wheel paths | **Done** (confirmed) | PyPI + pytorch index only |
| Hard-gate tests | **Done** | `tests/test_contract_sync.py` |
| Full unittest | **Done** | **73 OK** |
| Clinical Results AUROC | **待补充** | No Kaggle/PhysioNet/CUDA — not invented |

## Unittest

```text
python -m unittest discover -s tests -v
# Ran 73 tests in ~279 s — OK
```

## Data honesty

| Asset | Status |
|-------|--------|
| ODIR | Not downloaded (no Kaggle creds) |
| BRSET | Blocked (no PhysioNet creds) |
| RFMiD labels | Real CSVs where present; backbone features may be SYNTHETIC |
| Foundation extract | 待补充 (no CUDA) |
| Manuscript clinical tables | 待补充 |

## Git / public snapshot

**Pushed tip:** `69ef661a144aa18fa1d8f2c0cad19a3ab01ddd88` on `main`.  
Contract-sync code commit: `cc70b8d6`. Verify: https://github.com/Coucou2016/retinal-imaging-methods-public/commit/69ef661

Policy: code + docs only; no secrets, patient images, or large npz/ckpt.
