# Completeness audit — peer-review P0/P1 fixes

**Date:** 2026-09-14  
**Method:** Source-of-truth audit (code/docs/tests/git). `ACCEPTANCE.md` used only as a claim list, not evidence.  
**Unittest:** `python -m unittest discover -s tests -v` → **49 tests, OK** (~242 s).  
**Git:** local `HEAD` = `origin/main` = `686823dfbf39863156dbf8cf8e071044130a3c1d` (includes fix commit `f431a00` + ACCEPTANCE push note).

---

## Overall verdict

### **未严格完整落实** (audit snapshot)

> **Follow-up (same day):** local gaps closed in `docs/chatgpt-runs/2026-09-14-gap-closure/ACCEPTANCE.md` (threshold lock, bootstrap CI, narrative, E5/λ_q/intervention, CI/Docker/CITATION). **Clinical Results still 待补充** — engineering/narrative side is now as complete as possible without real data/GPU.

Core P0 *engineering* (monotone router, endpoint ontology, calibration disjointness + `paper_mode`, ensemble rename, masked BCE, honesty banners for Results) is largely present in code and in `docs/paper/manuscript.md`. That is **not** the same as strict, complete closure of the peer-review checklist: real Results remain empty (expected), several P1 method items were never implemented, threshold-selection protocol is incomplete, narrative sources are inconsistent (`build_paper_report.py` still embeds the pre-review title/claims), and CI/Docker/`CITATION.cff` are absent.

If forced to a softer engineering-only phrase: *多数 P0 工程改动已落地，但投稿闭环与部分审稿 P1 仍未齐* — still **not** “严格完整落实.”

---

## Completeness matrix

| # | Item | Status | Evidence (file:symbol / path) | Gap |
|---|------|--------|-------------------------------|-----|
| 1 | Real-data Results incomplete (honest) | **已完整**（诚实标注仍待补充） | `docs/paper/manuscript.md` §5 “Clinical result tables: 待补充”; README claim table; ACCEPTANCE 待补充 | Content still empty — **expected**; blocks submission efficacy claims |
| 2 | Calibration/DCA demoted to evaluation | **部分** | Good: `manuscript.md` Abstract/Intro; README; `docs/PAPER_PLAN.md` §1 IV; `docs/paper/OUTLINE.md`. Bad: `scripts/build_paper_report.py:manuscript_md` still old title *…Learnable Fusion, Calibration, and Public-Cohort Validation* and elevates cal/DCA in contributions; `PAPER_PLAN.md` L142 still “Co-primary … net benefit at 10%” | Dual narrative sources; PAPER_PLAN co-primary NB contradicts demotion |
| 3 | Published multitask vs released independent | **部分** | Good: `manuscript.md`, README “Published vs released”, `RetiPioneer.py` comments, E0 config. Bad: `build_paper_report.py:manuscript_md` still “single-task training” / “independent binary heads” without published-vs-released framing | Rebuild from builder would regress English MS template |
| 4 | Endpoint ontology; no over-broad `diabetes_related`; harmonization | **已完整** | `reti_pioneer/label_map.py:Endpoint`, `ENDPOINTS`, `assert_clinical_alignment`, `harmonization_table_rows`; `docs/PUBLIC_DATA.md` harmonization table; `evaluate.py --clinical-tables/--paper-mode`; tests in `tests/test_review_fixes.py:TestEndpointOntology` | — |
| 5 | Calibration fit ⊥ eval; `paper_mode` guard | **已完整** | `reti_pioneer/split.py:assert_calibration_disjoint`, `patient_level_train_cal_*`, `nested_calibration_from_val`, `split_hash`; `scripts/evaluate.py:_resolve_calibration_ids` raises in `paper_mode`; `scripts/train.py` persists `calibration_idx` | Nested-from-val still warns (OK); prefer explicit `calibration_idx` in paper runs |
| 6 | Monotone bounded quality router default | **已完整** | `model/QualityAware.py:QualityAware` (`monotone` / `monotone_weights`); `RetiPioneer.get_reti_pioneer` learnable→monotone; configs `quality_router: monotone`; tests `TestMonotoneQualityRouter` | — |
| 7 | Ensemble rename (`released_code`, `published_soft_vote`, …) | **已完整** | `model/RetiPioneer.py:normalize_ensemble`, `VALID_ENSEMBLES`; CLI choices in `train.py`/`evaluate.py`; DeprecationWarning for `"paper"`; tests `TestEnsembleNaming` | `val_weighted` still deferred (ACCEPTANCE) |
| 8 | Macro NB@0.10 not primary; per-disease DCA | **部分** | `utils/calibration.py:per_class_decision_curves`; `evaluate.py` emits `dca_curves` + illustrative note for `nb@0.10`; manuscript narrative OK | `PAPER_PLAN.md` L142 still co-primary NB@10%; ablations summary still columns NB@0.10 |
| 9 | Threshold on val only / test frozen | **部分** | Frozen test: `split.py` 3-/4-way splits; `train.py` copies `test_idx`; `test_followup.TestFrozenSplitAndDCA.test_train_copies_frozen_test_idx` | **No** val→test threshold lock: `evaluate._merge_operating_points` fits Youden / sens@95%spec **on the scored split itself** (test leakage risk if used as selected operating points) |
| 10 | Partial-label masked BCE multitask | **已完整** | `utils/run.py:masked_bce_with_logits`, `MaskedBCEWithLogitsLoss`; `train.py` `masked_bce`; E0–E4 configs; `TestMaskedBCE` | Dataset-level mask exposure is via target `<0`/NaN convention (adequate) |
| 11 | Patient-level bootstrap CI + paired comparison | **未做** | No `bootstrap` / DeLong / paired CI implementation under `scripts/` or `utils/`; manuscript/ACCEPTANCE mark 待补充 | Must implement for claimed multi-seed inference reporting |
| 12 | `pyproject.toml` no local wheel paths | **已完整** | `pyproject.toml` uses PyPI extras + `tool.uv.index` pytorch URL; comment explicitly forbids `../../wheel` | — |
| 13 | License vs research disclaimer; NOTICE | **已完整** | `LICENSE` (MIT only); `README.md` §License vs research disclaimer; `THIRD_PARTY_NOTICES.md` (stands in for NOTICE) | No file named `NOTICE` — content present under THIRD_PARTY_NOTICES |
| 14 | CI / Docker / CITATION | **未做** | No `.github/workflows`, no `Dockerfile`, no `CITATION.cff`; README has BibTeX only | P2 in original review; still open for “投稿前工程完备” |
| 15 | Quality-conditioned backbone gating E5 | **未做** | `configs/ablation_e4.yaml` comment “E5 … 待补充”; PAPER_PLAN / manuscript / README mark E5 待补充; no gating MLP in `RetiPioneer.py` | Explicitly deferred |
| 16 | Auxiliary quality loss on BRSET | **未做** | No `quality_loss` / `lambda_q` / aux head in `utils/run.py` or `dataset/brset.py` training path | Review §四 recommendation never implemented |
| 17 | Intervention experiment good→usable→bad | **未做** | No intervention / counterfactual quality-vector script or test | Review §四 never implemented |
| 18 | Unit tests listed in review | **已完整** | `tests/test_review_fixes.py`: disjoint cal, monotone+grad, ensemble rename, endpoints, masked BCE, default monotone; suite **49 OK** | Does not cover bootstrap, E5, intervention, threshold lock |

---

## ACCEPTANCE.md vs audit (overclaim check)

| ACCEPTANCE claim | Audit |
|------------------|-------|
| Paper narrative rewrite **Done** | `manuscript.md` / README / OUTLINE yes; **`build_paper_report.manuscript_md` still pre-review** |
| Unit tests → 49 OK | **Confirmed** this audit |
| E5 / bootstrap deferred | **Confirmed still missing** |
| Engineering hygiene Done | pyproject + NOTICE + disclaimer **yes**; CI/Docker/CITATION **no** (ACCEPTANCE did not claim those) |

---

## Remaining blockers (severity order)

1. **Fatal — real ODIR/BRSET/RFMiD pixels + GPU foundation features + filled Results tables** (still 待补充; no efficacy claim possible).
2. **High — threshold selection protocol**: select operating points on val only; freeze and apply to test (currently fit-on-eval).
3. **High — narrative consistency**: refresh `scripts/build_paper_report.py:manuscript_md` (and any regenerated English MS) to match `docs/paper/manuscript.md`; fix `PAPER_PLAN.md` co-primary NB@0.10.
4. **High — patient-level bootstrap 95% CI + paired comparison** (DeLong / paired bootstrap) — listed P1, still absent.
5. **Medium — E5 quality-conditioned backbone gating** (review “stronger methods” path; still stub).
6. **Medium — BRSET auxiliary quality loss + good→usable→bad intervention** (validity of learnable_q).
7. **Low/P2 — CI workflow, Dockerfile, `CITATION.cff`**.

---

## Commands / refs

```text
python -m unittest discover -s tests -v   # 49 OK
git rev-parse HEAD origin/main            # 686823d… identical
```

Primary claim artifact (not trusted alone): `docs/chatgpt-runs/2026-09-13-review-fixes/ACCEPTANCE.md`
