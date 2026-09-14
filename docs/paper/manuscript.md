# Extending Reti-Pioneer with Monotone Quality Routing and Endpoint-Aware Multi-Task Learning

**Manuscript draft (methods paper)** · 2026-09-13 (review-fixes)  
**Axes:** task=manuscript · paper_type=methods · language=en · journal=generic (Nature-family structure; not a flagship *Nature Medicine* clinical claim)  
**Realistic venues without UK Biobank:** *npj Digital Medicine*, *Medical Image Analysis*, *IEEE JBHI / TMI*, *Computers in Biology and Medicine*

> **One-sentence argument.** Reti-Pioneer showed that frozen foundation features plus quality-aware fusion can screen systemic disease from color fundus photographs (CFPs); we keep that skeleton, replace fixed quality weights with a **monotone bounded quality router**, replace independent binary heads in the **released training code** with **partial-label multi-task learning**, and evaluate with an **endpoint-aware cross-cohort** protocol plus a calibration/decision-curve **evaluation framework** (not claimed as methodological novelty) on public cohorts reviewers can download — without inventing UK Biobank-comparable AUROCs from synthetic demo caches.

---

## Abstract

Endocrine and metabolic disease screening from retinal imaging (oculomics) remains limited by image-quality heterogeneity, single-task training loops, and poorly calibrated risk scores. Reti-Pioneer (Zhang et al., *Nature Medicine*, 2026) demonstrated that frozen vision foundation models with quality-aware bilinear fusion can screen six systemic conditions on UK Biobank (UKB) and hospital cohorts. The **published** article describes a multitask screening framework; the **released training code** (`main.py`) trains **per-disease independent binary loops**. Quality routing is fixed at good=1 / usable=0.5 / bad=0. Here we propose a methods extension that (i) learns **monotone bounded** quality weights (bad ≤ usable ≤ good in [0,1]) as the default learnable path, (ii) shares a multi-label head with **masked BCE** over labels present for each sample/dataset, and (iii) reports expected calibration error (ECE), Brier score, and **per-disease decision-curve analysis (DCA)** under temperature scaling fitted on a fold **disjoint** from evaluation. We specify patient-level **endpoint-aware cross-cohort evaluation** on Ocular Disease Intelligent Recognition (ODIR-5K), Brazilian Multilabel Ophthalmological Dataset (BRSET), and Retinal Fundus Multi-disease Image Dataset (RFMiD), with explicit alignment flags (`direct` / `partial` / `related_not_equivalent`).

**Honesty banner.** Clinical AUROC / ECE / decision-curve tables are marked **待补充** until real images and foundation features replace synthetic caches. Pipeline sanity metrics on synthetic features appear only in the companion research report and are **not** manuscript claims. Calibration and DCA are an **evaluation framework**, not novelty claims. We do not invent UKB or hospital AUROCs.

**Keywords:** oculomics; fundus photography; monotone quality routing; multi-task learning; calibration (evaluation); endpoint harmonization; Reti-Pioneer

---

## 1. Introduction

Color fundus photography is widely available in community eye care and can encode microvascular signatures of systemic disease. Foundation models such as RETFound (Zhou et al., *Nature*, 2023) provide transferable retinal representations, while quality assessment resources such as EyeQ / MCF-Net (Fu et al., MICCAI 2019) support handling of imperfect acquisitions. Reti-Pioneer integrated frozen RETFound, Swin Transformer V2-B, and Vision Mamba-S with quality-aware fusion and metadata, reporting internal-test AUROCs of 0.699–0.833 across six diseases and demonstrating a primary-care silent trial.

Despite this progress, three methodological gaps remain actionable without UKB access. First, Reti-Pioneer freezes quality weights to good=1 / usable=0.5 / bad=0; unconstrained learnable linear weights can violate clinical orderings, so we constrain the learnable path to be **monotone and bounded**. Second, although the Reti-Pioneer **publication** describes multitask screening, the **released codebase trains independent binary heads in per-disease loops**; a shared multi-label head with masked supervision better matches multi-label public ocular datasets. Third, screening deployment needs calibrated probabilities and decision-curve net benefit. We treat temperature scaling, ECE, Brier, and DCA as a **standard evaluation framework** rather than as the paper’s methodological novelty. Population-independent multi-disease studies further show that single-site AUROC overstates transportability when cameras, taxonomies, and populations shift (e.g., Quellec et al., *Scientific Reports*, 2023).

**Contributions.**

1. A Reti-Pioneer-compatible **monotone bounded quality router** (default learnable path), with fixed 1/0.5/0 retained as the released-code baseline and unconstrained free-linear routing as an ablation only.
2. A **shared multi-task head** with **masked BCE** (supervise only labels present for a sample/dataset), with per-disease independent training retained as the released-code control.
3. An **endpoint ontology** with systemic vs ocular_manifestation kinds and alignment flags; **endpoint-aware cross-cohort evaluation** that refuses clinical tables when alignment ≠ `direct`.
4. An open engineering stack (configs E0–E4, ablation runner, public loaders, disjoint calibration splits) enabling independent reproduction once images are downloaded.

**Boundary.** We do not claim UKB type 2 diabetes mellitus (T2DM) AUROC 0.833 replication; we do not treat synthetic feature-cache AUROCs as clinical evidence; gout / osteoporosis / hyperlipidemia / thyroid remain UKB-only until public gold standards exist. ODIR diabetes (D) and BRSET diabetes labels are **not** UKB ICD endpoints. Over-broad `diabetes_related` mappings are **exploratory only** (`related_not_equivalent`).

---

## 2. Related work

**Foundation models for CFP.** RETFound and related retinal foundation models reduce labeled-data needs for ocular and oculomic tasks. Reti-Pioneer ensembles multiple frozen backbones rather than proposing a new foundation model; our work inherits that stance.

**Quality-aware analysis.** EyeQ-style classifiers produce good/usable/reject (bad) probabilities. Reti-Pioneer’s bilinear quality fusion is a strong baseline. Monotone routing is a **bounded refinement of the fusion weights**, not a new quality taxonomy.

**Multi-label fundus datasets and domain shift.** ODIR-5K, RFMiD, and BRSET provide multi-label ocular (and some systemic) annotations downloadable by reviewers. They are **not** UKB ICD endpoints. We keep Reti-Pioneer’s frozen-ensemble skeleton and change routing, head sharing, and endpoint-aware transfer reporting.

**Calibration and decision curves (evaluation framework).** Temperature scaling (Guo et al., ICML 2017), ECE, and DCA (Vickers & Elkin, 2006) are standard clinical-utility assessments. We adopt them as **reporting protocol**, not as claimed algorithmic novelty.

**Positioning.** This is a **methods paper on top of Reti-Pioneer**, not a competitor foundation model and not a *Nature Medicine*–scale clinical claim without biobank data.

---

## 3. Methods

### 3.1 Inputs

Paired or single CFPs, metadata (age, sex; weight/ethnicity padded when missing), and a 3-way quality probability vector. Patient IDs define folds so both eyes of one person stay in one split (`reti_pioneer.split`). RFMiD prefers official train/val/test CSVs when present. When `val_fraction` is set, we support **train / calibration / val / test** so temperature fitting never reuses the evaluation fold.

### 3.2 Frozen backbones and feature cache

Following Reti-Pioneer, we freeze RETFound / Swin V2-B / Vision Mamba-S and train on pre-extracted features (`fast_mode`). Public caches are built via `prepare_public_npz.py` then `extract_features.py`. Until real pixels and CUDA extraction succeed, caches may contain deterministic **synthetic** features flagged by `SYNTHETIC_FEATURES.txt` with `clinical_claim_allowed: false` — usable for continuous integration (CI) only.

### 3.3 Monotone bounded quality routing

`QualityAware` implements bilinear fusion with `quality_router ∈ {fixed, free_linear, monotone}`:

| Mode | Behavior |
|------|----------|
| `fixed` | Frozen weights good=1 / usable=0.5 / bad=0 (released-code baseline) |
| `monotone` | Learnable increments with softplus cumsum; enforces bad ≤ usable ≤ good ∈ [0,1], good=1 (default when `learnable_q=True`) |
| `free_linear` | Unconstrained `Linear(3→1)` (**ablation only**) |

### 3.4 Partial-label multi-task head

`ComplexModel` supports `num_classes=K`. **Masked BCE** supervises only targets with label present (`target ≥ 0`); missing entries (`-1` / NaN) are ignored. **Released-code control** = independent single-disease training loops as in upstream `main.py` (not a straw-man against the published “multitask” wording). Ensemble modes: `released_code` (train soft / eval max), `published_soft_vote`, `mean`, `temp_mean`. Legacy `ensemble="paper"` aliases to `released_code` with a deprecation warning.

### 3.5 Calibration and decision utility (evaluation framework)

Temperature *T* is fit on **calibration_ids** disjoint from **evaluation_ids**. If `--split val --calibrate` would fit and score the same set, `paper_mode` errors; otherwise we nest a held-out calibration slice and warn. **Operating points** (Youden / sens@95%spec) are selected on validation (or calibration) only and applied **frozen** on the evaluation fold; `paper_mode` refuses fit-on-eval. We report AUROC / average precision (AP) with optional **patient-level bootstrap 95% CI**, equal-width ECE, Brier score, optional calibration intercept/slope, sensitivity at high specificity, **per-disease DCA curves**, and (illustrative only) net benefit at threshold 0.10 versus treat-all / treat-none.

### 3.6 Training protocol

| Item | Setting |
|------|---------|
| Learning rate | 1×10⁻⁴ (config default) |
| Backbones | Frozen; train fusion + head on caches |
| Split | Patient-level; train/cal/val/(test); both eyes co-located |
| Seeds (final tables) | {42, 43, 44} — **待补充** on real data |
| Ablation arms | E0–E5 (`configs/ablation_e*.yaml`); E5 = quality-conditioned **backbone routing** (softmax over foundation heads from q) + optional BRSET `lambda_q` |
| Software (this workspace) | PyTorch 2.x CPU build; CUDA **unavailable** → foundation extraction blocked |

### 3.7 Endpoint ontology (pre-registered)

| Endpoint | Kind | ODIR | BRSET | Alignment notes |
|----------|------|------|-------|-----------------|
| `diabetes_ocular` | ocular_manifestation | D | `dr_referable` | direct/partial — not UKB T2DM |
| `diabetes_systemic` | systemic | — | `diabetes` | direct on BRSET; still not UKB ICD |
| `diabetes_related` | systemic (exploratory) | D | `diabetes` | **related_not_equivalent** — refuse clinical tables |
| `hypertension_ocular` | ocular_manifestation | H | hypertensive retinopathy | direct ocular; ≠ systemic HTN ICD |

See `docs/PUBLIC_DATA.md` and `reti_pioneer/label_map.py` for the full harmonization table. Cross-eval with `--clinical-tables` / `--paper-mode` requires `alignment=direct`.

### 3.8 Failure modes and out-of-scope

- Missing quality vector → pad / default distribution (loader-defined).  
- Vision Mamba optional on Windows if wheels/weights absent.  
- Synthetic cache path must never populate submission Results tables.  
- UKB-only diseases omitted from public tables.  
- BRSET requires PhysioNet credentialing; this workspace does not scrape PhysioNet.  
- Optional E5 quality-conditioned **backbone routing** (`quality_gating`: softmax over heads from q) and BRSET `lambda_q` aux are config-gated. MultiCohort joint vocabulary (ODIR+BRSET+RFMiD partial-label masks) is available via `--multi-cohort`.
- Extension inference: `scripts/predict_extension.py` (not upstream `inference.py`).
- Discrimination CIs: patient-level bootstrap and DeLong asymptotic CI in evaluate JSON; primary utility plots are **per-disease DCA curves**.

### 3.9 Reproducibility hooks

Configs: `configs/ablation_e0.yaml` … `e4.yaml` (plus legacy `ablation_*.yaml`). Runner: `scripts/run_ablations.py`. Evaluation: `scripts/evaluate.py` (`--calibrate`, `--test-data-dir`, `--paper-mode`). Public checklist: `python scripts/download_public_data.py`.

---

## 4. Experiments (protocol)

- **Datasets:** ODIR-5K, BRSET, RFMiD (full-image download status: see Data availability; clinical tables **待补充**).  
- **Baselines:** E0 released-code clone (fixed q, independent heads); E1 monotone; E2 multitask; E3 monotone+multitask; E4 full + disjoint calibration protocol.  
- **Primary endpoint:** macro AUROC on frozen test split (real data).  
- **Utility reporting (framework):** ECE, Brier, **per-disease DCA curves**; NB@0.10 illustrative only.  
- **Transfer:** **Endpoint-aware cross-cohort evaluation** (train ODIR, test BRSET overlapping heads with alignment flags; reverse).  
- **Subgroups:** age tertiles, sex on BRSET (**待补充**).  

**Honesty rule:** never place synthetic/demo AUROC in submission tables.

---

## 5. Results

**Clinical result tables: 待补充** (require real ODIR/BRSET/RFMiD pixels + foundation feature extraction on GPU).

For engineering verification only, a synthetic ablation matrix exists in `results/ablation_summary.csv` (companion report). Those values are **pipeline sanity**, typically near chance on tiny synthetic caches, and must not be compared to Reti-Pioneer’s 0.699–0.833 UKB internal AUROCs.

**Expected qualitative story after real features (hypothesis, not claim):** modest AUROC gains from multi-task correlation; larger ECE reductions from temperature scaling on a disjoint calibration fold; monotone routing helps more when true quality labels exist (BRSET); cross-cohort drop remains large and must be discussed under endpoint alignment constraints.

---

## 6. Discussion

Public ocular+systemic labels enable reproducible methods claims that UKB-gated papers cannot always support. Transferring Reti-Pioneer’s skeleton to ODIR/BRSET forces explicit **endpoint harmonization** and domain-shift reporting. Fairness and calibration should be first-class **evaluation** practices, matching the authors’ own limitations discussion — without overselling calibration/DCA as novelty.

**Figure reading contract (companion report; manuscript clinical panels 待补充).** Figure 1 is an information-flow schematic (yellow = monotone quality fusion; green = multi-task + calibration/DCA **evaluation**). Figures 2–4 currently plot **SYNTHETIC** feature-cache metrics near chance with tiny *n*; they justify engineering readiness and the *protocol* of reporting ECE and cross-cohort drop, not clinical superiority over Reti-Pioneer’s UKB AUROCs.

**Honest novelty phrasing.** Prefer: “we replace fixed quality weights with a monotone bounded router, train a masked multi-task head against the released independent-loop control, and evaluate with endpoint-aware cross-cohort splits plus a calibration/DCA reporting framework.” Avoid: “we outperform Reti-Pioneer” or “we achieve AUROC *X* on T2DM” without the original UKB endpoint and cohort.

**Draft risks (pre-submission).** (1) Results § still empty of real AUROC/ECE. (2) Published “multitask” vs released independent-head loops must stay explicit. (3) Endpoint drift if `diabetes_related` is misused clinically. (4) Monotone router without BRSET quality supervision may stay near (0, 0.5, 1). (5) DCA @ 0.10 is illustrative. (6) Without CUDA, foundation features remain **待补充**.

---

## 7. Limitations

No UKB / SEED access in this workspace; label mismatch vs ICD systemic endpoints; no prospective trial; Vision Mamba optional on Windows; EyeQ weights external; synthetic caches for CI only; ODIR Kaggle credentials absent; BRSET PhysioNet DUA not completed here; GPU absent for RETFound/Swin extraction.

---

## 8. Reproducibility

Code, configs (`configs/ablation_e*.yaml`), `scripts/run_ablations.py`, patient-level split with calibration hashes, and `docs/REPRODUCIBILITY.md`.

**Code availability (public snapshot for review):** https://github.com/Coucou2016/retinal-imaging-methods-public — code, configs, docs, and SYNTHETIC ablation CSV/figures only; no patient images or UKB extracts. See `THIRD_PARTY_NOTICES.md`. MIT `LICENSE` governs copyright; research disclaimers in README are not license restrictions.

**Data availability:** Public ODIR-5K / BRSET / RFMiD under their respective licenses (BRSET via PhysioNet credentialing). UK Biobank data are **not** redistributed here; clinical UKB AUROCs remain 待补充 until authorized access and re-analysis.

```powershell
python -m unittest discover -s tests -v
python scripts/plot_paper_figures.py
python scripts/run_ablations.py --quick
python scripts/download_public_data.py
python scripts/build_paper_report.py
```

---

## References

1. Zhang, X. et al. AI framework for multidisease detection via retinal imaging. *Nat Med* **32**, 2494–2503 (2026). doi:10.1038/s41591-026-04359-w  
2. Research Briefing. An AI framework for multi-disease detection via retinal imaging. *Nat Med* **32**, 2366–2367 (2026). doi:10.1038/s41591-026-04424-4  
3. Zhou, Y. et al. A foundation model for generalizable disease detection from retinal images. *Nature* **622**, 156–163 (2023). doi:10.1038/s41586-023-06555-x  
4. Fu, H. et al. Evaluation of retinal image quality assessment networks in different color-spaces (EyeQ / MCF-Net). *MICCAI* (2019). doi:10.1007/978-3-030-32239-7_6  
5. ODIR-5K Grand Challenge / dataset documentation. https://odir2019.grand-challenge.org/dataset/  
6. Nakayama, L. F. et al. BRSET: A Brazilian Multilabel Ophthalmological Dataset of Retina Fundus Photos. *PLOS Digit Health* **3**, e0000454 (2024). doi:10.1371/journal.pdig.0000454; PhysioNet v1.0.2  
7. Pachade, S. et al. Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research. *Data* **6**, 14 (2021). doi:10.3390/data6020014  
8. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. On calibration of modern neural networks. *ICML* / PMLR **70** (2017).  
9. Vickers, A. J. & Elkin, E. B. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making* **26**, 565–574 (2006).  
10. Quellec, G. et al. Towards population-independent, multi-disease detection in fundus photographs. *Sci Rep* **13**, 11807 (2023). doi:10.1038/s41598-023-38610-y  
11. Official Reti-Pioneer code: https://github.com/lyhyl/Reti-Pioneer  

---

## Assumptions or missing inputs

- Real ODIR/BRSET full-image downloads not completed (no Kaggle auth; BRSET PhysioNet credentialed).  
- RFMiD: **images + official CSVs present** under `data/raw/rfmid` (N=3200); labels prepared with `pairs.csv`. Full 3-backbone foundation features for clinical tables remain **待补充** (CPU torch only; see `docs/DATA_BLOCKERS.md`). ODIR (Kaggle) / BRSET (PhysioNet) downloads blocked without credentials.  
- No invented UKB / clinical AUROCs.  
- ChatGPT live consultation optional/blocked; local engineering prioritized.
