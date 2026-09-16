# Extending Reti-Pioneer with Monotone Quality Routing and Endpoint-Aware Multi-Task Learning

**Methods manuscript** · 2026-09-15  
**Framing:** methods paper (Nature-family structure); not a flagship *Nature Medicine* clinical claim without UK Biobank access  
**Candidate venues without UK Biobank:** *npj Digital Medicine*, *Medical Image Analysis*, *IEEE JBHI / TMI*, *Computers in Biology and Medicine*

---

## Abstract

Screening endocrine and metabolic disease from retinal imaging remains limited by heterogeneous image quality, fragmented single-disease training loops, and risk scores that are seldom examined under calibration or decision-analytic criteria. Reti-Pioneer (Zhang et al., *Nature Medicine*, 2026) showed that frozen vision foundation models with quality-aware bilinear fusion can support multi-disease screening on UK Biobank (UKB) and hospital cohorts. The published article frames a multitask screening system. The released training entry point, however, optimizes independent binary heads in per-disease loops, and quality fusion uses fixed weights (good = 1, usable = 0.5, bad = 0).

We present a methods extension that preserves the frozen-backbone ensemble while revising three controllable components. First, quality scalars follow a **monotone bounded** route so that bad ≤ usable ≤ good within [0, 1], with good normalized to 1. Second, a shared multi-label head is trained with **masked binary cross-entropy**, supervising only labels present for each sample or dataset (**endpoint-aware partial-label multi-task learning**). Third, optional **quality-conditioned backbone routing** (configuration E5) maps the quality distribution to a softmax over foundation heads. Evaluation follows an endpoint-aware cross-cohort protocol on ODIR-5K, BRSET, and RFMiD, with alignment flags (`direct` / `partial` / `related_not_equivalent`), and reports expected calibration error (ECE), Brier score, and per-disease decision-curve analysis (DCA) after temperature scaling fitted on a fold disjoint from the scored set.

Clinical AUROC / ECE / DCA on **real-pixel RFMiD** features (ImageNet foundation surrogates; see §5) are reported in this draft. ODIR / BRSET / UKB clinical tables remain **待补充** (credentials / access). Prior SYNTHETIC feature-cache AUROCs are excluded from main Results. Calibration and DCA constitute an evaluation framework rather than claimed algorithmic novelty.

**Keywords:** oculomics; fundus photography; monotone quality routing; partial-label multi-task learning; quality-conditioned backbone routing; calibration (evaluation); endpoint harmonization; Reti-Pioneer

---

## 1. Introduction

Color fundus photography is widely available in community eye care and can encode microvascular signatures of systemic disease. Foundation models such as RETFound provide transferable retinal representations, while quality assessment resources such as EyeQ / MCF-Net support handling of imperfect acquisitions. Reti-Pioneer integrated frozen RETFound, Swin Transformer V2-B, and Vision Mamba-S with quality-aware fusion and metadata, reporting internal-test AUROCs of 0.699–0.833 across six systemic conditions and describing a primary-care silent trial.

Three methodological gaps remain actionable without UKB access. First, Reti-Pioneer freezes quality weights; unconstrained learnable linear weights can violate the clinical ordering of quality strata, so we constrain the learnable path to be monotone and bounded. Second, although the publication describes multitask screening, the released codebase trains independent binary heads; a shared multi-label head with masked supervision better matches multi-label public ocular datasets whose label spaces only partially overlap. Third, quality may also modulate which foundation heads should dominate the ensemble; we therefore provide an optional quality-conditioned backbone router (E5) as a gated extension. Deployment-oriented screening further needs calibrated probabilities and decision-curve net benefit. We treat temperature scaling, ECE, Brier, and DCA as a standard evaluation framework rather than as the paper’s methodological novelty. Population-independent multi-disease studies further show that single-site AUROC overstates transportability when cameras, taxonomies, and populations shift.

**Contributions.**

1. A Reti-Pioneer-compatible **monotone bounded quality router** as the default learnable path, with fixed 1/0.5/0 retained as the released-code baseline and unconstrained free-linear routing as an ablation only.
2. A **shared multi-task head** with **masked BCE** (supervise only labels present for a sample or dataset), with per-disease independent training retained as the released-code control (**endpoint-aware partial-label MTL**).
3. Optional **E5 quality-conditioned backbone routing**: a small MLP maps the three-way quality distribution to a softmax over foundation heads, with optional soft-quality auxiliary loss when soft targets are available.
4. An **endpoint ontology** with systemic versus ocular_manifestation kinds and alignment flags; **endpoint-aware cross-cohort evaluation** that refuses clinical tables when alignment ≠ `direct`.
5. An open engineering stack (configs E0–E5, ablation runner, public loaders, disjoint calibration splits, frozen operating points, patient-level bootstrap / DeLong CIs) enabling independent reproduction once images and foundation features are available.

**Boundary.** We do not claim UKB type 2 diabetes mellitus (T2DM) AUROC 0.833 replication. We do not treat synthetic feature-cache AUROCs as clinical evidence. Gout, osteoporosis, hyperlipidemia, and thyroid remain UKB-only until public gold standards exist. ODIR diabetes (D) and BRSET diabetes labels are not UKB ICD endpoints. Over-broad `diabetes_related` mappings are exploratory only (`related_not_equivalent`).

---

## 2. Related work

**Foundation models for color fundus photographs (CFPs).** RETFound and related retinal foundation models reduce labeled-data needs for ocular and oculomic tasks. Reti-Pioneer ensembles multiple frozen backbones rather than proposing a new foundation model; our work inherits that stance.

**Quality-aware analysis.** EyeQ-style classifiers produce good / usable / reject (bad) probabilities. Reti-Pioneer’s bilinear quality fusion is a strong baseline. Monotone routing is a bounded refinement of the fusion weights, not a new quality taxonomy. Quality-conditioned mixture over backbones is a complementary route that reallocates ensemble weight rather than only rescaling a fused scalar.

**Multi-label fundus datasets and domain shift.** ODIR-5K, RFMiD, and BRSET provide multi-label ocular (and some systemic) annotations downloadable by reviewers. They are not UKB ICD endpoints. We keep Reti-Pioneer’s frozen-ensemble skeleton and change routing, head sharing, and endpoint-aware transfer reporting.

**Calibration and decision curves (evaluation framework).** Temperature scaling, ECE, and DCA are standard clinical-utility assessments. We adopt them as reporting protocol, not as claimed algorithmic novelty.

**Positioning.** This is a methods paper on top of Reti-Pioneer, not a competitor foundation model and not a *Nature Medicine*–scale clinical claim without biobank data.

---

## 3. Methods

### 3.1 Problem formulation and inputs

Let each eye (or paired eyes) be represented by frozen backbone features \(x \in \mathbb{R}^{d}\), a metadata vector \(m\), and a three-way quality probability \(q = (q_{\mathrm{good}}, q_{\mathrm{usable}}, q_{\mathrm{bad}})\). Patient identifiers define splits so that both eyes of one person remain in one fold. When a calibration fraction is configured, we materialize train / calibration / validation / test so that temperature fitting never reuses the evaluation fold. RFMiD prefers official train / val / test CSVs when present.

### 3.2 Frozen backbones and feature caches

Following Reti-Pioneer, we freeze foundation encoders and train fusion and heads on pre-extracted features (`fast_mode`). Public caches are built from prepared labels then pixel feature extraction. RFMiD in this workspace uses **real-pixel** ImageNet foundation surrogates (see §5.1); CI may still use deterministic synthetic caches flagged by `SYNTHETIC_FEATURES.txt` with `clinical_claim_allowed: false`, which must never populate submission Results tables.

### 3.3 Monotone bounded quality routing

**Motivation.** Fixed weights cannot adapt to site-specific quality distributions. Unconstrained linear weights can invert clinical orderings (for example, up-weighting “bad”). Monotonicity encodes the inductive bias that better quality should not receive a lower fusion weight than worse quality, while remaining differentiable end-to-end.

**Mechanism.** Quality-aware bilinear fusion maps features and a quality-derived scalar into a fused representation. We implement three routers:

| Mode | Behavior |
|------|----------|
| `fixed` | Frozen weights good = 1 / usable = 0.5 / bad = 0 (released-code baseline) |
| `monotone` | Learnable non-negative increments with softplus and cumulative normalization; enforces bad ≤ usable ≤ good ∈ [0, 1] with good = 1 (default when learnable quality is enabled) |
| `free_linear` | Unconstrained linear map \(3 \to 1\) (ablation only) |

For the monotone route we parameterize increments \(\delta \in \mathbb{R}^{3}\), set \(u = \mathrm{softplus}(\delta) + \varepsilon\), form \(c = \mathrm{cumsum}(u)\), and obtain ordered weights \(w = c / c_{3}\). Reordering aligns \(w\) with the (good, usable, bad) layout of \(q\), yielding the scalar \(s = q^{\top} w\). The scalar enters the same bilinear fusion block used by Reti-Pioneer, so the architectural change is localized to the quality path.

**Role in evaluation.** Ablation arms contrast `fixed` versus `monotone` under otherwise matched training (configuration matrix E0–E4), isolating the effect of the ordering constraint from other factors.

### 3.4 Endpoint-aware partial-label multi-task learning

**Motivation.** Public ocular datasets expose partially overlapping label vocabularies. Independent binary loops in the released training code cannot share representation across co-occurring labels when some endpoints are missing for a given sample or cohort.

**Mechanism.** `ComplexModel` exposes \(K\) logits. Masked BCE supervises only targets with a present label (\(y \ge 0\)); missing entries (\(-1\) / NaN) are ignored in the loss denominator. This supports joint training across datasets whose label vocabularies only partially overlap (for example, ODIR eight-way ocular codes versus BRSET systemic / ocular fields), including an optional multi-cohort joint vocabulary with per-endpoint masks.

The released-code control retains independent single-disease training loops as in upstream `main.py`. Ensemble modes include `released_code` (soft vote in training / max at inference), `published_soft_vote`, `mean`, and temperature-averaged variants. Legacy `ensemble="paper"` aliases to `released_code` with a deprecation warning to avoid conflating publication wording with training mechanics.

**Role in evaluation.** Multitask versus single-task arms (E2–E4) hold the quality router fixed or jointly varied, so masked supervision can be attributed separately from monotone routing.

### 3.5 Optional quality-conditioned backbone routing (E5)

**Motivation.** Uniform mean or fixed soft-vote ensembles ignore that different foundation heads may be more reliable under poor acquisition. Attenuating features by a gate can hide that effect inside the fusion block; an explicit mixture over heads makes the reallocation inspectable.

**Mechanism.** When enabled (`quality_gating`), a small multilayer perceptron maps \(q\) to a softmax over the three foundation heads, replacing uniform mean or fixed soft-vote with a quality-conditioned mixture (`QualityBackboneRouter`). An optional soft-quality auxiliary loss (\(\lambda_q\)) can be used when soft quality targets are available (for example, BRSET). E5 is config-gated (`configs/ablation_e5.yaml`: `quality_gating: true`, `lambda_q: 0.1`, monotone + multitask + masked BCE) and is not asserted as a clinical improvement in this draft. A legacy feature-attenuation gate remains available for ablation only.

**Role in evaluation.** E5 is reported as an optional arm once real foundation features are available; synthetic pipeline tables in Section 5.1 focus on E0–E4-style baseline / learnq / multitask / full runs that completed under CPU feature caches.

### 3.6 Calibration and decision utility (evaluation framework)

Temperature \(T\) is fit on **calibration_ids** disjoint from **evaluation_ids**. If a request would fit and score the same set, `paper_mode` errors; otherwise a held-out calibration slice is nested and a warning is emitted. Operating points (Youden / sensitivity at 95% specificity) are selected on validation or calibration only and applied frozen on the evaluation fold. We report AUROC / average precision with optional patient-level bootstrap 95% CI and DeLong asymptotic CI, equal-width ECE, Brier score, optional calibration intercept / slope, sensitivity at high specificity, per-disease DCA curves, and (illustrative only) net benefit at threshold 0.10 versus treat-all / treat-none.

### 3.7 Training protocol

| Item | Setting |
|------|---------|
| Learning rate | \(1\times10^{-4}\) (config default) |
| Backbones | Frozen; train fusion + head on caches |
| Split | Patient-level; train / cal / val / (test); both eyes co-located |
| Seeds (final tables) | {42, 43, 44} — **待补充** on real data |
| Ablation arms | E0–E5 (`configs/ablation_e*.yaml`) |
| Software (this study) | PyTorch 2.x; CUDA unavailable in the drafting workspace → foundation extraction blocked |

### 3.8 Endpoint ontology (pre-registered)

| Endpoint | Kind | ODIR | BRSET | Alignment notes |
|----------|------|------|-------|-----------------|
| `diabetes_ocular` | ocular_manifestation | D | `dr_referable` | direct / partial — not UKB T2DM |
| `diabetes_systemic` | systemic | — | `diabetes` | direct on BRSET; still not UKB ICD |
| `diabetes_related` | systemic (exploratory) | D | `diabetes` | **related_not_equivalent** — refuse clinical tables |
| `hypertension_ocular` | ocular_manifestation | H | hypertensive retinopathy | direct ocular; ≠ systemic HTN ICD |

Cross-evaluation with clinical-table or paper-mode guards requires `alignment=direct`. Full harmonization lives with the public-data documentation and label map.

### 3.9 Failure modes and out-of-scope

Missing quality vectors are padded with a loader-defined default. Vision Mamba is optional when wheels or weights are absent. Synthetic caches must never populate submission clinical tables. UKB-only diseases are omitted from public tables. BRSET requires PhysioNet credentialing. Extension inference uses the project predict entry point rather than upstream image-API inference when only caches are available.

---

## 4. Experiments (protocol)

- **Datasets:** ODIR-5K, BRSET, RFMiD (full-image download and three-backbone extraction status: see Data availability; clinical tables **待补充**).
- **Baselines / arms:** E0 released-code clone (fixed \(q\), independent heads); E1 monotone; E2 multitask; E3 monotone + multitask; E4 full + disjoint calibration protocol; E5 optional quality-conditioned backbone routing.
- **Primary endpoint (intended clinical tables):** macro AUROC on a frozen test split with real foundation features.
- **Utility reporting (framework):** ECE, Brier, per-disease DCA; NB@0.10 illustrative only.
- **Transfer:** endpoint-aware cross-cohort evaluation (for example, train ODIR, test BRSET overlapping heads with alignment flags; reverse).
- **Subgroups:** age tertiles, sex on BRSET (**待补充**).

Synthetic or demo AUROC must never appear in submission clinical tables without an explicit SYNTHETIC label and `clinical_claim_allowed = false`.

---

## 5. Results

### 5.1 Intuitive results on real-pixel RFMiD (figures first)

RFMiD official-split images (\(N=3200\); train/val/test = 1920/640/640) were used to build a **real-pixel** UKB-style feature cache (`scripts/build_real_rfmid_cache.py`). Soft quality vectors were estimated from a sharpness/exposure proxy (argmax strata: good 179 / usable 2793 / bad 228). Backbone slots were filled from real fundus pixels with ImageNet-pretrained surrogates because RETFound (Hugging Face gated, HTTP 401 without token) and Vision Mamba (unavailable on this Windows host) could not be loaded: Swin-V2-B → `UKB_swin.npz`; ViT-B/16 projected to 1024-d → `UKB_RETF.npz`; ViT-S/16 → `UKB_vim.npz` (384-d). `SYNTHETIC_FEATURES.txt` was removed. Labels were restricted to eight heads with adequate prevalence: Disease_Risk, DR, MH, ODC, TSLN, DN, ARMD, MYA. These are **not** Reti-Pioneer UKB ICD endpoints and **not** RETFound/Vim replications; they are real-pixel RFMiD ocular-sign results with honest surrogate-backbone disclosure (`data/rfmid/FEATURE_PROVENANCE.json`).

**Figure panel (SciencePlots + Times New Roman; regenerated 2026-09-17):**

1. Architecture schematic (Figure 1) — unchanged methods diagram.  
2. RFMiD fundus examples by quality stratum with soft \(q\) (Figure 5: `fig5_fundus_quality`).  
3. Quality-stratum pie / counts (Figure 6b) and monotone / E5 routing under good→usable→bad intervention (Figure 6).  
4. Per-disease ROC curves on the held-out test fold for the multitask E5 head (Figure 8) — curves precede AUC numbers.  
5. Reliability diagrams for DR / Disease_Risk / MH (Figure 3).  
6. Per-disease decision curves (NB vs threshold) (Figure 7).  
7. Ablation bars E0–E5 on RFMiD test (Figure 2).  
8. DR score distributions (Figure 9).

ODIR (Kaggle) and BRSET (PhysioNet) remain **待补充** — no credentials in this workspace (`docs/DATA_BLOCKERS.md`).

### 5.2 Quantitative RFMiD test metrics (after figures)

Arms E0–E5 were trained for 12 epochs on the frozen official split with nested calibration (\(n_{\mathrm{cal}}=64\)) carved from validation; temperature scaling was fit on calibration and applied on val/test. Metrics below are **test** (\(n=640\)), with patient-level bootstrap AUROC 95% CI (\(B=200\)). Primary comparable column for single-task arms is AUROC\(_{\mathrm{DR}}\); multitask arms also report macro AUROC across eight heads.

**Table 1. RFMiD test ablation (real-pixel features; repository-computed).**

| Arm | Macro AUROC | AUROC\(_{\mathrm{DR}}\) | ECE | Cal. ECE | \(T\) | \(n\) | Bootstrap 95% CI (macro) |
|-----|-------------|-------------------------|-----|----------|-------|-------|--------------------------|
| E0 (fixed-q, DR) | 0.925 | 0.925 | 0.026 | 0.031 | 1.06 | 640 | [0.902, 0.947] |
| E1 (monotone-q, DR) | 0.931 | 0.931 | 0.034 | 0.032 | 1.11 | 640 | [0.908, 0.949] |
| E2 (MTL, fixed-q) | 0.886 | 0.910 | 0.170 | 0.176 | 1.06 | 640 | [0.872, 0.902] |
| E3 (MTL + monotone) | 0.898 | 0.928 | 0.178 | 0.188 | 1.11 | 640 | [0.882, 0.914] |
| E4 (E3 + cal. protocol) | 0.898 | 0.928 | 0.178 | 0.188 | 1.11 | 640 | [0.882, 0.914] |
| E5 (+ backbone routing) | 0.901 | 0.916 | 0.156 | 0.154 | 0.98 | 640 | [0.887, 0.914] |

Source JSON/CSV: `results/real_rfmid/metrics_*_rfmid_test.json`, `results/real_rfmid/ablation_summary.csv`. E5 per-head test AUROCs (illustrative): Disease_Risk 0.959, DR 0.916, MH 0.957, ODC 0.742, TSLN 0.924, DN 0.780, ARMD 0.939, MYA 0.989. Do not compare these ocular-sign AUROCs with Reti-Pioneer UKB systemic AUROCs (0.699–0.833).

### 5.3 Still 待补充

| Cohort / claim | Status |
|----------------|--------|
| ODIR-5K real pixels + foundation extract | **待补充** (no Kaggle credentials) |
| BRSET real pixels + foundation extract | **待补充** (PhysioNet DUA not completed) |
| RETFound / Vim-S exact Reti-Pioneer weights | **待补充** (HF gated / mamba unavailable); surrogates used above |
| UKB / hospital systemic clinical tables | **待补充** (no UKB access) |
| Multi-seed {42,43,44} final tables | **待补充** (single seed 42 reported) |

Provenance: `docs/paper/AUTHENTICITY_AUDIT.md`, `docs/DATA_BLOCKERS.md`.

---

## 6. Discussion

Public ocular and systemic labels enable reproducible methods claims that UKB-gated papers cannot always support. Transferring Reti-Pioneer’s skeleton to ODIR / BRSET / RFMiD forces explicit endpoint harmonization and domain-shift reporting. Fairness and calibration should be first-class evaluation practices, matching the authors’ own limitations discussion, without overselling calibration or DCA as novelty.

Figure 1 is an information-flow schematic (yellow = monotone quality fusion; green = partial-label multi-task head and calibration / DCA evaluation). Figures 2–3 and 5–9 report **real-pixel RFMiD** results (ImageNet foundation surrogates). ODIR/BRSET/UKB clinical transportability remains out of scope until credentials and biobank access exist.

A concise novelty statement is: we replace fixed quality weights with a monotone bounded router, train a masked multi-task head against the released independent-loop control, optionally route backbone mixture weights from quality (E5), and evaluate with endpoint-aware splits plus a calibration / DCA reporting framework. We do not claim to outperform Reti-Pioneer on UKB, and we do not report a UKB T2DM AUROC without the original endpoint and cohort.

Open risks include surrogate (not RETFound/Vim) backbones on RFMiD, empty ODIR/BRSET Results, published-versus-released multitask wording, endpoint drift if `diabetes_related` is misused clinically, and monotone weights that may remain near (0, 0.5, 1) without EyeQ supervision.

---

## 7. Limitations

No UKB / SEED access in this drafting workspace; RFMiD labels are retinal disease signs rather than ICD systemic endpoints; no prospective trial; Vision Mamba optional / unavailable here; EyeQ weights external; RETFound HF gated without token → ImageNet surrogates used for RFMiD; ODIR Kaggle credentials absent; BRSET PhysioNet DUA not completed; GTX 950M present but CUDA PyTorch unavailable (CPU extract). ODIR / BRSET / UKB clinical tables therefore remain **待补充**.

---

## 8. Reproducibility and availability

Code, ablation configs, patient-level splits with calibration hashes, and evaluation guards are released for independent checking. A public code-and-docs snapshot is available at https://github.com/Coucou2016/retinal-imaging-methods-public (no patient images or UKB extracts). See third-party notices for upstream Reti-Pioneer attribution.

**Data availability.** Public ODIR-5K / BRSET / RFMiD under their respective licenses (BRSET via PhysioNet credentialing). UK Biobank data are not redistributed; clinical UKB AUROCs remain **待补充** until authorized access and re-analysis.

Representative commands (environment-dependent):

```text
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
