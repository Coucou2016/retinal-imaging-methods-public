# Quality-Adaptive Multi-Task Oculomics: Extending Reti-Pioneer with Learnable Fusion, Calibration, and Public-Cohort Validation

**Manuscript draft (methods paper)** · 2026-08-16  
**Target venues (realistic without UK Biobank):** *npj Digital Medicine*, *Medical Image Analysis*, *IEEE JBHI / TMI*, *Computers in Biology and Medicine*  
**Axes (nature-writing):** task=manuscript · paper_type=methods · language=en · journal=generic (Nature-family structure, not flagship *Nature Medicine* clinical claim)

> **One-sentence argument.** Reti-Pioneer showed that frozen foundation features plus quality-aware fusion can screen systemic disease from color fundus photographs (CFPs); we keep that skeleton, replace fixed quality weights with learnable quality routing, replace independent binary heads with a shared multi-task head, and evaluate calibration / decision-curve utility and cross-dataset generalization on public cohorts reviewers can download — without claiming UK Biobank-comparable area under the receiver operating characteristic curve (AUROC) from synthetic demo caches.

---

## Abstract

Endocrine and metabolic disease screening from retinal imaging (oculomics) remains limited by image-quality heterogeneity, single-task training, and poorly calibrated risk scores. Reti-Pioneer (Zhang et al., *Nature Medicine*, 2026) demonstrated that frozen vision foundation models with quality-aware bilinear fusion can screen six systemic conditions on UK Biobank (UKB) and hospital cohorts, but uses **fixed** quality routing, trains **independent** binary heads, and reports discrimination without treating calibration and decision-curve analysis (DCA) as primary utility. Here we propose a methods extension that (i) unfreezes quality routing (`learnable_q`), (ii) shares a multi-label head across correlated ocular/systemic labels, and (iii) reports expected calibration error (ECE), Brier score, and net benefit under temperature scaling. We specify patient-level evaluation on publicly downloadable Ocular Disease Intelligent Recognition (ODIR-5K), Brazilian Multilabel Ophthalmological Dataset (BRSET), and Retinal Fundus Multi-disease Image Dataset (RFMiD). **Results tables with clinical AUROCs are marked 待补充 until real images and foundation features replace synthetic caches.** Pipeline sanity metrics on synthetic features are reported only in the companion research report and are **not** manuscript claims.

**Keywords:** oculomics; fundus photography; quality-aware fusion; multi-task learning; calibration; domain generalization; Reti-Pioneer

---

## 1. Introduction

Color fundus photography is widely available in community eye care and can encode microvascular signatures of systemic disease. Foundation models such as RETFound (Zhou et al., *Nature*, 2023) provide transferable retinal representations, while quality assessment toolkits such as EyeQ support handling of imperfect acquisitions. Reti-Pioneer integrated frozen RETFound, Swin Transformer V2-B, and Vision Mamba-S with quality-aware fusion and metadata, reporting internal-test AUROCs of 0.699–0.833 across six diseases and demonstrating a primary-care silent trial.

Despite this progress, three methodological gaps remain actionable without UKB access. First, Reti-Pioneer freezes quality weights to good=1 / usable=0.5 / bad=0, which may be suboptimal when quality label distributions differ across cameras and sites. Second, training six independent binary models ignores label correlation (e.g., diabetes-related and hypertensive findings) that multi-label ocular datasets naturally provide. Third, screening deployment needs calibrated probabilities and decision-curve net benefit, not AUROC alone.

**Contributions.**

1. A Reti-Pioneer-compatible **learnable quality routing** module initialized at the paper’s fixed weights.
2. A **shared multi-task head** with joint binary cross-entropy (BCE) over K public labels, with per-class ensemble behavior retained as an ablation.
3. An evaluation protocol that elevates **temperature scaling, ECE, Brier, and DCA net benefit**, plus **patient-level** ODIR<->BRSET head-mapped transfer.
4. An open engineering stack (configs, ablation runner, public loaders) enabling independent reproduction once images are downloaded.

**Boundary.** We do not claim UKB T2DM AUROC 0.833 replication; we do not treat synthetic feature-cache AUROCs as clinical evidence; gout / osteoporosis / hyperlipidemia / thyroid remain UKB-only until public gold standards exist.

---

## 2. Related work

**Foundation models for CFP.** RETFound and related retinal foundation models reduce labeled-data needs for ocular and oculomic tasks. Reti-Pioneer ensembles multiple frozen backbones rather than proposing a new foundation model; our work inherits that stance.

**Quality-aware analysis.** EyeQ-style classifiers produce good/usable/bad probabilities. Prior screening studies show community CFPs are often degraded; Reti-Pioneer’s bilinear quality fusion is a strong baseline. Learnable routing is a bounded refinement, not a new quality taxonomy.

**Multi-label fundus datasets.** ODIR-5K, RFMiD, and BRSET provide multi-label ocular (and some systemic) annotations downloadable by reviewers. They are **not** UKB ICD endpoints; Methods must state label semantics explicitly.

**Calibration and decision curves.** Temperature scaling, ECE, and DCA are standard for clinical utility assessment. RETFound-enhanced community screening studies have used DCA; we adopt the same utility language for public oculomics baselines.

**Positioning.** This is a **methods paper on top of Reti-Pioneer**, not a competitor foundation model and not a Nature Medicine–scale clinical claim without biobank data.

---

## 3. Methods

### 3.1 Inputs

Paired or single CFPs, metadata (age, sex; weight/ethnicity padded when missing), and a 3-way quality probability vector. Patient IDs define folds so both eyes of one person stay in one split.

### 3.2 Frozen backbones and feature cache

Following Reti-Pioneer, we freeze RETFound / Swin V2-B / Vision Mamba-S and train on pre-extracted features (`fast_mode`). Public caches are built via `prepare_public_npz.py` then `extract_features.py`.

### 3.3 Quality routing

`QualityAware` implements bilinear fusion. With `learnable_q=False`, `q_fc` is frozen at (1, 0.5, 0). With `learnable_q=True`, the same initialization is used but `q_fc` is trainable. Optional supervised quality on BRSET focus/illumination/artifact labels is deferred (待补充).

### 3.4 Multi-task head

`ComplexModel` supports `num_classes=K`. Joint BCE (with optional `pos_weight`) shares the fused representation across labels. Independent single-disease training remains the Reti-Pioneer-clone control.

### 3.5 Ensemble

Train-time softmax soft voting and eval-time max across three backbone heads are retained; temperature-scaled mean is an alternative (`--ensemble temp_mean`).

### 3.6 Calibration and decision utility

Temperature T is fit on validation logits. We report AUROC / average precision (AP), ECE, Brier, sensitivity at high specificity, and net benefit at threshold 0.10 versus treat-all / treat-none.

### 3.7 Training protocol

Learning rate 1e-4 (config default), frozen backbones, patient-level split, seeds {42,43,44} for final tables (待补充 on real data). Ablation arms: baseline, learnable_q, multitask, full.

### 3.8 Label mapping (pre-registered)

| Head | ODIR | BRSET | Honesty |
|------|------|-------|---------|
| diabetes-related | D (ocular) | diabetes / DR-related mapping | Not UKB T2DM ICD |
| hypertension-ocular | H | hypertensive retinopathy | Ocular sign ≠ systemic HTN ICD |

---

## 4. Experiments (protocol)

- **Datasets:** ODIR-5K, BRSET, RFMiD (download status: 待补充 for full images).
- **Baselines:** linear probe on RETFound-only (待补充); Reti-Pioneer clone (fixed q, independent/single-task); each add-on.
- **Primary endpoint:** macro AUROC on frozen test split (real data).
- **Co-primary:** ECE and net benefit @ 10% after temperature scaling.
- **Transfer:** train ODIR, test BRSET overlapping heads; reverse.
- **Subgroups:** age tertiles, sex on BRSET (待补充).

**Honesty rule:** never place synthetic/demo AUROC in submission tables.

---

## 5. Results

**Clinical result tables: 待补充** (require real ODIR/BRSET/RFMiD pixels + foundation feature extraction).

For engineering verification only, a synthetic ablation matrix exists in `results/ablation_summary.csv` (see companion report). Those values are **pipeline sanity**, typically near chance on tiny synthetic caches, and must not be compared to Reti-Pioneer’s 0.699–0.833 UKB internal AUROCs.

**Expected qualitative story after real features (hypothesis, not claim):** modest AUROC gains from multi-task correlation; larger ECE reductions from temperature scaling; quality routing helps more when true quality labels exist (BRSET); cross-domain drop remains large and must be discussed.

---

## 6. Discussion

Public ocular+systemic labels enable reproducible methods claims that UKB-gated papers cannot always support. Transferring Reti-Pioneer’s skeleton to ODIR/BRSET forces explicit label semantics and domain-shift reporting. Fairness and calibration should be first-class, matching the authors’ own limitations discussion.

**Figure reading contract (for companion report figures; manuscript clinical panels 待补充).** Figure 1 is an information-flow schematic only (yellow = learnable quality fusion; green = multi-task + calibration/DCA). Figures 2–4 currently plot **SYNTHETIC** feature-cache metrics near chance with tiny *n*; they justify engineering readiness and the *protocol* of reporting ECE and cross-dataset drop, not clinical superiority over Reti-Pioneer’s UKB AUROCs.

**Honest novelty phrasing.** Prefer: “we unfreeze quality routing initialized at Reti-Pioneer’s fixed weights and evaluate jointly with multi-label heads and calibration/DCA on public cohorts.” Avoid: “we outperform Reti-Pioneer” or “we achieve AUROC X on T2DM” without the original UKB endpoint and cohort.

---

## 7. Limitations

No UKB / SEED access in this workspace; label mismatch vs ICD systemic endpoints; no prospective trial; Vision Mamba optional on Windows; EyeQ weights external; synthetic caches for CI only.

---

## 8. Reproducibility

Code, configs (`configs/ablation_*.yaml`), `scripts/run_ablations.py`, patient-level split, and `docs/REPRODUCIBILITY.md`. BRSET requires PhysioNet credentialing.

```powershell
python -m unittest discover -s tests -v
python scripts/plot_paper_figures.py
python scripts/run_ablations.py --quick
```

---

## References (seed list; expand on real submission)

1. Zhang et al. AI framework for multidisease detection via retinal imaging. *Nat Med* (2026). doi:10.1038/s41591-026-04359-w  
2. Zhou et al. A foundation model for generalizable disease detection from retinal images. *Nature* (2023). doi:10.1038/s41586-023-06555-x  
3. Fu et al. Evaluation of retinal image quality assessment systems (EyeQ). *MICCAI* workshops / related EyeQ resources.  
4. ODIR-5K Grand Challenge / dataset documentation.  
5. BRSET: Brazilian Multilabel Ophthalmological Dataset. *PLOS Digit Health* (doi:10.1371/journal.pdig.0000454); PhysioNet.  
6. RFMiD / RFMiD 2.0 multi-disease fundus datasets.  
7. Guo et al. On calibration of modern neural networks. *ICML* 2017 (temperature scaling).  
8. Vickers & Elkin. Decision curve analysis. *Med Decis Making* 2006.  
9. RETFound-enhanced community screening with DCA (e.g., PubMed 38693205).  
10. Official Reti-Pioneer code: https://github.com/lyhyl/Reti-Pioneer  

---

## Assumptions or missing inputs

- Real ODIR/BRSET/RFMiD downloads and GPU feature extraction not completed in this workspace.  
- ChatGPT Pro/Plus live consult was attempted but Cursor browser tabs failed to stick; literature outline above was independently verified via web search + `docs/PAPER_PLAN.md`.  
- No invented UKB results.
