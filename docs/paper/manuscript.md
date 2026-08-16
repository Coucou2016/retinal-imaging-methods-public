# Quality-Adaptive Multi-Task Oculomics: Extending Reti-Pioneer with Learnable Fusion, Calibration, and Public-Cohort Validation

**Manuscript draft (methods paper)** · 2026-08-16 (iter-5 mature)  
**Axes (nature-writing):** task=manuscript · paper_type=methods · language=en · journal=generic (Nature-family structure; not a flagship *Nature Medicine* clinical claim)  
**Realistic venues without UK Biobank:** *npj Digital Medicine*, *Medical Image Analysis*, *IEEE JBHI / TMI*, *Computers in Biology and Medicine*

> **One-sentence argument.** Reti-Pioneer showed that frozen foundation features plus quality-aware fusion can screen systemic disease from color fundus photographs (CFPs); we keep that skeleton, replace fixed quality weights with learnable quality routing, replace independent binary heads with a shared multi-task head, and evaluate calibration / decision-curve utility and patient-level cross-dataset generalization on public cohorts reviewers can download — without claiming UK Biobank-comparable area under the receiver operating characteristic curve (AUROC) from synthetic demo caches.

---

## Abstract

Endocrine and metabolic disease screening from retinal imaging (oculomics) remains limited by image-quality heterogeneity, single-task training, and poorly calibrated risk scores. Reti-Pioneer (Zhang et al., *Nature Medicine*, 2026) demonstrated that frozen vision foundation models with quality-aware bilinear fusion can screen six systemic conditions on UK Biobank (UKB) and hospital cohorts, but uses **fixed** quality routing, trains **independent** binary heads in the released codebase, and reports discrimination without treating calibration and decision-curve analysis (DCA) as co-primary utility. Here we propose a methods extension that (i) unfreezes quality routing (`learnable_q`), (ii) shares a multi-label head across correlated ocular/systemic labels, and (iii) reports expected calibration error (ECE), Brier score, and net benefit under temperature scaling. We specify patient-level evaluation on publicly downloadable Ocular Disease Intelligent Recognition (ODIR-5K), Brazilian Multilabel Ophthalmological Dataset (BRSET), and Retinal Fundus Multi-disease Image Dataset (RFMiD).

**Honesty banner.** Clinical AUROC / ECE / decision-curve tables are marked **待补充** until real images and foundation features replace synthetic caches. Pipeline sanity metrics on synthetic features appear only in the companion research report and are **not** manuscript claims. We do not invent UKB or hospital AUROCs.

**Keywords:** oculomics; fundus photography; quality-aware fusion; multi-task learning; calibration; domain generalization; Reti-Pioneer

---

## 1. Introduction

Color fundus photography is widely available in community eye care and can encode microvascular signatures of systemic disease. Foundation models such as RETFound (Zhou et al., *Nature*, 2023) provide transferable retinal representations, while quality assessment resources such as EyeQ / MCF-Net (Fu et al., MICCAI 2019) support handling of imperfect acquisitions. Reti-Pioneer integrated frozen RETFound, Swin Transformer V2-B, and Vision Mamba-S with quality-aware fusion and metadata, reporting internal-test AUROCs of 0.699–0.833 across six diseases and demonstrating a primary-care silent trial.

Despite this progress, three methodological gaps remain actionable without UKB access. First, Reti-Pioneer freezes quality weights to good=1 / usable=0.5 / bad=0, which may be suboptimal when quality label distributions differ across cameras and sites. Second, although the Reti-Pioneer abstract describes a multitask screening framework, the **released training loop uses independent binary heads**; training separate models ignores label correlation that multi-label ocular datasets naturally provide. Third, screening deployment needs calibrated probabilities and decision-curve net benefit, not AUROC alone. Population-independent multi-disease studies further show that single-site AUROC overstates transportability when cameras, taxonomies, and populations shift (e.g., Quellec et al., *Scientific Reports*, 2023).

**Contributions.**

1. A Reti-Pioneer-compatible **learnable quality routing** module initialized at the paper’s fixed weights (`q_fc` → 1 / 0.5 / 0).
2. A **shared multi-task head** with joint binary cross-entropy (BCE) over *K* public labels, with per-class independent training retained as the clone control.
3. An evaluation protocol that elevates **temperature scaling, ECE, Brier, and DCA net benefit**, plus **patient-level** ODIR↔BRSET head-mapped transfer.
4. An open engineering stack (configs, ablation runner, public loaders) enabling independent reproduction once images are downloaded.

**Boundary.** We do not claim UKB type 2 diabetes mellitus (T2DM) AUROC 0.833 replication; we do not treat synthetic feature-cache AUROCs as clinical evidence; gout / osteoporosis / hyperlipidemia / thyroid remain UKB-only until public gold standards exist. ODIR diabetes (D) and BRSET diabetes labels are **not** UKB ICD endpoints.

---

## 2. Related work

**Foundation models for CFP.** RETFound and related retinal foundation models reduce labeled-data needs for ocular and oculomic tasks. Reti-Pioneer ensembles multiple frozen backbones rather than proposing a new foundation model; our work inherits that stance.

**Quality-aware analysis.** EyeQ-style classifiers produce good/usable/reject (bad) probabilities. Prior screening studies show community CFPs are often degraded; Reti-Pioneer’s bilinear quality fusion is a strong baseline. Learnable routing is a **bounded refinement of the fusion weights**, not a new quality taxonomy.

**Multi-label fundus datasets and domain shift.** ODIR-5K, RFMiD, and BRSET provide multi-label ocular (and some systemic) annotations downloadable by reviewers. They are **not** UKB ICD endpoints. Multi-label ODIR classifiers (CNNs, transformers, label-attention graphs) optimize ocular taxonomies; we instead keep Reti-Pioneer’s frozen-ensemble skeleton and change routing, head sharing, and utility metrics. Cross-population multi-disease detection work motivates reporting patient-level transfer rather than source-only AUROC.

**Calibration and decision curves.** Temperature scaling (Guo et al., ICML 2017), ECE, and DCA (Vickers & Elkin, 2006) are standard for clinical utility assessment. We adopt the same utility language for public oculomics baselines.

**Positioning.** This is a **methods paper on top of Reti-Pioneer**, not a competitor foundation model and not a *Nature Medicine*–scale clinical claim without biobank data.

---

## 3. Methods

### 3.1 Inputs

Paired or single CFPs, metadata (age, sex; weight/ethnicity padded when missing), and a 3-way quality probability vector. Patient IDs define folds so both eyes of one person stay in one split (`reti_pioneer.split.patient_level_train_val_indices`). RFMiD prefers official train/val/test CSVs when present.

### 3.2 Frozen backbones and feature cache

Following Reti-Pioneer, we freeze RETFound / Swin V2-B / Vision Mamba-S and train on pre-extracted features (`fast_mode`). Public caches are built via `prepare_public_npz.py` then `extract_features.py`. Until real pixels and CUDA extraction succeed, caches may contain deterministic **synthetic** features flagged by `SYNTHETIC_FEATURES.txt` — usable for continuous integration (CI) only.

### 3.3 Quality routing

`QualityAware` implements bilinear fusion. With `learnable_q=False`, `q_fc` is frozen at (1, 0.5, 0). With `learnable_q=True`, the same initialization is used but `q_fc` is trainable. Optional supervised quality on BRSET focus/illumination/artifact labels is deferred (**待补充**). Without quality supervision, learnable weights may collapse toward the fixed initialization; we treat that as a failure mode to monitor (weight drift diagnostics **待补充** on real data).

### 3.4 Multi-task head

`ComplexModel` supports `num_classes=K`. Joint BCE (with optional `pos_weight`) shares the fused representation across labels. **Reti-Pioneer clone control** = independent single-disease training as in the released codebase (not a straw-man “no multitask abstract wording”). Soft voting in train and max across three backbone heads at eval are retained; temperature-scaled mean is an alternative (`--ensemble temp_mean`).

### 3.5 Calibration and decision utility

Temperature *T* is fit on validation logits by minimizing negative log-likelihood. We report AUROC / average precision (AP), equal-width ECE (`n_bins=10`, empty bins skipped), Brier score, sensitivity at high specificity, and net benefit at threshold 0.10 versus treat-all / treat-none. Threshold 0.10 is **illustrative** until cohort-specific prevalence and cost ratios are fixed.

### 3.6 Training protocol

| Item | Setting |
|------|---------|
| Learning rate | 1×10⁻⁴ (config default) |
| Backbones | Frozen; train fusion + head on caches |
| Split | Patient-level; both eyes co-located |
| Seeds (final tables) | {42, 43, 44} — **待补充** on real data |
| Ablation arms | `baseline`, `learnq`, `multitask`, `full` |
| Software (this workspace) | PyTorch 2.x CPU build; CUDA **unavailable** → foundation extraction blocked |

### 3.7 Label mapping (pre-registered)

| Head | ODIR | BRSET | Honesty |
|------|------|-------|---------|
| diabetes-related | D (ocular) | diabetes / DR-related mapping | Not UKB T2DM ICD |
| hypertension-ocular | H | hypertensive retinopathy | Ocular sign ≠ systemic HTN ICD |

Do not pool BRSET diabetes, DR grade, and hypertensive retinopathy without an explicit footnote.

### 3.8 Failure modes and out-of-scope

- Missing quality vector → pad / default distribution (loader-defined).  
- Vision Mamba optional on Windows if wheels/weights absent.  
- Synthetic cache path must never populate submission Results tables.  
- UKB-only diseases (gout, osteoporosis, hyperlipidemia, thyroid) omitted from public tables.  
- BRSET requires PhysioNet credentialing; this workspace does not scrape PhysioNet.

### 3.9 Reproducibility hooks

Configs: `configs/ablation_*.yaml`. Runner: `scripts/run_ablations.py`. Evaluation: `scripts/evaluate.py` (`--calibrate`, `--test-data-dir`). Public checklist: `python scripts/download_public_data.py`.

---

## 4. Experiments (protocol)

- **Datasets:** ODIR-5K, BRSET, RFMiD (full-image download status: see Data availability; clinical tables **待补充**).  
- **Baselines:** linear probe on RETFound-only (**待补充**); Reti-Pioneer clone (fixed q, independent heads); each add-on; full combination.  
- **Primary endpoint:** macro AUROC on frozen test split (real data).  
- **Co-primary:** ECE and net benefit @ 10% after temperature scaling.  
- **Transfer:** train ODIR, test BRSET overlapping heads; reverse.  
- **Subgroups:** age tertiles, sex on BRSET (**待补充**).  

**Honesty rule:** never place synthetic/demo AUROC in submission tables.

---

## 5. Results

**Clinical result tables: 待补充** (require real ODIR/BRSET/RFMiD pixels + foundation feature extraction on GPU).

For engineering verification only, a synthetic ablation matrix exists in `results/ablation_summary.csv` (companion report). Those values are **pipeline sanity**, typically near chance on tiny synthetic caches (*n*≈12 validation / 48 transfer in the quick run), and must not be compared to Reti-Pioneer’s 0.699–0.833 UKB internal AUROCs.

**Expected qualitative story after real features (hypothesis, not claim):** modest AUROC gains from multi-task correlation; larger ECE reductions from temperature scaling; quality routing helps more when true quality labels exist (BRSET); cross-domain drop remains large and must be discussed.

---

## 6. Discussion

Public ocular+systemic labels enable reproducible methods claims that UKB-gated papers cannot always support. Transferring Reti-Pioneer’s skeleton to ODIR/BRSET forces explicit label semantics and domain-shift reporting. Fairness and calibration should be first-class, matching the authors’ own limitations discussion.

**Figure reading contract (companion report; manuscript clinical panels 待补充).** Figure 1 is an information-flow schematic only (yellow = learnable quality fusion; green = multi-task + calibration/DCA). Figures 2–4 currently plot **SYNTHETIC** feature-cache metrics near chance with tiny *n*; they justify engineering readiness and the *protocol* of reporting ECE and cross-dataset drop, not clinical superiority over Reti-Pioneer’s UKB AUROCs.

**Honest novelty phrasing.** Prefer: “we unfreeze quality routing initialized at Reti-Pioneer’s fixed weights and evaluate jointly with multi-label heads and calibration/DCA on public cohorts.” Avoid: “we outperform Reti-Pioneer” or “we achieve AUROC *X* on T2DM” without the original UKB endpoint and cohort.

**Draft risks (pre-submission; independent audit).** (1) Results § still empty of real AUROC/ECE — reviewers will reject if SYNTHETIC companion figures are mistaken for claims. (2) Reti-Pioneer’s abstract says “multitask”; our clone control is **independent binary heads** as in the released codebase — state that distinction explicitly. (3) ODIR-D / BRSET diabetes labels remain ocular or mixed, not UKB T2DM ICD. (4) Learnable `q_fc` without BRSET quality supervision may collapse to near-fixed weights. (5) DCA @ 0.10 is illustrative until prevalence and costs are cohort-specific. (6) Without CUDA, foundation features remain **待补充** even if public images arrive.

---

## 7. Limitations

No UKB / SEED access in this workspace; label mismatch vs ICD systemic endpoints; no prospective trial; Vision Mamba optional on Windows; EyeQ weights external; synthetic caches for CI only; ODIR Kaggle credentials absent; BRSET PhysioNet DUA not completed here; GPU absent for RETFound/Swin extraction.

---

## 8. Reproducibility

Code, configs (`configs/ablation_*.yaml`), `scripts/run_ablations.py`, patient-level split, and `docs/REPRODUCIBILITY.md`.

**Code availability (public snapshot for review):** https://github.com/Coucou2016/retinal-imaging-methods-public — code, configs, docs, and SYNTHETIC ablation CSV/figures only; no patient images or UKB extracts.

**Data availability:** Public ODIR-5K / BRSET / RFMiD under their respective licenses (BRSET via PhysioNet credentialing). UK Biobank data are **not** redistributed here; clinical UKB AUROCs remain 待补充 until authorized access and re-analysis. RFMiD label CSVs may be retrieved via Hugging Face `ctmedtech/RFMID` when network allows; full-image + foundation-feature caches are **待补充** on CPU-only hosts.

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
- RFMiD: label CSVs downloadable via Hugging Face; full images + foundation features **待补充** (CPU-only; no CUDA).  
- ChatGPT live 5-round automation blocked (`cursor-ide-browser` absent / Cloudflare). Five local structured rounds + paste briefs in `docs/chatgpt-runs/2026-08-16-iter5/`; **no invented ChatGPT replies**.  
- No invented UKB / clinical AUROCs.
