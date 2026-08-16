# Follow-up paper plan on Reti-Pioneer

Baseline: Zhang et al., *AI framework for multidisease detection via retinal imaging*. *Nature Medicine* (2026). DOI: [10.1038/s41591-026-04359-w](https://doi.org/10.1038/s41591-026-04359-w).

This document is a writing and experiment plan. It does **not** claim clinical AUROCs from demo tensors.

## 1. What can actually be an innovation

Reti-Pioneer already did: frozen RETFound + Swin V2-B + Vision Mamba-S, **fixed** quality weights (good=1, usable=0.5, bad=0), bilinear fusion with metadata, **one binary head per disease**, soft voting in train / max at inference, UKB+hospital data, SEED external validation, silent trial.

The authors themselves list the gaps you should write against (*Nature Medicine* discussion):

1. Accuracy is still below the bar for broad clinical adoption.
2. Longitudinal task is **binary at fixed 5/10-year cutoffs**, not survival analysis.
3. Scope excludes CAD, stroke, mortality, rare disease.
4. Residual confounding; SEED shows ethnicity imbalance still matters.
5. Prospective benefit needs larger RCTs.

A follow-up paper that only “reimplements Reti-Pioneer” is not publishable. A follow-up that **closes one or two of those gaps with methods you can ablate on public data** is.

### Recommended thesis (one sentence)

Reti-Pioneer showed that frozen foundation features plus quality-aware fusion can screen systemic disease from fundus photographs; we keep that skeleton, replace **fixed quality weights** with **learnable quality routing**, replace **six independent binary models** with a **shared multi-task head**, and evaluate **calibration / decision-curve utility** and **cross-dataset generalization** on public cohorts that independent reviewers can download.

Suggested title:

> Quality-Adaptive Multi-Task Oculomics: Extending Reti-Pioneer with Learnable Fusion, Calibration, and Public-Cohort Validation

Target venues (realistic without UKB): *npj Digital Medicine*, *Medical Image Analysis*, *IEEE JBHI*, *IEEE TMI*, *Computers in Biology and Medicine*. Do not aim at *Nature Medicine* unless you obtain UKB + a prospective cohort.

### Innovation stack (priority)

| ID | Innovation | Why it is new vs Reti-Pioneer | Feasible here | Public verification | Risk |
|----|------------|-------------------------------|---------------|---------------------|------|
| I | Learnable quality routing | Paper freezes `q_fc` to 1 / 0.5 / 0 | `QualityAware(learnable_q=True)` already exists | BRSET has focus/illumination/artifact labels | Low |
| II | Joint multi-label head | Paper trains six separate binary models | Change `num_classes` and loss in `ComplexModel` | ODIR-5K (8 labels), BRSET multi-label | Low |
| III | Calibration + decision curve as primary utility | Helpers exist but are not in the train/eval loop | `utils/calibration.py` | Any public binary/multi-label set | Low |
| IV | Patient-level domain generalization | External validation was reported, shift was not modelled | Feature caches + train on A / test on B | ODIR ↔ BRSET ↔ RFMiD | Medium |
| V | Uncertainty / conformal prediction | Inference uses max of three heads, no interval | Post-hoc on logits | Coverage on public test splits | Medium |
| VI | Fairness slices (age, sex) | Authors flag ethnicity imbalance | Subgroup AUROC in `evaluate.py` | BRSET demographics | Low |
| VII | Distill three backbones → one | Deployment needs three frozen giants | Distillation loss on cached features | Latency vs AUROC on public sets | Medium |
| VIII | Survival (Cox / DeepHit) for 5/10y | Authors call this a limitation | Needs time-to-event labels | **Not public** — UKB only | High |

**Write I+II+III as the paper core. Use IV and VI as required experiments. Keep V/VII as optional. Do not promise VIII without UKB.**

What not to sell as innovation: swapping one backbone without a mechanism; reporting demo AUROC; equating diabetic-retinopathy grade with systemic T2DM screening; beating 0.833 T2DM AUROC without the original cohort.

## 2. What this repo can implement

Map each claim to existing files so the methods section is not vapour.

| Claim | Implement in | How |
|-------|--------------|-----|
| Fixed-quality baseline (Reti-Pioneer) | `model/QualityAware.py`, `configs/default.yaml` | `learnable_q=False` (current default) |
| Learnable quality | same | `learnable_q=True`; optionally predict quality from BRSET labels instead of EyeQ |
| Multi-task logits | `model/RetiPioneer.py` `ComplexModel` | `num_classes = K`; BCE with logits on all labels; shared fusion |
| Independent-head control | `scripts/train.py` | Keep current one-disease loop as ablation |
| Feature extraction | `scripts/extract_features.py` | Run RETF / Swin / Vim on public images → same `UKB_*.npz` layout |
| Public loaders | `dataset/public_fundus.py` | Add ODIR (paired eyes + age), BRSET (laterality + metadata + quality), RFMiD (official splits) |
| No patient leakage | `reti_pioneer/split.py` | Split by **patient id**, not image; both eyes of one person stay in one fold |
| Calibration / DCA | `utils/calibration.py`, `scripts/evaluate.py` | Temperature scaling on val; ECE, Brier, net benefit vs treat-all / treat-none |
| Screening metrics | `utils/metrics.py` | Sensitivity, specificity, PPV, NPV at Youden and at high-NPV operating points |
| Reproducibility | `docs/REPRODUCIBILITY.md`, configs | Seeds, split files `split.npz`, software versions |

Minimal new code (order):

1. `dataset/odir.py`, `dataset/brset.py`, `dataset/rfmid.py` — **implemented** (CSV → records; images not required for tests).
2. `scripts/prepare_public_npz.py` — **implemented** (synthetic features fallback; `SYNTHETIC_FEATURES.txt`).
3. `model/RetiPioneer.py` — **implemented** (`num_classes>1`, `learnable_q`, per-class ensemble).
4. `scripts/train.py` — **implemented** (`--multitask`, `--learnable-q`, `--dataset odir|brset|rfmid|ukb|demo`).
5. `scripts/evaluate.py` — **implemented** (ECE, Brier, net benefit, subgroups, `--calibrate`, `--test-data-dir` / `--test-dataset` for overlapping heads).
6. Ablation configs — **implemented**: `configs/ablation_{baseline,learnq,multitask,full}.yaml`.

Training can stay on **cached features** (`fast_mode: true`). That matches the paper workflow and avoids needing Vision Mamba wheels on Windows.

## 3. Public data you can actually use to verify

These are independently downloadable. Reviewers can repeat the tables.

| Dataset | Size | Overlap with the six diseases | Extra labels / metadata | Split | Access |
|---------|------|-------------------------------|-------------------------|-------|--------|
| **ODIR-5K** | 5,000 patients, paired CFPs | Diabetes (D), Hypertension (H) | Glaucoma, cataract, AMD, myopia, other, age | Official train / off-site / on-site | [Grand Challenge](https://odir2019.grand-challenge.org/dataset/), [Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) |
| **BRSET** | 16,266 images, 8,524 patients | Diabetes diagnosis, hypertensive retinopathy | DR grade, DME, AMD, quality (focus/illumination/field/artifacts), age, sex, nationality, insulin, diabetes duration | Define **patient-level** 70/10/20 and freeze it | [PhysioNet 1.0.2](https://www.physionet.org/content/brazilian-ophthalmological/1.0.2/), paper [10.1371/journal.pdig.0000454](https://doi.org/10.1371/journal.pdig.0000454) |
| **RFMiD / RFMiD 2.0** | 3,200 (2.0 adds more) | Hypertensive retinopathy and other retinal signs | 46–51 conditions, long-tail | Official 1920/640/640 | IEEE DataPort, [Hugging Face](https://huggingface.co/datasets/ctmedtech/RFMID), [MDPI Data 2021](https://www.mdpi.com/2306-5729/6/2/14) |
| APTOS 2019 / MESSIDOR | DR grades only | Proxy for referable DR, **not** systemic T2DM | None useful for six-disease claims | Kaggle / ADCIS | Already sketched in `dataset/public_fundus.py` |
| UK Biobank | Paper primary | All six + 5/10-year incidence | Ethnicity, labs, proteomics | Application | **Not public** |

**Label honesty (write this in Methods):**

- ODIR “Diabetes” is mostly **diabetic retinopathy / diabetic ocular findings**, not UKB ICD-style T2DM. Report it as *ocular-evidence diabetes*, not as a drop-in replica of Reti-Pioneer T2DM AUROC 0.833.
- BRSET has **self-reported diabetes** plus specialist **DR / hypertensive retinopathy** labels — closer to mixed systemic + ocular endpoints. Use both, and never pool them without a table footnote.
- Gout, osteoporosis, hyperlipidemia, thyroid: **no public CFP gold standard**. Either drop them from the public paper or mark them “UKB-only extension”.

**Pretrained weights (public):**

- RETFound: Hugging Face `YukunZhou/RETFound_mae_natureCFP`
- Swin V2-B: torchvision
- EyeQ DenseNet: [HzFu/EyeQ](https://github.com/HzFu/EyeQ)
- Optional: compare RETFound vs ImageNet Swin vs a third public retinal FM (if you add one, it is an ablation, not the main claim)

## 4. Paper outline (write this)

**Abstract (150–200 words).** Problem; Reti-Pioneer limitation (fixed quality, single-task, uncalibrated scores); three changes; three public datasets; main numbers (macro AUROC, ECE, net benefit, ODIR→BRSET drop); one limitation sentence on label semantics.

**1 Introduction.** Oculomics screening; cite Reti-Pioneer and RETFound (Zhou et al., *Nature* 2023); state the three gaps; contributions as a numbered list.

**2 Related work.** Foundation models for CFP; quality assessment (EyeQ); multi-label fundus (ODIR, RFMiD, BRSET); calibration and decision curves in medical AI. Position clearly: *method paper on top of Reti-Pioneer*, not a new foundation model.

**3 Methods.**

3.1 Inputs: paired or single CFP, metadata, quality vector.  
3.2 Frozen backbones and feature cache (same as paper).  
3.3 Quality routing: fixed vs learnable `q_fc`; optional supervised quality on BRSET.  
3.4 Multi-task head: shared `FuseBase`, K logits, BCE (with pos_weight).  
3.5 Ensemble: keep paper’s train-softmax / eval-max as one ablation; add temperature-scaled average.  
3.6 Calibration: temperature scaling on validation; ECE, Brier; decision-curve net benefit.  
3.7 Optional: split-conformal 90% sets.  
3.8 Training details: lr 1e-4, freeze backbones, patient-level split, seeds.

**4 Experiments.**

- Datasets and inclusion (patient-level, missing-eye policy).
- Metrics: per-class AUROC/AP, macro AUROC, ECE, Brier, sensitivity@95% specificity, NPV at paper-style low thresholds, DCA.
- Baselines: linear probe on RETFound only; Reti-Pioneer clone (fixed q, independent heads); each proposed add-on.
- Cross-dataset: train ODIR, test BRSET (hypertension / diabetes-related labels only); reverse.
- Subgroups: age tertiles, sex (BRSET).
- Compute: GPU hours, inference ms (optional distillation).

**5 Results.** Tables first, then one figure: architecture vs Reti-Pioneer; one figure: ablation bars; one figure: calibration + DCA; one figure: cross-domain drop.

**6 Discussion.** What transferred from systemic oculomics to public ocular+systemic labels; what did not; fairness; why gout/osteoporosis cannot be claimed here.

**7 Limitations.** No UKB; label mismatch; no prospective trial; Windows/Vim optional path.

**8 Reproducibility.** Code, configs, checksums, PhysioNet credentialing note for BRSET.

## 5. Experiment protocol (so numbers stay honest)

1. Freeze splits before any training. Save `split.npz` and a CSV of patient IDs.
2. Three seeds (42, 43, 44); report mean ± std and 95% bootstrap CI on the pooled or median seed.
3. Always report the **Reti-Pioneer-clone** on the **same public data** — that is the only fair “improvement” claim.
4. Primary endpoint: **macro AUROC** on the official or frozen test split.
5. Co-primary clinical endpoint: **ECE** and **net benefit at 10% threshold** (screening-like).
6. Pre-register in the paper which labels map to which heads (table in Methods). Do not cherry-pick classes after seeing test AUROC.
7. Never put demo/synthetic AUROC in the paper.

Expected (qualitative) story if the method works: small but consistent AUROC gain from joint training on correlated labels (diabetes–hypertension); larger ECE drop from calibration; quality routing helps more on BRSET (real quality labels) than on ODIR (no quality); cross-domain drop remains large — discuss it, do not hide it.

## 6. Further extensions (papers 2–3)

After the methods paper:

1. **UKB transfer.** Same code, original six diseases, 0/5/10-year labels. Survival models (Cox-PH on fused embedding, or DeepHit). This is the paper that can cite 0.70–0.83 as a baseline to beat.
2. **New targets.** CAD, stroke, CKD, mortality — authors invited this; needs biobank or hospital EHR.
3. **Multimodal.** CFP + OCT (public OCT sets exist but pairing is rare) or CFP + labs.
4. **Test-time adaptation / continual learning** when a new hospital appears.
5. **Single-backbone student** for 6 GB GPU screening (they already mention consumer GPU).
6. **Prospective silent trial** — only with IRB; not a methods-paper claim.

## 7. 12-week writing schedule

| Week | Work | Status (this repo) |
|------|------|--------------------|
| 1 | Download ODIR + RFMiD; apply PhysioNet BRSET; freeze splits; write label-mapping table | **Next (you):** real downloads — see `scripts/download_public_data.py` + [PUBLIC_DATA.md](PUBLIC_DATA.md) |
| 2 | Public dataset loaders + feature extraction to `npz` | Loaders **done**; **bridge done** (`--write-manifest` + `--cache-dir --all-backbones`); GPU real weights still external |
| 3 | Multi-task + learnable_q in `ComplexModel`; unit tests on shapes | **Done** |
| 4 | Train Reti-Pioneer clone + proposed model on ODIR | CLI **done**; real ODIR train awaits features |
| 5 | BRSET + RFMiD; calibration + DCA in evaluate | **Done** (pipeline); real cohorts pending |
| 6 | Cross-dataset and subgroup tables | CLI **done** (`--test-data-dir`) |
| 7 | Ablations (quality off, single backbone, independent heads) | Matrix runner **done** (`scripts/run_ablations.py`) — synthetic only until real caches |
| 8 | Figures; failure cases; quality-weight visualization | Next after real metrics |
| 9 | Draft Methods + Results | Next |
| 10 | Intro, related work, discussion; limitation paragraph | Next |
| 11 | Reproducibility pack; seed reruns | Partial (`docs/REPRODUCIBILITY.md`) |
| 12 | Internal review: every number has a command; submit | Next |

## 8. Claims checklist (before submission)

- [ ] Every AUROC names dataset, split, and label definition.
- [ ] Improvement is vs Reti-Pioneer clone on the same public data, not vs 0.833.
- [ ] Patient-level split; no eye from train patient in test.
- [ ] Calibration reported, not only AUROC.
- [ ] Code and configs match the paper tables.
- [ ] ODIR/BRSET “diabetes” is not called “UKB T2DM”.
- [ ] Demo data never appears in the manuscript.

## 9. Implementation status (this repo)

Engineering is in place for I+II+III plus patient-level split, subgroups, ODIR↔BRSET head mapping, ablation automation, and download helpers. **No clinical AUROC is claimed.** Remaining gaps: actual ODIR/BRSET/RFMiD image downloads, GPU feature extract, EyeQ weights, UKB application.

| Item | Status | How to run |
|------|--------|------------|
| K=1 paper clone + K>1 joint head | Done | `get_reti_pioneer(fast, num_classes=K, learnable_q=...)`; per-class train-softmax / eval-max |
| Learnable quality | Done | `training.learnable_q` / `--learnable-q`; `q_fc` init 1 / 0.5 / 0 |
| Ablation configs | Done | `configs/ablation_{baseline,learnq,multitask,full}.yaml` |
| Ablation matrix runner | Done | `python scripts/run_ablations.py --quick` → `results/ablation_summary.{csv,md}` (**SYNTHETIC**) |
| ODIR / BRSET / RFMiD loaders | Done | `dataset/odir.py`, `brset.py`, `rfmid.py` (CSV fixtures; images optional) |
| Public feature cache | Done | `python scripts/prepare_public_npz.py --dataset odir --synthetic-demo --out data/odir` |
| Extract ↔ prepare bridge | Done | `--write-manifest --require-images` then `extract_features.py --cache-dir … --all-backbones` (or `--stub-identity` / `--dry-run`) |
| Download helper | Done | `python scripts/download_public_data.py` (steps + optional Kaggle/HF; BRSET URL only) |
| Patient-level split | Done | `reti_pioneer.split.patient_level_train_val_indices` |
| Train CLI | Done | `--multitask --learnable-q --dataset odir\|brset\|rfmid\|ukb\|demo` |
| Eval ECE / Brier / DCA / T-scaling | Done | `python scripts/evaluate.py ... --calibrate --split val` |
| Eval JSON export | Done | `--out-json results/metrics.json` |
| Cross-dataset CLI | Done | `--test-data-dir data/brset --test-dataset brset` (overlapping heads only) |
| Tests | Done | `python -m unittest discover -s tests -v` |

```powershell
python -m unittest discover -s tests -v
python scripts/download_public_data.py
python scripts/run_ablations.py --quick
python scripts/train.py --demo --disease t2dm --horizon 0
python scripts/train.py --demo --multitask --learnable-q --horizon 0
python scripts/evaluate.py --demo --disease t2dm --horizon 0 --split val --ckpt ckpt/<run>/t2dm/y0 --calibrate --out-json results/eval.json
python scripts/evaluate.py --ckpt ckpt/<odir-run>/multitask/y0 --dataset odir --data-dir data/odir --test-data-dir data/brset --test-dataset brset
```

**Your next steps for real paper numbers:** (1) download ODIR + credentialed BRSET (+ RFMiD), (2) run `prepare_public_npz` then GPU `extract_features.py`, (3) re-run the ablation matrix on real caches (not `--synthetic-demo`).

See [docs/PUBLIC_DATA.md](PUBLIC_DATA.md) for download links and label-honesty tables.

