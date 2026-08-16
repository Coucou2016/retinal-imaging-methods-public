# ChatGPT task brief — public GitHub read + literature/writing (text paste)

Paste into a **new** ChatGPT chat. Enable **web search** if the UI offers it.  
Do **not** upload ZIP/files — read the public repository URL below instead.

---

## Public repository (read this)

**GitHub URL (fill after push):** `GITHUB_URL_PLACEHOLDER`

Please open and skim:
- `README.md`
- `docs/PAPER_PLAN.md`, `docs/paper/OUTLINE.md`, `docs/paper/manuscript.md`
- `docs/report/report.md` (or `report.html`)
- `model/QualityAware.py`, `model/RetiPioneer.py`
- `scripts/run_ablations.py`, `scripts/plot_paper_figures.py`
- `configs/ablation_*.yaml`

Ignore any SYNTHETIC AUROC as clinical performance.

---

## Project context

You are an academic advisor for a **methods follow-up** on Reti-Pioneer.

**Baseline:** Zhang et al., AI framework for multidisease detection via retinal imaging. *Nature Medicine* (2026). DOI 10.1038/s41591-026-04359-w.

**Reti-Pioneer already did:** frozen RETFound + Swin V2-B + Vision Mamba-S; fixed quality weights (good=1, usable=0.5, bad=0); bilinear fusion + metadata; one binary head per disease; soft voting train / max inference; UKB+hospital; SEED external; silent trial.

**Our implemented extensions (public-data feasible):**
1. Learnable quality routing (`learnable_q`) vs frozen `q_fc`
2. Shared multi-task / multi-label head vs independent binary models
3. Calibration (temperature scaling, ECE, Brier) + DCA net benefit as co-primary utility
4. Patient-level domain generalization protocol on ODIR ↔ BRSET ↔ RFMiD
5. Fairness slices (age, sex) on BRSET (evaluator ready; real numbers 待补充)

**Honesty constraints:**
- ODIR “Diabetes” ≠ UKB T2DM ICD
- Do **not** invent clinical AUROCs or UKB numbers
- Synthetic demo AUROCs in `results/ablation_summary.csv` / report figures are **pipeline sanity only**
- Do not claim beating UKB T2DM AUROC 0.833 without that cohort

---

## Please return

1. Compact literature set (8–15 peer-reviewed items): Reti-Pioneer/oculomics, RETFound, EyeQ/quality, multi-label fundus (ODIR/RFMiD/BRSET), calibration/DCA, domain shift.
2. Which Nature-family / digital-medicine **methods** article architecture to emulate (not flagship Nat Med clinical claim).
3. Section-by-section outline with **bounded** novelty claims.
4. Honest novelty phrasing vs Reti-Pioneer (what to claim / what to defer until UKB).
5. Concrete edits to strengthen `docs/paper/manuscript.md` and the Chinese research report figure captions (来龙去脉), without inventing results.

Keep answers citation-grounded and concrete.
