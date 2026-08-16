# ChatGPT task brief — literature + Nature-style outline (text paste only)

Paste into a new ChatGPT chat. Enable web search if the UI offers it. Do **not** upload files.

---

You are an academic advisor for a methods follow-up paper on Reti-Pioneer.

**Baseline paper:** Zhang et al., AI framework for multidisease detection via retinal imaging. Nature Medicine (2026). DOI 10.1038/s41591-026-04359-w.

**What Reti-Pioneer already did:** frozen RETFound + Swin V2-B + Vision Mamba-S; fixed quality weights (good=1, usable=0.5, bad=0); bilinear fusion with metadata; one binary head per disease; soft voting train / max inference; UKB+hospital data; SEED external validation; silent trial.

**Author-stated gaps:** accuracy still below broad clinical adoption; longitudinal tasks are binary 5/10y cutoffs not survival; scope excludes CAD/stroke/mortality/rare disease; residual confounding / ethnicity imbalance; need larger RCTs.

**Our proposed innovations (public-data feasible, no UKB yet):**
1. Learnable quality routing (vs frozen q_fc)
2. Shared multi-task / multi-label head (vs six independent binary models)
3. Calibration (temperature scaling, ECE, Brier) + decision-curve net benefit as co-primary utility
4. Patient-level domain generalization on public ODIR ↔ BRSET ↔ RFMiD
5. Fairness slices (age, sex) on BRSET

**Honesty constraints:** ODIR “Diabetes” ≠ UKB T2DM ICD; do not invent clinical AUROCs; synthetic demo AUROCs are pipeline sanity only; do not claim beating 0.833 T2DM AUROC without original cohort.

**Ask you to return:**
1. A compact literature set (8–15 papers) covering: Reti-Pioneer/oculomics, RETFound, EyeQ/quality, multi-label fundus (ODIR/RFMiD/BRSET), calibration/DCA in medical AI, domain shift.
2. Which Nature-family or digital-medicine article architecture to emulate for a *methods* paper (not flagship Nature Medicine clinical claim).
3. A section-by-section outline with credible, bounded novelty claims.
4. How to phrase novelty honestly vs Reti-Pioneer without overclaiming.
5. Which claims to defer until UKB access.

Keep answers concrete and citation-grounded. Prefer peer-reviewed venues.
