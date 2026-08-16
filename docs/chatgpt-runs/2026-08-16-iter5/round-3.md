# Round 3 — Results honesty / figure design critique

**Date:** 2026-08-16  
**Mode:** Local structured critique + figure regen plan. Paste: `PASTE_BRIEF_R3.md`.  
**Chat URL:** unavailable.

## Honesty contract (accepted)

1. Manuscript §Results contains **no** numeric AUROC/ECE tables from synthetic caches.  
2. Companion report may show SYNTHETIC tables/figures with red badges and deep captions explaining *why* values are near chance (tiny n, label-driven synthetic features).  
3. Hypotheses about “what real features may show” remain labeled **hypothesis, not claim**.  
4. Forbidden: comparing SYNTHETIC AUROC to Reti-Pioneer 0.699–0.833.

## Figure design checklist

| Fig | Role | Caption must say |
|-----|------|------------------|
| 1 | Information-flow schematic | Not a performance claim; yellow=learnable_q; green=multitask+cal/DCA |
| 2 | Ablation AUROC bars | **SYNTHETIC**; n tiny; pipeline sanity only |
| 3 | ECE before/after temperature | Calibration can move ECE without changing AUROC |
| 4 | Cross-dataset drop | Protocol figure for ODIR→BRSET head-mapped transfer on SYNTHETIC |

## Applied

- Regenerated SciencePlots figures via `scripts/plot_paper_figures.py`.  
- Strengthened manuscript Results + report figure 来龙去脉 captions.  
- Kept Times New Roman / science style; English axes.
