# Paste brief — Round 2 (Methods critique)

**GitHub:** https://github.com/Coucou2016/retinal-imaging-methods-public

Web search ON. Critique **Methods only** for a Reti-Pioneer methods extension (learnable_q + multitask + calibration/DCA + public cohorts).

## Key excerpts (paste as-is)

**Quality routing:** With `learnable_q=False`, `q_fc` is frozen at (1, 0.5, 0). With `learnable_q=True`, same init but trainable. Optional supervised quality on BRSET focus/illumination/artifact labels is deferred (待补充).

**Multi-task:** `ComplexModel` supports `num_classes=K`. Joint BCE shares fused representation. Independent single-disease training remains the Reti-Pioneer-clone control. Note: Reti-Pioneer abstract says “multitask” but released code uses independent binary heads — state this explicitly.

**Calibration:** Temperature T fit on validation logits; report AUROC/AP, ECE, Brier, sensitivity@high-spec, net benefit @0.10 vs treat-all/none.

**Label map:** ODIR-D / BRSET diabetes are **not** UKB T2DM ICD.

## Ask
1. What is underspecified for reproducibility?  
2. Any straw-man vs Reti-Pioneer?  
3. What must be added before submission without real AUROCs?
