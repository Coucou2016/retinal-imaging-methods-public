# Adopted writing architecture

**Emulate:** Nature-family *methods* article (argument: gap → method → fair ablation → calibration/utility → reproducibility → boundary). Closest venue archetypes without UKB: *npj Digital Medicine* methods / *MedIA* technical papers — not flagship *Nat Med* clinical discovery.

**Do not emulate:** flagship *Nature Medicine* clinical discovery narrative requiring UKB-scale cohorts and prospective pilots as primary claims.

**Innovation claims (bounded):** learnable quality routing; shared multi-task head; calibration + DCA as co-primary; public ODIR/BRSET/RFMiD patient-level validation.

**Non-claims:** synthetic AUROC; beating 0.833 T2DM; ODIR-D as UKB T2DM.

## Section map (bounded novelty)

| Section | Job | Claim ceiling |
|---------|-----|---------------|
| Abstract / Intro | Gap vs fixed-q + independent heads + AUROC-only utility | Methods extension of Reti-Pioneer skeleton |
| Related work | RETFound, EyeQ, BRSET/ODIR/RFMiD, calibration/DCA | Position as follow-up, not new foundation model |
| Methods | `learnable_q`, multi-label BCE, temp scaling, patient splits | Mechanism + protocol only |
| Experiments | Ablation arms + transfer + fairness slots | Real-data tables 待补充 |
| Results | Empty / deferred | No SYNTHETIC numbers in submission tables |
| Discussion / Limits | Label mismatch, domain shift, no UKB | Honest novelty phrasing only |

## Draft risks (keep visible)

1. SYNTHETIC companion figures mistaken for clinical evidence.  
2. Straw-man vs Reti-Pioneer if “multitask” wording in their abstract is not reconciled with independent-head code clone.  
3. Endpoint drift (ocular D/H ≠ UKB ICD).  
4. Unsupervised learnable quality may not move off (1, 0.5, 0).  
5. Single DCA threshold without prevalence justification.
