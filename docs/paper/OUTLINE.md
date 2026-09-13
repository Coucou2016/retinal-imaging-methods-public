# Adopted writing architecture

**Emulate:** Nature-family *methods* article (gap → method → fair ablation → calibration/utility as **evaluation** → reproducibility → boundary).

**Title direction:** Extending Reti-Pioneer with Monotone Quality Routing and Endpoint-Aware Multi-Task Learning

**Innovation claims (bounded):** monotone bounded quality routing; masked partial-label multitask vs released-code independent loops; endpoint ontology + endpoint-aware cross-cohort evaluation.

**Evaluation framework (not novelty):** temperature scaling, ECE, Brier, per-disease DCA (NB@0.10 illustrative only).

**Non-claims:** synthetic AUROC; beating 0.833 T2DM; ODIR-D as UKB T2DM; `diabetes_related` as clinical head; calibration/DCA as methodological novelty.

## Section map (bounded novelty)

| Section | Job | Claim ceiling |
|---------|-----|---------------|
| Abstract / Intro | Gap vs fixed-q + released independent heads | Methods extension |
| Methods | monotone router, masked BCE, endpoint ontology, disjoint cal | Mechanism + protocol |
| Experiments | E0–E4 + endpoint-aware transfer | Real-data tables 待补充 |
| Results | Empty / deferred | No SYNTHETIC in submission tables |
| Discussion | Published vs released multitask; endpoint drift | Honest novelty only |

## Draft risks (keep visible)

1. SYNTHETIC companion figures mistaken for clinical evidence.  
2. Published “multitask” vs released independent-head loops.  
3. Endpoint drift (`diabetes_related` / ocular ≠ UKB ICD).  
4. Monotone router without quality supervision may stay near (0, 0.5, 1).  
5. Single DCA threshold — keep per-disease curves primary.
