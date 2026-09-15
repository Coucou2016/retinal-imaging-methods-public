# Adopted writing architecture

**Emulate:** Nature-family *methods* article (gap → method → fair ablation → calibration/utility as **evaluation** → reproducibility → boundary).

**Title:** Extending Reti-Pioneer with Monotone Quality Routing and Endpoint-Aware Multi-Task Learning

**Spine:** peer-review-aligned `docs/paper/manuscript.md` (post–2026-09-13 rewrite + 2026-09-15 Results alignment).

**Innovation claims (bounded):** monotone bounded quality routing; masked partial-label multitask vs released-code independent loops; endpoint ontology + endpoint-aware cross-cohort evaluation; optional E5 gating.

**Evaluation framework (not novelty):** temperature scaling, ECE, Brier, per-disease DCA (NB@0.10 illustrative only).

**Non-claims:** synthetic AUROC as clinical evidence; beating 0.833 T2DM; ODIR-D as UKB T2DM; `diabetes_related` as clinical head; calibration/DCA as methodological novelty.

## Section map

| Section | Job | Claim ceiling |
|---------|-----|---------------|
| Abstract / Intro | Gap vs fixed-q + released independent heads | Methods extension |
| Methods | monotone router (math), masked BCE, endpoint ontology, E5 optional, disjoint cal | Mechanism + protocol |
| Experiments | E0–E5 + endpoint-aware transfer | Real-data tables 待补充 |
| Results §5.1 | Our computed SYNTHETIC ablation table | Pipeline sanity only |
| Results §5.2 | Clinical real-feature tables | 待补充 |
| Discussion | Published vs released multitask; endpoint drift | Honest novelty only |

## Draft risks

1. SYNTHETIC figures mistaken for clinical evidence.
2. Published “multitask” vs released independent-head loops.
3. Endpoint drift (`diabetes_related` / ocular ≠ UKB ICD).
4. Monotone router without quality supervision may stay near (0, 0.5, 1).
5. Single DCA threshold — keep per-disease curves primary.

## Authenticity

See `docs/paper/AUTHENTICITY_AUDIT.md`.
