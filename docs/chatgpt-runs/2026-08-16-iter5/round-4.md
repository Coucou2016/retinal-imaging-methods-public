# Round 4 — Discussion + limitations + related-work citation check

**Date:** 2026-08-16  
**Mode:** Local WebSearch citation audit + Discussion rewrite. Paste: `PASTE_BRIEF_R4.md`.  
**Chat URL:** unavailable.

## Citation audit

| # | Claimed | Verified? | Notes |
|---|---------|-----------|-------|
| 1 | Reti-Pioneer Nat Med 2026 | Yes | Vol 32, pp 2494–2503 |
| 2 | RETFound Nature 2023 | Yes | Vol 622, pp 156–163 |
| 3 | BRSET PLOS Digit Health | Yes | 3(7):e0000454 |
| 4 | RFMiD Data 2021 | Yes | 6(2):14 |
| 5 | Guo ICML 2017 | Yes | PMLR 70 |
| 6 | Vickers & Elkin 2006 | Yes | MDM |
| 7 | Fu EyeQ MICCAI 2019 | Yes | |
| 8 | Quellec Sci Rep 2023 | Yes | 10.1038/s41598-023-38610-y |
| — | PubMed 38693205 DCA community screening | Keep as “e.g.” until full biblio polish | Do not invent title |

## Discussion improvements applied

- Position vs multi-label ODIR literature (EfficientNet/ViT heads): we are **not** proposing a new fundus backbone; we extend Reti-Pioneer’s frozen-ensemble + quality fusion skeleton.  
- Domain shift: cite population-independent multi-disease work as motivation for patient-level ODIR↔BRSET protocol.  
- Limitations: no UKB/SEED; PhysioNet BRSET blocked without credentials; no Kaggle auth for ODIR; CPU-only → no real foundation features; SYNTHETIC only.

## Rejected

- Adding unverified “first ever quality-aware multitask oculomics” claim.
