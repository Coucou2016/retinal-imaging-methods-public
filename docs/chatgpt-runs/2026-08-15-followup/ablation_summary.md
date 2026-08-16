# Ablation summary (SYNTHETIC)

> **SYNTHETIC FEATURE CACHE - not for manuscript AUROC. Replace with real ODIR/BRSET images + extract_features.py before paper tables.**

Primary comparable column: **auroc_D** (ODIR D head). `auroc` is macro-K for multitask arms.

Generated: 2026-08-15T22:43:54

| Ablation | Eval | Status | AUROC | AUROC_D | AP | ECE | Brier | NB@0.10 | Treat-all | Cal AUROC | Cal ECE | T | n |
|----------|------|--------|-------|---------|----|-----|-------|---------|-----------|-----------|---------|---|---|
| baseline | odir_val | ok | 0.4688 | 0.4688 | 0.3930 | 0.3389 | 0.3674 | 0.2685 | 0.2593 | 0.4688 | 0.1838 | 10.0000 | 12 |
| baseline | odir_to_brset | ok | 0.4988 | 0.4988 | 0.3340 | 0.3720 | 0.3111 | 0.0926 | 0.1435 | 0.4988 | 0.2610 | 10.0000 | 48 |
| learnq | odir_val | ok | 0.4688 | 0.4688 | 0.3930 | 0.3389 | 0.3674 | 0.2685 | 0.2593 | 0.4688 | 0.1838 | 10.0000 | 12 |
| learnq | odir_to_brset | ok | 0.4988 | 0.4988 | 0.3340 | 0.3720 | 0.3111 | 0.0926 | 0.1435 | 0.4988 | 0.2610 | 10.0000 | 48 |
| multitask | odir_val | ok | 0.3797 | 0.4688 | 0.3295 | 0.6472 | 0.5735 | 0.0880 | 0.1088 | 0.3797 | 0.3703 | 9.9999 | 12 |
| multitask | odir_to_brset | ok | 0.4152 | 0.3611 | 0.2683 | 0.4797 | 0.4332 | 0.0567 | 0.0856 | 0.4152 | 0.3378 | 9.9999 | 48 |
| full | odir_val | ok | 0.3797 | 0.4688 | 0.3295 | 0.6472 | 0.5735 | 0.0880 | 0.1088 | 0.3797 | 0.3703 | 9.9999 | 12 |
| full | odir_to_brset | ok | 0.4152 | 0.3611 | 0.2683 | 0.4797 | 0.4332 | 0.0567 | 0.0856 | 0.4152 | 0.3378 | 9.9999 | 48 |

CSV: `E:\Projects\20260522-retinal-imaging\results\ablation_summary.csv`
