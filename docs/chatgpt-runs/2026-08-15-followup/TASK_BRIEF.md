# Task brief for ChatGPT (follow-up — remaining gaps)

**Title:** Reti-Pioneer follow-up: Youden ops + ablation verify + extract bridge

## Background

Prior audit ZIP patches (D1–D10) already landed locally. Unittest was 25/25 green before this follow-up. ChatGPT browser MCP still broken — this brief is for manual paste.

## Post-patch ZIP (preferred baseline)

- Path: `E:\Projects\20260522-retinal-imaging\artifacts\chatgpt-handoff\reti-pioneer-followup-postpatch-20260815-223100.zip`
- SHA-256: `3D2663273E07C4160F6854017E057CC68A09C2C09BCF6E9FCCAE342B32D1DBDF`

## Remaining scope (minimal)

1. **Youden + sens@95%spec** in `scripts/evaluate.py` / metrics helpers; include in `--out-json`; unit tests on synthetic probs.
2. Verify `python scripts/run_ablations.py --quick` (or `--quick --no-cross`) and that summary documents `auroc_D` vs macro.
3. Confirm `prepare_public_npz` ↔ `extract_features` bridge does not clobber real backbone npz (`--labels-only` / `--force-features`).
4. Do **not** claim clinical AUROC on synthetic caches.

## Required tests

```powershell
python -m unittest discover -s tests -v
python scripts/run_ablations.py --quick
```

## Forbidden

- Clinical AUROC on synthetic data
- Equating ODIR D / BRSET diabetes with UKB T2DM 0.833
- git commit/push/PR
