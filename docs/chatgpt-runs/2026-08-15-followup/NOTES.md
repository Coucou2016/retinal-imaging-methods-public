# Follow-up notes — 2026-08-15

## ChatGPT

**URL:** none (handoff failed).

Browser MCP still cannot hold a tab: create returns a `viewId`, then list is empty / navigate reports `Browser view not found` or `No browser tab available`. Evidence: `BROWSER_FAILURE.md`. Manual brief: `TASK_BRIEF.md`.

## Local engineering landed

1. **Youden + sens@95%spec** in `utils/calibration.py`; wired into `scripts/evaluate.py` `score_predictions` / console / `--out-json` (`youden_*`, `sens@95%spec*`).
2. **Cross-eval K=1 bugfix:** `collect_probs` no longer flattens multi-column test labels when model outputs a singleton class (was `n*K` vs `n` → ablation ODIR→BRSET failures for baseline/learnq).
3. Unit tests for operating points + K=1 multilabel collect; lightly hardened flaky learnable-q grad test.

## Ablation smoke

```text
python scripts/run_ablations.py --quick
→ Done: 8/8 eval rows ok (SYNTHETIC)
```

Artifacts: `ablation_quick_stdout.txt`, `ablation_summary.md`, `ablation_summary.csv` (copied here). AUROCs are synthetic-only — not for manuscript.

## Tests

```text
python -m unittest discover -s tests -v
→ Ran 32 tests … OK
```

Log: `unittest-final.txt`.

## ZIP

No new ZIP rebuilt this turn. Prior post-patch ZIP remains:

- `artifacts/chatgpt-handoff/reti-pioneer-followup-postpatch-20260815-223100.zip`
- SHA-256 `3D2663273E07C4160F6854017E057CC68A09C2C09BCF6E9FCCAE342B32D1DBDF`

## Unverified / remaining risks

- Real ODIR / BRSET / RFMiD downloads + GPU `extract_features.py` still external
- Extract↔prepare bridge beyond `--labels-only` guard not fully e2e on real images
- ChatGPT never reviewed these patches
- Operating points on tiny synthetic val (n≈12) are noisy; `sens@95%spec` may sit on the FPR=0 corner
- No git commit/push (local-only workspace)

## Local-only status

All changes uncommitted; no PR.
