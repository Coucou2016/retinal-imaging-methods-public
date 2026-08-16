# Task brief for ChatGPT (manual paste if browser MCP blocked)

**Title:** Reti-Pioneer follow-up: audit & advance

## Background / goals

Nature Medicine Reti-Pioneer reproduction + follow-up methods paper (learnable quality, multitask head, calibration/DCA, public ODIR↔BRSET). Keep `fast_mode` feature-cache path. No clinical AUROC from synthetic data.

ZIP attached (or path below). You cannot see the local filesystem otherwise.

## ZIP baseline

- Path: `E:\Projects\20260522-retinal-imaging\artifacts\chatgpt-handoff\reti-pioneer-followup-audit-20260815-220501.zip`
- SHA-256: `FBF0EFB667EACE280C04ED3ED988DECD789302FE1BBBA8F2E723EEF0293D780F`
- Size: 96514 bytes
- Baseline: no-git workspace snapshot 2026-08-15

## Architecture & non-negotiable boundaries

- Train/eval on cached foundation features (`fast_mode: true`)
- Patient-level splits; no eye leakage
- ODIR D ≠ UKB T2DM; honesty footnotes required
- Do not expand to UKB without data; do not require EyeQ for this milestone
- No git push; no speculative backbone rewrites

## Scope (if still needed after lead patches)

Lead agent already applied D1–D10 fixes locally after this ZIP was packed. If reviewing this ZIP, verify and propose only remaining minimal patches, especially:

1. Confirm frozen `split.npz` consumption + `test_idx` round-trip
2. Ablation metrics comparable on ODIR D
3. DCA raw NB + treat-all/none
4. `prepare` ↔ extract bridge without clobbering real features
5. Optional: Youden / sens@95%spec wiring in evaluate

## Deliverables

- Patch files or full file contents
- Short report of defects found/fixed
- Exact test commands

## Required tests

```powershell
python -m unittest discover -s tests -v
```

## Forbidden claims / ops

- Clinical AUROC on synthetic caches
- Equating ODIR D / BRSET diabetes with UKB T2DM 0.833
- git commit/push/PR
- UKB survival claims without data

## Acceptance criteria

- Frozen splits honored; `--split test` works when `test_idx` present
- Ablation summary documents macro vs `auroc_D`
- DCA can be negative; treat-all/none reported
- Labels-only prepare does not overwrite real backbone npz
- Unittest suite green
