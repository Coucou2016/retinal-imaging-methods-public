# Audit notes — 2026-08-15

## Defects found (feature-cache path)

| ID | Severity | Issue | Disposition |
|----|----------|-------|-------------|
| D1/D2 | High | `train.py` ignored `data_dir/split.npz` / never wrote `test_idx` | **Fixed** — consume frozen split; copy `test_idx`; public 3-way when no frozen split |
| D3 | High | Ablation summary compared macro-K vs single-D AUROC | **Fixed** — add `auroc_D` comparable column + metric note |
| D4 | High | `multitask`/`full` share tag; `find_latest_run` mtime-only | **Fixed** — filter by `run_meta.learnable_q` |
| D5 | Medium | DCA net benefit clamped ≥0 | **Fixed** (already present / confirmed) — raw NB; treat-all / treat-none helpers |
| D6 | Medium | Eval fallback was sample-level stratified | **Fixed** — patient-level fallback |
| D7 | Medium | Public train via `default.yaml` → `val_fraction=0` | **Fixed** — public/demo default val_fraction 0.25 |
| D8 | Medium | RFMiD mapped to nonexistent `HR` | **Fixed** — removed false RFMiD HTN mapping; warn on skip |
| D9 | Medium | `prepare_public_npz --force` could wipe real features | **Fixed** — `--labels-only` / `--force-features` guard |
| D10 | Low | No torch/numpy seed | **Fixed** — seed from split config |
| ENV | — | Disk full on C: broke tests | **Mitigated** — cleaned TEMP/caches |

## ChatGPT

Browser MCP could not hold a ChatGPT tab. Task brief preserved in `TASK_BRIEF.md` for manual paste when browser works.

## Remaining external blockers (not code)

- Real ODIR / PhysioNet BRSET / RFMiD downloads
- GPU `extract_features.py` on real images
- EyeQ weights (raw-image quality path only)
- UKB application (out of scope for public paper)
