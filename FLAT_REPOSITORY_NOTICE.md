# FLAT_REPOSITORY_NOTICE

**This repository intentionally contains no directories. Every file - code,
configs, the paper, the report, figures, metrics and provenance - sits directly
in the repository root.**

---

## 1. Why

The primary consumer of this snapshot is not a human clicking through a GitHub
tree; it is an **AI agent** (ChatGPT with browsing, another coding agent, an
automated reviewer) that has to load *everything* and cross-check code against
claims against results.

Nested layouts make that expensive and error-prone:

* Path depth must be guessed or discovered before a single file can be fetched.
* Agents routinely stop at the first level, so `manuscript.md`,
  `AUTHENTICITY_AUDIT.md` or `metrics_E5_rfmid_test.json` get missed.
* Cross-references inside documents have to be resolved against an assumed
  root; a wrong guess silently yields the wrong file.
* A flat listing is one cheap round-trip, with no recursion and nothing hidden.

With a flat tree an agent can be sure it saw everything:

```text
1. GET /repos/<owner>/<repo>/contents   -> the complete file list (one level)
2. fetch each file                      -> complete corpus
3. cross-check manuscript <-> metrics JSON <-> code <-> configs
```

This is deliberate, not an accident of packaging. Where a filename looks like it
came from a folder (for example `upstream_model_mamba.py`), the directory
structure was folded into the filename to keep the tree one level deep.

---

## 2. What was flattened

| Original nested path | Flat path here |
|----------------------|----------------|
| `docs/paper/manuscript.{md,html,pdf}` | `manuscript.{md,html,pdf}` |
| `docs/report/report.{md,html,pdf}` | `report.{md,html,pdf}` |
| `docs/paper/AUTHENTICITY_AUDIT.md`, `docs/paper/OUTLINE.md` | `AUTHENTICITY_AUDIT.md`, `OUTLINE.md` |
| `docs/*.md` | `DATA.md`, `DATA_BLOCKERS.md`, `PAPER.md`, `PAPER_PLAN.md`, `PUBLIC_DATA.md`, `REPRODUCIBILITY.md` |
| `docs/paper_assets/figN_*.{png,pdf}` | `figN_*.{png,pdf}` |
| `results/real_rfmid/figures/figN_*` | `figN_*` |
| `results/real_rfmid/metrics_E*_rfmid_{val,test}.json` | `metrics_E*_rfmid_{val,test}.json` |
| `results/real_rfmid/ablation_summary.csv`, `run_manifest.json`, `test_predictions.npz` | same names, root |
| `model/`, `scripts/`, `dataset/`, `utils/`, `reti_pioneer/`, `tests/` | filenames only: `QualityAware.py`, `train.py`, `rfmid.py`, `calibration.py`, `label_map.py`, `test_smoke.py`, ... |
| `configs/*.yaml` | `ablation_e0.yaml`, `default.yaml`, ... |
| `data/rfmid/{pairs.csv,label_map.json,split.npz,FEATURE_PROVENANCE.json,UKB_mqd,y0,y5,y10.npz}` | same names, root |
| `Reti-Pioneer-main/**` (upstream reference clone) | `upstream_*` (see `FLAT_FILE_INDEX.md`) |
| `.github/workflows/ci.yml` | `ci.yml` |

`FLAT_FILE_INDEX.md` is the authoritative per-file map.

---

## 3. The code was made to actually run flat

Renaming files alone would have produced an un-runnable tree, because the source
uses package-qualified imports. Those were rewritten mechanically:

| Before | After |
|--------|-------|
| `from model.RetiPioneer import get_reti_pioneer` | `from RetiPioneer import get_reti_pioneer` |
| `from dataset.UKBDataset import UKBDatasetFast` | `from UKBDataset import UKBDatasetFast` |
| `from utils.calibration import fit_temperature` | `from calibration import fit_temperature` |
| `from reti_pioneer.split import load_split` | `from split import load_split` |
| `from scripts.evaluate import collect_probs` | `from evaluate import collect_probs` |

Each file also gets a bootstrap block, so modules resolve their flat siblings no
matter how the entry point was launched:

```python
import os as _os, sys as _sys
_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
```

Because `configs/`, `scripts/` and `docs/` no longer exist, ROOT-anchored path
literals that pointed into them were shortened to the root:

| Before | After |
|--------|-------|
| `ROOT / "configs" / "default.yaml"` | `ROOT / "default.yaml"` |
| `os.path.join(ROOT, "scripts", "train.py")` | `os.path.join(ROOT, "train.py")` |
| `ASSETS = ROOT / "docs" / "paper_assets"` | `ASSETS = ROOT` |

**Deliberately unchanged:** `data/`, `results/` and `ckpt/` remain *runtime*
directories. The pipeline creates them on demand and they are gitignored, so the
published CLI flags (`--cache-dir data/rfmid`, `--out-dir results/real_rfmid`)
keep working verbatim and the run commands quoted in `AUTHENTICITY_AUDIT.md`
stay literally true.

### Path mapping for cross-checking

If a string mentions an original nested path, resolve it like this:

| If you read | The shipped file is |
|-------------|---------------------|
| `docs/paper/X` or `docs/X` | `X` at the root |
| `configs/X.yaml` | `X.yaml` at the root |
| `scripts/X.py` | `X.py` at the root |
| `results/real_rfmid/<file>` | `<file>` at the root |
| `data/rfmid/<file>` | `<file>` at the root |
| `model/X.py`, `dataset/X.py`, `utils/X.py`, `reti_pioneer/X.py`, `tests/X.py` | `X.py` at the root |
| `Reti-Pioneer-main/a/b.py` | `upstream_a_b.py` |
| `results/real_rfmid/ckpts/**` | not shipped (regenerate) |

The **complete original nested tree is archived intact on the
`nested-archive-2026-09` branch** of this repository.

---

## 4. Recipe: load the whole repository into an agent

```text
GET  /repos/Coucou2016/retinal-imaging-methods-public/contents
     -> complete flat file list (no recursion needed)
GET  .../contents/<name>              -> that file

Suggested reading order
  1. FLAT_REPOSITORY_NOTICE.md   (this file)
  2. FLAT_FILE_INDEX.md          (map of everything)
  3. AUTHENTICITY_AUDIT.md       (what is real, what is not)
  4. manuscript.md               (the paper)
  5. report.md                   (the report)
  6. metrics_E*_rfmid_test.json  (numbers behind the tables)
  7. ablation_e*.yaml + train.py + evaluate.py + QualityAware.py
  8. test_*.py                   (what is actually enforced)
```

Machine-readable provenance for this build: `FLAT_BUILD_MANIFEST.json`
(copied files, generated files, SHA-256 per file, exclusions and their reasons).

---

## 5. What is deliberately NOT here

| Excluded | Why |
|----------|-----|
| `*.pt` / `*.pth` checkpoints (~2 GB) | Model weights; over GitHub limits, reproducible from configs |
| `data/rfmid/UKB_{RETF,swin,vim}.npz` (~57 MB) | Real-pixel feature caches; rebuild with `build_real_rfmid_cache.py` |
| `results/ablation_runs/**` | Throwaway synthetic run trees |
| `results/ablation_summary.{csv,md}` (root copies) | **SYNTHETIC** ODIR numbers, superseded by real-pixel RFMiD |
| `metrics_*_odir_*.json` (root) | **SYNTHETIC** feature-cache metrics (not clinical) |
| `data/raw/**`, `ckpt/`, `pretrained/`, `models/` | Downloaded corpora and weights |
| UK Biobank tensors, patient fundus images | Controlled-access data; never redistributed |
| `docs/chatgpt-runs/**` | Internal agent-collaboration logs (full copy on `nested-archive-2026-09`) |

---

## 6. Citing a flat file

Every file is one level deep, so a citation is just the filename:

> See `manuscript.md` section 5, `metrics_E5_rfmid_test.json`, and
> `ablation_e5.yaml`.

No `../` chains, no ambiguous roots.
