# Reti-Pioneer methods extension - FLAT public snapshot

> **Everything in this repository lives in the root directory. There are no
> folders - on purpose.** See `FLAT_REPOSITORY_NOTICE.md` for why and how to
> load it.

Public, folder-free snapshot of a methods extension to **Reti-Pioneer**
(Zhang et al., *Nature Medicine* 2026), *"AI framework for multidisease
detection via retinal imaging"*.  Working title: *Extending Reti-Pioneer with
Monotone Quality Routing and Endpoint-Aware Multi-Task Learning*.

## Read me first

| File | What it is |
|------|------------|
| `FLAT_REPOSITORY_NOTICE.md` | Why this repo has no folders; how to load it into an AI agent |
| `FLAT_FILE_INDEX.md` | Every file: flat name -> original nested path -> role |
| `manuscript.md` / `.html` / `.pdf` | The paper draft (Nature-family methods structure) |
| `report.md` / `.html` / `.pdf` | Standalone research report (self-contained HTML, Base64 figures, no CDN) |
| `AUTHENTICITY_AUDIT.md` | Real-pixel vs synthetic provenance, exact run commands, claim boundaries |
| `methodology_extensions_flat.py` | The three methods contributions, self-contained and self-testing |
| `FLAT_BUILD_MANIFEST.json` | Provenance of this build + SHA-256 for every file |

## What is (and is not) claimed

| Claim | Status |
|-------|--------|
| Monotone bounded quality router (`bad <= usable <= good`) | Methods contribution |
| Masked partial-label multitask vs released-code independent loops | Methods contribution |
| Endpoint ontology + endpoint-aware cross-cohort evaluation | Methods / protocol |
| MultiCohort joint vocabulary (ODIR+BRSET+RFMiD masks) | Methods / protocol |
| E5 quality-conditioned backbone routing (softmax over heads from q) | Optional, config-gated |
| Calibration / ECE / decision curves | Evaluation framework, not novelty |
| RFMiD real-pixel test AUROC (E0-E5) | **Real-pixel**, ImageNet foundation surrogates |
| UKB / clinical AUROCs of the Reti-Pioneer paper (0.699-0.833) | **Not reproduced here** - controlled access |
| ODIR / BRSET clinical results | **Not available** - credentialed access |

Backbone disclosure: RETFound on Hugging Face is gated (HTTP 401 without a
token) and Vision Mamba could not be built on the Windows host, so the RFMiD
feature cache uses ImageNet-pretrained **surrogates** (Swin-V2-B, ViT-B/16 ->
1024-d, ViT-S/16 -> 384-d) over **real fundus pixels**.  See
`FEATURE_PROVENANCE.json` and `AUTHENTICITY_AUDIT.md`.

## Run it (flat layout, from the repository root)

```powershell
python -m pip install -r requirements-flat.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python generate_demo_data.py --out-dir data/UKBCompressed
python train.py --demo --disease t2dm --horizon 0
python evaluate.py --demo --disease t2dm --horizon 0 --split val
python -m unittest discover -s . -p "test_*.py" -v
```

Full real-data pipeline (needs CUDA + RFMiD images):

```powershell
python run_paper_pipeline_flat.py --stages cache train figures report
```

or stage by stage:

```powershell
python build_real_rfmid_cache.py --cache-dir data/rfmid
python run_real_rfmid_experiments.py --data-dir data/rfmid --out-dir results/real_rfmid
python plot_real_results.py
python build_paper_report.py
```

`data/`, `results/` and `ckpt/` are **runtime output directories** created by the
pipeline and gitignored; they keep their original names so the documented
commands remain literally correct.

## License vs research disclaimer

Software copyright: `LICENSE` (MIT, upstream Reti-Pioneer).  Third-party notes:
`THIRD_PARTY_NOTICES.md`.

**Research disclaimer:** not a medical device, not for clinical decisions, and
synthetic-feature metrics are pipeline sanity only - never clinical AUROC.
