# Reti-Pioneer — multidisease detection via retinal imaging

Reproduction workspace for the **Nature Medicine (2026)** paper [*AI framework for multidisease detection via retinal imaging*](https://doi.org/10.1038/s41591-026-04359-w) (**Reti-Pioneer**).

This project integrates the [official Reti-Pioneer release](https://github.com/lyhyl/Reti-Pioneer) with local scripts, configuration, demo data, public-dataset adapters, and documentation for end-to-end training and evaluation when UK Biobank / hospital cohorts are unavailable.

## Diseases screened (6)

Type 2 diabetes, hypertension, hyperlipidemia, gout, osteoporosis, thyroid disease — from bilateral color fundus photographs plus clinical metadata.

See [docs/PAPER.md](docs/PAPER.md) for architecture details, [docs/DATA.md](docs/DATA.md) for data access, [docs/PUBLIC_DATA.md](docs/PUBLIC_DATA.md) for ODIR/BRSET/RFMiD mapping, and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for seeds, demo vs production, and expected metric ranges.

## Quick start (demo, CPU-friendly)

```powershell
cd E:\Projects\20260522-retinal-imaging
python -m pip install -r requirements.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python scripts/generate_demo_data.py
python scripts/train.py --demo --disease t2dm --horizon 0
python scripts/evaluate.py --demo --disease t2dm --horizon 0 --split val --ckpt ckpt/<run>/t2dm/y0
python -m unittest discover -s tests -v
```

Training uses **pre-extracted foundation features** (`fast_mode: true`), matching the authors' efficient fine-tuning workflow.

`--demo` uses 3 epochs on CPU with a **stratified train/val split** (25% held out; see `configs/demo.yaml`). Demo AUROC is for pipeline sanity only — not comparable to paper clinical AUROCs ([docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)).

On Windows, if `import torchvision` fails with `_lzma`, use Python 3.12 with full stdlib or install [liblzma](https://docs.python.org/3/using/windows.html); feature-cache training does not require torchvision.

## Training (all diseases / horizons)

```powershell
python scripts/train.py --config configs/default.yaml
# or single task:
python scripts/train.py --disease hypertension --horizon 5
```

Checkpoints: `ckpt/<timestamp>/<disease>/y<horizon>/`.

## Ablations (follow-up methods)

Configs keep the Reti-Pioneer **feature-cache** path (`fast_mode: true`). `--demo` forces 3 CPU epochs and will not overwrite existing `UKB_*.npz` unless `--regenerate-data`.

```powershell
python scripts/train.py --config configs/ablation_baseline.yaml --demo --disease t2dm --horizon 0
python scripts/train.py --config configs/ablation_learnq.yaml --demo --disease t2dm --horizon 0
python scripts/train.py --config configs/ablation_multitask.yaml --demo --multitask --horizon 0
python scripts/train.py --config configs/ablation_full.yaml --demo --multitask --horizon 0
python scripts/evaluate.py --demo --disease t2dm --horizon 0 --split val --ckpt ckpt/<run>/t2dm/y0 --calibrate
python scripts/evaluate.py --demo --split val --ckpt ckpt/<run>/multitask/y0 --calibrate --out-json results/metrics.json
```

### Ablation matrix automation (synthetic CI)

Builds tiny synthetic ODIR + BRSET caches if missing, trains the four ablation arms with short epochs, evaluates with `--calibrate` (and optional ODIR→BRSET cross), writes `results/ablation_summary.csv` + `.md`. Tables are marked **SYNTHETIC** — not for manuscript AUROC.

```powershell
python scripts/run_ablations.py --quick
# one arm only (faster smoke):
python scripts/run_ablations.py --quick --no-cross --only baseline
```

Download helper (prints steps; optional Kaggle/HF if credentials exist; BRSET PhysioNet URL only):

```powershell
python scripts/download_public_data.py
python scripts/download_public_data.py --try-download
```

- `ablation_baseline.yaml` — fixed quality weights, independent/single-task (paper clone)
- `ablation_learnq.yaml` — `training.learnable_q: true`
- `ablation_multitask.yaml` — joint K-class BCE head
- `ablation_full.yaml` — learnable quality + multitask + calibration in eval

CLI flags `--multitask` and `--learnable-q` override the same keys. Ensemble: paper train-softmax / eval-max **per class** when K>1 (alternative: `--ensemble mean` or `temp_mean`).

## Public datasets (ODIR / BRSET / RFMiD)

See [docs/PUBLIC_DATA.md](docs/PUBLIC_DATA.md) for download links, label mapping, and honesty notes (ODIR-D is not UKB T2DM). Gout / osteoporosis / hyperlipidemia / thyroid are omitted on public sets.

```powershell
python scripts/prepare_public_npz.py --dataset odir --synthetic-demo --out data/odir --force
python scripts/train.py --config configs/ablation_full.yaml --dataset odir --data-dir data/odir --multitask --horizon 0
python scripts/evaluate.py --config configs/ablation_full.yaml --dataset odir --data-dir data/odir --ckpt ckpt/<run>/multitask/y0 --split val --calibrate

# Cross-dataset: train on ODIR cache, score overlapping heads on BRSET (not identical labels)
python scripts/evaluate.py --ckpt ckpt/<odir-run>/multitask/y0 --dataset odir --data-dir data/odir --test-data-dir data/brset --test-dataset brset
```

`--heads` defaults to `diabetes_related,hypertension_ocular`. ODIR **D** maps to BRSET **diabetes**; ODIR **H** maps to **hypertensive_retinopathy**. These are related screening targets, not UKB ICD endpoints. See [docs/PUBLIC_DATA.md](docs/PUBLIC_DATA.md).

`--synthetic-demo` writes **synthetic** features from labels for CI. Real images:

```powershell
python scripts/prepare_public_npz.py --dataset odir --src <raw> --out data/odir --labels-only --force --write-manifest --require-images
python scripts/extract_features.py --cache-dir data/odir --dry-run
python scripts/extract_features.py --cache-dir data/odir --all-backbones
```

Legacy single-file form still works: `python scripts/extract_features.py --manifest pairs.csv --out data/odir/UKB_RETF.npz --backbone retf`.

## Evaluation

```powershell
python scripts/evaluate.py --demo --disease t2dm --horizon 0 --split val --ckpt ckpt/<run>/t2dm/y0
python scripts/evaluate.py --demo --disease t2dm --horizon 0 --split val --ckpt ckpt/<run>/t2dm/y0 --calibrate --out-json results/eval.json
```

`--split` can be `train`, `val`, `test`, or `all`. Evaluation reuses existing demo `.npz` unless `--regenerate-data` is set. Prints AUROC, AP, **ECE**, **Brier**, and decision-curve **net benefit @ 10%**. `--calibrate` fits temperature T on val and applies it to `--split`. `--out-json` dumps the same metrics for automation. Subgroup AUROC is printed when age/sex columns exist.

## End-to-end inference (GPU + weights)

Requires CUDA, RETFound (Hugging Face), EyeQ weights, optional Vim-S. See [docs/DATA.md](docs/DATA.md).

```powershell
$env:RETI_PIONEER_EYEQ_WEIGHTS = "pretrained\fundus\DenseNet121_v3_v1.tar"
python inference.py
```

Place TorchScript models under `models/` (`T2D_y0.pt`, etc.) as in the upstream release.

## Feature extraction from raw images

```powershell
python scripts/extract_features.py --cache-dir data/odir --all-backbones
# or legacy:
# python scripts/extract_features.py --manifest pairs.csv --out data/UKBCompressed/UKB_RETF.npz --backbone retf
```

`pairs.csv`: one `left_path,right_path` per line (same order as `UKB_mqd.npz`).

## Project layout

```
configs/default.yaml      # hyperparameters
configs/ablation_*.yaml   # baseline / learnq / multitask / full
model/                    # Reti-Pioneer, QualityAware, backbones
dataset/                  # UKB cache, ODIR, BRSET, RFMiD, public re-exports
utils/                    # training loop, metrics, calibration
scripts/                  # train, evaluate, ablations, download helper, public npz
reti_pioneer/             # constants, config, patient-level split
docs/                     # paper, data, public-data honesty, reproducibility
tests/                    # CPU smoke + follow-up (shapes, split, calibration)
main.py                   # upstream training entry (CUDA)
inference.py              # deployment inference API
```

## Implementation vs paper

| Fully supported locally | Requires approved external data |
|-------------------------|----------------------------------|
| Model code & training loop | 107k UKB + hospital CFPs |
| Demo synthetic `.npz` | SEED / China external validation |
| AUROC/AP on held-out tensors | Reported paper AUROC magnitudes |
| Public fundus **pipeline** tests | Six systemic disease labels on public sets |

## Citation

```bibtex
@article{zhang2026retipioneer,
  title={AI framework for multidisease detection via retinal imaging},
  author={Zhang, X. and Li, Q. and Liang, Y. and others},
  journal={Nature Medicine},
  year={2026},
  doi={10.1038/s41591-026-04359-w}
}
```

## License

See [LICENSE](LICENSE) (upstream Reti-Pioneer). Research use only; not a medical device.
