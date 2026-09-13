# Reti-Pioneer methods extension — monotone quality routing & endpoint-aware multitask

Reproduction / methods workspace extending the **Nature Medicine (2026)** paper [*AI framework for multidisease detection via retinal imaging*](https://doi.org/10.1038/s41591-026-04359-w) (**Reti-Pioneer**).

**Working title:** *Extending Reti-Pioneer with Monotone Quality Routing and Endpoint-Aware Multi-Task Learning*

This project integrates the [official Reti-Pioneer release](https://github.com/lyhyl/Reti-Pioneer) with local scripts, configuration, demo data, public-dataset adapters, and documentation for end-to-end training and evaluation when UK Biobank / hospital cohorts are unavailable.

See [docs/paper/manuscript.md](docs/paper/manuscript.md), [docs/PAPER.md](docs/PAPER.md), [docs/DATA.md](docs/DATA.md), [docs/PUBLIC_DATA.md](docs/PUBLIC_DATA.md), and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## What is (and is not) claimed

| Claim | Status |
|-------|--------|
| Monotone bounded quality router (`bad ≤ usable ≤ good`) | Methods contribution |
| Masked partial-label multitask vs **released-code** independent loops | Methods contribution |
| Endpoint ontology + endpoint-aware cross-cohort evaluation | Methods / protocol |
| Calibration / ECE / DCA | **Evaluation framework** (not novelty) |
| UKB / clinical AUROCs from this workspace | **Not claimed** (待补充) |
| Synthetic feature-cache metrics | CI only (`clinical_claim_allowed: false`) |

**Published vs released:** the Reti-Pioneer *article* describes multitask screening; the *released* `main.py` trains per-disease binary loops. Our clone control matches the released code.

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

Training uses **pre-extracted foundation features** (`fast_mode: true`). Demo AUROC is pipeline sanity only.

## Ablations (E0–E4)

```powershell
python scripts/train.py --config configs/ablation_e0.yaml --demo --disease t2dm --horizon 0
python scripts/train.py --config configs/ablation_e1.yaml --demo --disease t2dm --horizon 0
python scripts/train.py --config configs/ablation_e4.yaml --demo --multitask --horizon 0
python scripts/evaluate.py --demo --split val --ckpt ckpt/<run>/multitask/y0 --calibrate --out-json results/metrics.json
python scripts/run_ablations.py --quick
```

| Config | Role |
|--------|------|
| `ablation_e0.yaml` | Fixed q, independent head, `released_code` ensemble |
| `ablation_e1.yaml` | Monotone quality router |
| `ablation_e2.yaml` | Multitask + masked BCE |
| `ablation_e3.yaml` | Monotone + multitask |
| `ablation_e4.yaml` | Full + disjoint calibration eval protocol |
| E5 quality-conditioned gating | Optional (`configs/ablation_e5.yaml`; `quality_gating` / `lambda_q`) |

`quality_router`: `fixed` | `free_linear` (ablation) | `monotone` (default when `learnable_q`).  
Ensemble: `released_code` (train soft / eval max; legacy alias `paper`), `published_soft_vote`, `mean`, `temp_mean`.

## Public datasets

See [docs/PUBLIC_DATA.md](docs/PUBLIC_DATA.md). Default cross-cohort endpoints: `hypertension_ocular`, `diabetes_ocular`. `--clinical-tables` refuses non-direct alignments.

## License vs research disclaimer

Software copyright: [LICENSE](LICENSE) (MIT, upstream Reti-Pioneer). Third-party notes: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Research disclaimer** (not a medical device; not for clinical decisions; synthetic metrics are not clinical AUROC) is **scientific guidance**, not an additional restriction on the MIT license.

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
