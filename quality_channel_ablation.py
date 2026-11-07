#!/usr/bin/env python3
"""Does the per-image quality vector change any prediction at inference time?

Every arm consumes the three-way quality vector in two places:

* the fusion scalar ``s = q · w`` inside each backbone block, and
* for E5 only, the backbone mixture ``softmax(MLP(q))``.

Both were trained. This script asks what happens at run time when the quality
vector is replaced by a constant, so the model can no longer tell a good image
from a poor one. For the fixed and monotone routers a constant vector collapses
to a constant fusion scalar, which removes per-image quality modulation while
leaving the architecture and every weight intact.

Outputs ``results/real_rfmid/analysis_sel/stats/quality_channel_ablation.json``.

Usage
-----
    python scripts/quality_channel_ablation.py --arms E2 E3 E4 E5
"""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))

from UKBDataset import UKBDatasetFast  # noqa: E402
from evaluate import (  # noqa: E402
    collect_probs,
    load_run_meta,
    resolve_eval_split,
    resolve_run_dir,
)
from router_intervention import build_model  # noqa: E402

ANALYSIS = ROOT
OUT = ROOT / "quality_channel_ablation.json"

UNIFORM = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=torch.float32)


def macro_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    for k in range(labels.shape[1]):
        y = labels[:, k]
        if y.size and len(np.unique(y)) > 1:
            vals.append(roc_auc_score(y, probs[:, k]))
    return float(np.mean(vals))


class ConstantQuality(torch.utils.data.Dataset):
    """Yield every sample unchanged except q, which becomes the same constant."""

    def __init__(self, inner, fill: torch.Tensor):
        self.inner = inner
        self.fill = fill

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        (l, r), m, (_ql, _qr), y = self.inner[i]
        return (l, r), m, (self.fill.clone(), self.fill.clone()), y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["E2", "E3", "E4", "E5"])
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "rfmid"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))

    payload: dict[str, dict] = {}
    for arm in args.arms:
        if arm not in manifest:
            continue
        entry = manifest[arm]
        run_dir = Path(resolve_run_dir(str(entry["run_dir"])))
        meta = load_run_meta(str(run_dir))
        ckpt = entry["ckpt"]
        if not Path(ckpt).is_file():
            ckpt = str(Path(entry["run_dir"]) / "ckpt" / Path(ckpt).name)

        model = build_model(meta, ckpt).to(device)
        ds = UKBDatasetFast(
            args.data_dir,
            meta=["baselineage", "gender", "weight", "ethnicity"],
            y=0,
            disease=[0],
            use_pretrain=["RETF", "SwinB", "VimS"],
            incident_exclude_prior=False,
        )
        diseases = list(meta.get("diseases") or []) or list(ds.disease_names)
        ds.set_target(0, diseases, incident_exclude_prior=False)
        eval_ds, _, _ = resolve_eval_split(str(run_dir), ds, args.split, 0.25, 42)

        p_raw, y, _ = collect_probs(model, eval_ds, device, batch_size=128, return_logits=True)
        if p_raw.ndim == 1:
            p_raw, y = p_raw.reshape(-1, 1), y.reshape(-1, 1)
        p_cst, y2, _ = collect_probs(
            model, ConstantQuality(eval_ds, UNIFORM), device, batch_size=128, return_logits=True
        )
        if p_cst.ndim == 1:
            p_cst, y2 = p_cst.reshape(-1, 1), y2.reshape(-1, 1)
        assert np.array_equal(np.asarray(y), np.asarray(y2))

        a_raw = macro_auroc(y, p_raw)
        a_cst = macro_auroc(y, p_cst)
        payload[arm] = {
            "n": int(y.shape[0]),
            "macro_auroc_as_trained": a_raw,
            "macro_auroc_constant_quality": a_cst,
            "delta": a_cst - a_raw,
            "max_abs_prob_change": float(np.abs(p_raw - p_cst).max()),
            "identical_within_1e-6": bool(np.abs(p_raw - p_cst).max() < 1e-6),
        }
        print(
            f"{arm}: as trained {a_raw:.4f}  constant q {a_cst:.4f}  "
            f"delta {a_cst - a_raw:+.5f}  max|dp| {payload[arm]['max_abs_prob_change']:.4f}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
