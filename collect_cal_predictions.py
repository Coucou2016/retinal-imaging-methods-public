#!/usr/bin/env python3
"""Collect per-sample predictions on the held-out calibration fold for every arm.

``calibration_idx`` in each run's ``split.npz`` holds 64 images carved out of the
validation fold. They overlap with nothing: not train, not val, not test. That is
where temperature has to be fitted, and where operating points have to be chosen.

Training never stored predictions for this fold, so it has to be scored once per
arm. The cost is small (64 images).

Output: ``results/real_rfmid/analysis_sel/cal_<ARM>.npz`` with
    logits (n, K), probs (n, K), labels (n, K), names (K,)

Usage:
    python scripts/collect_cal_predictions.py --arms E0 E1 E2 E3 E4 E5
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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UKBDataset import UKBDatasetFast  # noqa: E402
from collect_arm_predictions import build_model  # noqa: E402
from evaluate import (  # noqa: E402
    collect_probs,
    load_run_meta,
    resolve_eval_split,
    resolve_run_dir,
)

ANALYSIS = ROOT


def collect(arm: str, manifest: dict, data_dir: Path, device: torch.device) -> dict:
    entry = manifest[arm]
    run_dir = Path(resolve_run_dir(str(entry["run_dir"])))
    meta = load_run_meta(str(run_dir))
    meta["_run_dir"] = str(run_dir)
    meta["_ckpt"] = entry["ckpt"]
    diseases = list(meta.get("diseases") or [])

    model = build_model(meta).to(device)

    base_ds = UKBDatasetFast(
        str(data_dir),
        meta=["baselineage", "gender", "weight", "ethnicity"],
        y=0,
        disease=[0],
        use_pretrain=["RETF", "SwinB", "VimS"],
        incident_exclude_prior=False,
    )
    if not diseases:
        diseases = list(base_ds.disease_names)
    base_ds.set_target(0, diseases, incident_exclude_prior=False)

    cal_ds, _name, _bundle = resolve_eval_split(str(run_dir), base_ds, "calibration", 0.25, 42)
    probs, labels, logits = collect_probs(model, cal_ds, device, batch_size=64, return_logits=True)
    if probs.ndim == 1:
        probs, labels, logits = probs.reshape(-1, 1), labels.reshape(-1, 1), logits.reshape(-1, 1)

    return {
        "probs": probs.astype(np.float32),
        "labels": labels.astype(np.float32),
        "logits": logits.astype(np.float32),
        "names": np.asarray(diseases),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["E0", "E1", "E2", "E3", "E4", "E5"])
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "rfmid"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))

    for arm in args.arms:
        out = ANALYSIS / f"cal_{arm}.npz"
        if out.is_file():
            print(f"{arm}: {out.name} already present, skipping", flush=True)
            continue
        payload = collect(arm, manifest, Path(args.data_dir), device)
        np.savez_compressed(out, **payload)
        print(
            f"{arm}: n={payload['probs'].shape[0]} K={payload['probs'].shape[1]} -> {out.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
