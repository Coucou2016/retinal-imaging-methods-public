#!/usr/bin/env python3
"""Threshold-transfer diagnostics for every arm.

Two questions the manuscript has to answer with numbers rather than assertion:

1. Does the fitted temperature actually help? The primary calibration fold holds
   64 images, which is small. This script also fits a temperature on the full
   640-image validation fold (576 + 64) as a sensitivity analysis, so the reader
   can see how much of the calibration result is an artefact of fold size.

2. How much of the drop at a fixed specificity comes from the threshold
   travelling badly, and how much from discrimination? The oracle column picks
   the threshold on test itself, which is not deployable but bounds what any
   threshold rule could achieve at that specificity.

Output: ``stats/threshold_transfer.json``

Usage:
    python scripts/threshold_transfer.py
"""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration import (  # noqa: E402
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_temperature,
)

ANALYSIS = ROOT
OUT = ROOT
BINS = 10
EPS = 1e-6
TARGET_SPEC = 0.95


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def macro(y, p, fn):
    vals = []
    for k in range(y.shape[1]):
        m = y[:, k] >= 0
        if m.sum() == 0 or len(np.unique(y[m, k])) < 2:
            continue
        vals.append(fn(y[m, k], p[m, k]))
    return float(np.mean(vals)) if vals else float("nan")


def pooled(y, p, fn):
    m = y >= 0
    return float(fn(y[m].ravel(), p[m].ravel()))


def ece10(a, b):
    return expected_calibration_error(a, b, n_bins=BINS)


def sens_at_spec(y: np.ndarray, p: np.ndarray, target: float = TARGET_SPEC) -> dict:
    """Highest sensitivity whose empirical specificity is at least ``target``."""
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1.0 - fpr
    ok = spec >= target
    if not ok.any():
        return {"threshold": float("nan"), "sensitivity": float("nan"), "specificity": float("nan")}
    idx = int(np.argmax(np.where(ok, tpr, -1.0)))
    return {
        "threshold": float(thr[idx]),
        "sensitivity": float(tpr[idx]),
        "specificity": float(spec[idx]),
    }


def load_val(arm: str, manifest: dict):
    entry = manifest[arm]
    run_dir = Path(entry["run_dir"])
    epoch = Path(entry["ckpt"]).stem
    d = np.load(run_dir / "ckpt" / f"py_{epoch}_val.npz", allow_pickle=True)
    p, y = d["p"], d["y"]
    if p.ndim == 1:
        p, y = p.reshape(-1, 1), y.reshape(-1, 1)
    return p, y


def main() -> None:
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))
    out = {}

    for arm in manifest:
        cal = np.load(ANALYSIS / f"cal_{arm}.npz", allow_pickle=True)
        c_p, c_y, c_z = cal["probs"], cal["labels"], cal["logits"]
        v_p, v_y = load_val(arm, manifest)
        te = np.load(ANALYSIS / f"arm_{arm}.npz", allow_pickle=True)
        t_p, t_y, t_z = te["probs"], te["labels"], te["logits"]

        m = c_y >= 0
        T_small = float(fit_temperature(c_z[m], c_y[m]))

        # full validation fold = 64 calibration + 576 validation
        full_p = np.concatenate([c_p, v_p], axis=0)
        full_y = np.concatenate([c_y, v_y], axis=0)
        full_z = np.concatenate([c_z, to_logit(v_p)], axis=0)
        mf = full_y >= 0
        T_full = float(fit_temperature(full_z[mf], full_y[mf]))

        row = {
            "temperature_cal64": round(T_small, 4),
            "temperature_val640": round(T_full, 4),
            "ece_on_cal64_raw": round(pooled(c_y, c_p, ece10), 4),
            "ece_on_cal64_after_T64": round(pooled(c_y, apply_temperature(c_z, T_small), ece10), 4),
            "ece_on_cal64_after_T640": round(pooled(c_y, apply_temperature(c_z, T_full), ece10), 4),
            "test_ece_raw": round(pooled(t_y, t_p, ece10), 4),
            "test_ece_after_T64": round(pooled(t_y, apply_temperature(t_z, T_small), ece10), 4),
            "test_ece_after_T640": round(pooled(t_y, apply_temperature(t_z, T_full), ece10), 4),
            "test_brier_raw": round(pooled(t_y, t_p, brier_score), 4),
            "test_brier_after_T64": round(pooled(t_y, apply_temperature(t_z, T_small), brier_score), 4),
            "test_brier_after_T640": round(pooled(t_y, apply_temperature(t_z, T_full), brier_score), 4),
            "test_mean_head_ece_raw": round(macro(t_y, t_p, ece10), 4),
            "test_mean_head_ece_after_T64": round(macro(t_y, apply_temperature(t_z, T_small), ece10), 4),
            "test_mean_head_ece_after_T640": round(macro(t_y, apply_temperature(t_z, T_full), ece10), 4),
            "auroc_unchanged": round(macro(t_y, t_p, roc_auc_score), 4),
        }

        # DR threshold transfer
        names = [str(x) for x in te["names"]]
        k = names.index("DR")
        yv = v_y[:, k]
        pv = apply_temperature(to_logit(v_p[:, k]), T_small).ravel()
        yt = t_y[:, k]
        pt = apply_temperature(t_z[:, k], T_small).ravel()
        keep = yt >= 0
        yt, pt = yt[keep], pt[keep]

        # oracle temperature: fitted on the test labels themselves, an upper bound
        T_oracle = float(fit_temperature(t_z[:, k], t_y[:, k]))

        frozen = sens_at_spec(yv, pv, TARGET_SPEC)
        oracle = sens_at_spec(yt, pt, TARGET_SPEC)
        pred = (pt >= frozen["threshold"]).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        tn = int(((pred == 0) & (yt == 0)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())

        row["DR"] = {
            "val_threshold_at_95spec": round(frozen["threshold"], 4),
            "val_sensitivity_at_95spec": round(frozen["sensitivity"], 4),
            "val_specificity_at_95spec": round(frozen["specificity"], 4),
            "test_sensitivity_frozen": round(tp / max(tp + fn, 1), 4),
            "test_specificity_frozen": round(tn / max(tn + fp, 1), 4),
            "test_tp": tp, "test_fn": fn, "test_tn": tn, "test_fp": fp,
            "test_sensitivity_oracle": round(oracle["sensitivity"], 4),
            "test_specificity_oracle": round(oracle["specificity"], 4),
            "transfer_loss": round(oracle["sensitivity"] - tp / max(tp + fn, 1), 4),
            "temperature_oracle_test": round(T_oracle, 4),
            "ece_raw": round(float(ece10(yt, pt)), 4),
            "ece_after_oracle_T": round(
                float(ece10(yt, apply_temperature(t_z[:, k], T_oracle).ravel()[keep])), 4
            ),
        }
        out[arm] = row

        print(
            f"{arm}: T64={T_small:.3f} T640={T_full:.3f} | "
            f"test ECE {row['test_ece_raw']:.4f} -> T64 {row['test_ece_after_T64']:.4f} "
            f"-> T640 {row['test_ece_after_T640']:.4f} | "
            f"DR@95spec frozen sens {row['DR']['test_sensitivity_frozen']:.3f} "
            f"(spec {row['DR']['test_specificity_frozen']:.3f}) vs oracle {row['DR']['test_sensitivity_oracle']:.3f}",
            flush=True,
        )

    (OUT / "threshold_transfer.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT / "threshold_transfer.json")


if __name__ == "__main__":
    main()
