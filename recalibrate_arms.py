#!/usr/bin/env python3
"""Recalibrate every arm on the held-out calibration fold and refit operating points there.

Why this replaces the earlier numbers
-------------------------------------
Two problems in the first pass:

1. ``analyze_arms.py`` scored raw sigmoid probabilities. No temperature was ever
   fitted for the reported numbers, yet the text described the scores as
   temperature-scaled. The temperature column in Table 6 was not produced by any
   script.

2. ``operating_points.json`` fitted Youden and 95%-specificity thresholds on the
   *test* split, because it read the cached test predictions only. The caption
   said the thresholds came from the validation fold.

This script fixes both using artefacts training already wrote:

* ``cal_<ARM>.npz`` holds predictions on ``calibration_idx`` (64 images carved
  out of validation, disjoint from train/val/test).
* ``ckpt/py_<epoch>_val.npz`` holds per-sample validation predictions for the
  reported epoch, over the remaining 576 validation images.

Temperature is fitted on the 64 calibration images only. Thresholds are fitted on
the 576 validation images only. Neither quantity sees a test image. Test scores
are then reported after applying the fitted temperature.

Outputs (``results/real_rfmid/analysis_sel/stats/``):

    headline_calibrated.json    AUROC, AP, ECE, Brier before/after temperature
    operating_points_cal.json   Youden and 95% specificity thresholds from val
    calibration_fit.json        fitted temperature and calibration-fold ECE

Usage:
    python scripts/recalibrate_arms.py
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
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration import (  # noqa: E402
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    sensitivity_at_specificity,
    youden_operating_point,
)

ANALYSIS = Path(os.environ.get("ARM_ANALYSIS_DIR", ROOT))
OUT = Path(os.environ.get("ARM_STATS_DIR", ROOT))
BINS = 10
EPS = 1e-6


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def observed(y: np.ndarray) -> np.ndarray:
    return y >= 0


def macro(y: np.ndarray, p: np.ndarray, fn) -> float:
    vals = []
    for k in range(y.shape[1]):
        m = observed(y[:, k])
        if m.sum() == 0 or len(np.unique(y[m, k])) < 2:
            continue
        vals.append(fn(y[m, k], p[m, k]))
    return float(np.mean(vals)) if vals else float("nan")


def pooled(y: np.ndarray, p: np.ndarray, fn) -> float:
    m = observed(y)
    return float(fn(y[m].ravel(), p[m].ravel()))


def ece10(a, b) -> float:
    return expected_calibration_error(a, b, n_bins=BINS)


def load_val(arm: str, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    entry = manifest[arm]
    run_dir = Path(entry["run_dir"])
    epoch = Path(entry["ckpt"]).stem
    path = run_dir / "ckpt" / f"py_{epoch}_val.npz"
    d = np.load(path, allow_pickle=True)
    p, y = d["p"], d["y"]
    if p.ndim == 1:
        p, y = p.reshape(-1, 1), y.reshape(-1, 1)
    return p, y


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))

    fit_rows, head_rows, op_rows = {}, {}, {}

    for arm in manifest:
        cal = np.load(ANALYSIS / f"cal_{arm}.npz", allow_pickle=True)
        cal_p, cal_y = cal["probs"], cal["labels"]
        n_cal = int(cal_p.shape[0])

        # ---- temperature fitted on the 64 held-out calibration images -----
        m = observed(cal_y)
        T = fit_temperature(cal["logits"][m], cal_y[m])
        cal_logit = cal["logits"]
        cal_p_post = apply_temperature(cal_logit, T)
        ece_before_cal = pooled(cal_y, cal_p, ece10)
        ece_after_cal = pooled(cal_y, cal_p_post, ece10)

        # ---- test scores, before and after --------------------------------
        d = np.load(ANALYSIS / f"arm_{arm}.npz", allow_pickle=True)
        y_te, p_te = d["labels"], d["probs"]
        p_te_post = apply_temperature(d["logits"], T)

        head_rows[arm] = {
            "temperature": round(float(T), 4),
            "n_cal": n_cal,
            "macro_auroc_raw": round(macro(y_te, p_te, roc_auc_score), 4),
            "macro_auroc_cal": round(macro(y_te, p_te_post, roc_auc_score), 4),
            "macro_ap_raw": round(macro(y_te, p_te, average_precision_score), 4),
            "macro_ap_cal": round(macro(y_te, p_te_post, average_precision_score), 4),
            "pooled_ece_raw": round(pooled(y_te, p_te, ece10), 4),
            "pooled_ece_cal": round(pooled(y_te, p_te_post, ece10), 4),
            "mean_head_ece_raw": round(macro(y_te, p_te, ece10), 4),
            "mean_head_ece_cal": round(macro(y_te, p_te_post, ece10), 4),
            "brier_raw": round(pooled(y_te, p_te, brier_score), 4),
            "brier_cal": round(pooled(y_te, p_te_post, brier_score), 4),
        }
        fit_rows[arm] = {
            "temperature": round(float(T), 4),
            "n_cal_images": n_cal,
            "ece_on_cal_raw": round(float(ece_before_cal), 4),
            "ece_on_cal_post": round(float(ece_after_cal), 4),
        }
        print(
            f"{arm}: T={T:.4f} | cal ECE {ece_before_cal:.4f}->{ece_after_cal:.4f} | "
            f"test macroAUROC {head_rows[arm]['macro_auroc_raw']:.4f}->{head_rows[arm]['macro_auroc_cal']:.4f} | "
            f"meanHeadECE {head_rows[arm]['mean_head_ece_raw']:.4f}->{head_rows[arm]['mean_head_ece_cal']:.4f} | "
            f"pooledECE {head_rows[arm]['pooled_ece_raw']:.4f}->{head_rows[arm]['pooled_ece_cal']:.4f} | "
            f"Brier {head_rows[arm]['brier_raw']:.4f}->{head_rows[arm]['brier_cal']:.4f}",
            flush=True,
        )

        # ---- operating points on validation, applied to test ---------------
        vp, vy = load_val(arm, manifest)
        vp_post = apply_temperature(to_logit(vp), T)
        names = [str(x) for x in d["names"]]
        rows = {}
        for k, ep in enumerate(names):
            mv = observed(vy[:, k])
            if mv.sum() == 0 or len(np.unique(vy[mv, k])) < 2:
                continue
            yj = youden_operating_point(vy[mv, k], vp_post[mv, k])
            s95 = sensitivity_at_specificity(vy[mv, k], vp_post[mv, k], target_specificity=0.95)

            pt, yt = p_te_post[:, k], y_te[:, k]
            keep = observed(yt)
            pt, yt = pt[keep], yt[keep]

            def apply(thr: float) -> dict:
                pred = (pt >= thr).astype(int)
                tp = int(((pred == 1) & (yt == 1)).sum())
                fn = int(((pred == 0) & (yt == 1)).sum())
                tn = int(((pred == 0) & (yt == 0)).sum())
                fp = int(((pred == 1) & (yt == 0)).sum())
                return {
                    "test_tp": tp, "test_fn": fn, "test_tn": tn, "test_fp": fp,
                    "test_sensitivity": round(tp / max(tp + fn, 1), 4),
                    "test_specificity": round(tn / max(tn + fp, 1), 4),
                }

            rows[ep] = {
                "youden": {**{a: float(b) for a, b in yj.items()}, **apply(yj["threshold"])},
                "sens_at_95spec": {**{a: float(b) for a, b in s95.items()}, **apply(s95["threshold"])},
            }
        op_rows[arm] = rows
        dr = rows.get("DR")
        if dr:
            print(
                f"   DR youden thr={dr['youden']['threshold']:.3f} "
                f"test sens={dr['youden']['test_sensitivity']:.3f} spec={dr['youden']['test_specificity']:.3f} | "
                f"95spec thr={dr['sens_at_95spec']['threshold']:.3f} "
                f"sens={dr['sens_at_95spec']['test_sensitivity']:.3f} "
                f"spec={dr['sens_at_95spec']['test_specificity']:.3f}",
                flush=True,
            )

    (OUT / "headline_calibrated.json").write_text(json.dumps(head_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "calibration_fit.json").write_text(json.dumps(fit_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "operating_points_cal.json").write_text(json.dumps(op_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
