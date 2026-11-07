#!/usr/bin/env python3
"""Seed-level replication of the 95%-specificity operating point.

Section 4.11 of the manuscript reports a large threshold-transfer effect for E5
at a single training seed. That observation was flagged as unreplicated. Every
replicate run already stored its 576-image validation predictions next to its
test predictions, so the operating point can be recomputed for all four seeds
without retraining anything.

For each arm and each seed the script does the following.

1. Pick the reporting epoch by validation macro AUROC, exactly as the main
   protocol does (the choice is already recorded in ``replicate_manifest.json``).
2. Freeze the diabetic-retinopathy threshold at 95% specificity on the 576
   validation images.
3. Apply that threshold unchanged to the 640 test images and record sensitivity,
   realised specificity, and the confusion counts.
4. Compute the matched threshold on test, which is the upper bound any threshold
   rule could reach at the same specificity budget.
5. Record the shift of the negative-score distribution between validation and
   test, so the transfer loss can be read against a distribution statistic
   rather than against the model ranking.

Outputs ``results/real_rfmid/analysis_seed/threshold_transfer_replication.json``.

Usage
-----
    python scripts/threshold_transfer_replication.py
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
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
SEED_DIR = ROOT
MAIN_DIR = ROOT
OUT = SEED_DIR / "threshold_transfer_replication.json"

ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
SEEDS = [42, 43, 44, 45]
TARGET_SPEC = 0.95
ENDPOINT = "DR"


def sens_at_spec(y: np.ndarray, p: np.ndarray, target: float = TARGET_SPEC) -> dict:
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


def confusion(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum())
    return {
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
    }


def load_run_pred(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = np.load(path, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    return d["probs"], d["labels"], names


def val_predictions(arm: str, seed: int, manifest: dict, primary: dict) -> tuple[np.ndarray, np.ndarray]:
    key = f"{arm}_s{seed}"
    if key in manifest:
        rec = manifest[key]
        run_dir = Path(rec["run_dir"])
        epoch = int(rec["epoch"])
    else:
        # seed 42 is the primary run, recorded in the main analysis manifest
        rec = primary[arm]
        run_dir = Path(rec["run_dir"])
        epoch = int(Path(rec["ckpt"]).stem)
    d = np.load(run_dir / "ckpt" / f"py_{epoch}_val.npz", allow_pickle=True)
    p, y = d["p"], d["y"]
    if p.ndim == 1:
        p, y = p.reshape(-1, 1), y.reshape(-1, 1)
    return p, y


def test_predictions(arm: str, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if seed == 42:
        path = MAIN_DIR / f"arm_{arm}.npz"
    else:
        path = SEED_DIR / f"arm_{arm}_s{seed}.npz"
    return load_run_pred(path)


def partial_auc(y: np.ndarray, p: np.ndarray, max_fpr: float = 0.10) -> float:
    """Standardised partial AUC over the low false-positive region.

    McClish-style normalisation: the area under the ROC between 0 and ``max_fpr``
    rescaled so that a random classifier scores 0.5 and a perfect one 1.0. This
    is the discrimination measure that matches a high-specificity screening
    requirement, where a whole-AUROC summary hides the local ROC shape.
    """
    fpr, tpr, _ = roc_curve(y, p)
    keep = fpr <= max_fpr
    if keep.sum() < 2:
        return float("nan")
    f = fpr[keep]
    t = tpr[keep]
    area = float(np.trapezoid(t, f)) if hasattr(np, "trapezoid") else float(np.trapz(t, f))
    min_area = 0.5 * max_fpr ** 2
    max_area = max_fpr
    return float(0.5 * (1.0 + (area - min_area) / (max_area - min_area)))


def local_slope(y: np.ndarray, p: np.ndarray, target: float = TARGET_SPEC) -> float:
    """Local ROC slope around the target specificity.

    A steep ROC near 95% specificity means a small upward drift of specificity
    costs a large amount of sensitivity, which is the quantity that converts a
    distribution shift into a sensitivity loss.
    """
    fpr, tpr, _ = roc_curve(y, p)
    spec = 1.0 - fpr
    order = np.argsort(spec)
    spec, tpr = spec[order], tpr[order]
    lo = np.searchsorted(spec, target + 0.02)
    hi = np.searchsorted(spec, max(target - 0.02, 0.0))
    lo = min(lo, len(spec) - 1)
    hi = min(hi, len(spec) - 1)
    ds = spec[lo] - spec[hi]
    if abs(ds) < 1e-9:
        return float("nan")
    return float((tpr[lo] - tpr[hi]) / ds)


def main() -> None:
    manifest = json.loads((SEED_DIR / "replicate_manifest.json").read_text(encoding="utf-8"))
    primary = json.loads((MAIN_DIR / "manifest.json").read_text(encoding="utf-8"))
    per_arm: dict[str, dict] = {}

    for arm in ARMS:
        rows = []
        for seed in SEEDS:
            key = f"{arm}_s{seed}"
            if seed != 42 and key not in manifest:
                continue
            vp, vy = val_predictions(arm, seed, manifest, primary)
            tp, ty, names = test_predictions(arm, seed)
            k = names.index(ENDPOINT)

            yv, pv = vy[:, k], vp[:, k]
            keep_v = yv >= 0
            yv, pv = yv[keep_v], pv[keep_v]
            yt, pt = ty[:, k], tp[:, k]
            keep_t = yt >= 0
            yt, pt = yt[keep_t], pt[keep_t]

            frozen = sens_at_spec(yv, pv)
            applied = confusion(yt, pt, frozen["threshold"])
            matched = sens_at_spec(yt, pt)

            # distribution shift: mean score of the negatives, and the location
            # of the validation threshold inside the test negative distribution
            val_neg = pv[yv == 0]
            test_neg = pt[yt == 0]
            thr = frozen["threshold"]
            slope = local_slope(yt, pt)
            excess_spec = applied["specificity"] - TARGET_SPEC
            # mechanism check: if the whole negative distribution merely slid by
            # its measured mean displacement, the frozen threshold would sit at a
            # predictable place in the test negative distribution.  Compare that
            # prediction with the realised specificity.
            shift = float(test_neg.mean() - val_neg.mean())
            spec_pred = float((val_neg < thr - shift).mean())
            rows.append({
                "seed": seed,
                "epoch": int(manifest[key]["epoch"]) if key in manifest else int(Path(primary[arm]["ckpt"]).stem),                "val_threshold": frozen["threshold"],
                "val_sensitivity": frozen["sensitivity"],
                "val_specificity": frozen["specificity"],
                "test_sensitivity_frozen": applied["sensitivity"],
                "test_specificity_frozen": applied["specificity"],
                "test_specificity_predicted_shift": spec_pred,
                "specificity_prediction_error": applied["specificity"] - spec_pred,
                "test_sensitivity_matched": matched["sensitivity"],
                "test_specificity_matched": matched["specificity"],
                "transfer_loss": matched["sensitivity"] - applied["sensitivity"],
                "excess_specificity": excess_spec,
                "local_roc_slope": slope,
                "expected_loss_from_excess_spec": (
                    float(-slope * excess_spec) if np.isfinite(slope) else float("nan")
                ),
                "test_auroc": float(roc_auc_score(yt, pt)),
                "test_partial_auroc": partial_auc(yt, pt),
                "val_negative_mean": float(val_neg.mean()),
                "test_negative_mean": float(test_neg.mean()),
                "negative_mean_shift": float(test_neg.mean() - val_neg.mean()),
                "val_negative_p95": float(np.quantile(val_neg, 0.95)),
                "test_negative_p95": float(np.quantile(test_neg, 0.95)),
                "val_spec_at_test_p95": float((val_neg >= np.quantile(test_neg, 0.95)).mean()),
                "tp": applied["tp"], "fn": applied["fn"],
                "tn": applied["tn"], "fp": applied["fp"],
                "ppv": applied["tp"] / max(applied["tp"] + applied["fp"], 1),
                "npv": applied["tn"] / max(applied["tn"] + applied["fn"], 1),
                "missed_per_100": 100.0 * applied["fn"] / max(applied["tp"] + applied["fn"] + applied["tn"] + applied["fp"], 1),
                "false_alarm_per_100": 100.0 * applied["fp"] / max(applied["tp"] + applied["fn"] + applied["tn"] + applied["fp"], 1),
            })

        def agg(name: str) -> dict:
            v = np.array([r[name] for r in rows], dtype=float)
            return {
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if v.size > 1 else None,
                "min": float(v.min()),
                "max": float(v.max()),
            }

        per_arm[arm] = {
            "seeds": [r["seed"] for r in rows],
            "rows": rows,
            "aggregate": {
                k: agg(k)
                for k in (
                    "val_threshold",
                    "test_sensitivity_frozen",
                    "test_specificity_frozen",
                    "test_specificity_predicted_shift",
                    "specificity_prediction_error",
                    "test_sensitivity_matched",
                    "transfer_loss",
                    "negative_mean_shift",
                    "test_auroc",
                    "test_partial_auroc",
                    "excess_specificity",
                    "local_roc_slope",
                    "expected_loss_from_excess_spec",
                    "ppv",
                    "npv",
                    "missed_per_100",
                    "false_alarm_per_100",
                )
            },
        }

        # decomposition: how much of the loss the excess specificity explains
        loss = np.array([r["transfer_loss"] for r in rows], dtype=float)
        explained = np.array([r["expected_loss_from_excess_spec"] for r in rows], dtype=float)
        ok = np.isfinite(loss) & np.isfinite(explained)
        per_arm[arm]["decomposition"] = {
            "mean_loss": float(loss[ok].mean()) if ok.any() else None,
            "mean_explained": float(explained[ok].mean()) if ok.any() else None,
            "share_explained": (
                float((explained[ok].sum() / loss[ok].sum())) if ok.any() and loss[ok].sum() != 0 else None
            ),
        }
        a = per_arm[arm]["aggregate"]
        print(
            f"{arm}: frozen sens {a['test_sensitivity_frozen']['mean']:.3f} "
            f"± {a['test_sensitivity_frozen']['sd']:.3f} "
            f"(range {a['test_sensitivity_frozen']['min']:.3f}-{a['test_sensitivity_frozen']['max']:.3f}) | "
            f"matched {a['test_sensitivity_matched']['mean']:.3f} "
            f"± {a['test_sensitivity_matched']['sd']:.3f} | "
            f"loss {a['transfer_loss']['mean']:+.3f} | "
            f"neg shift {a['negative_mean_shift']['mean']:+.3f}",
            flush=True,
        )

    payload = {
        "endpoint": ENDPOINT,
        "target_specificity": TARGET_SPEC,
        "seeds": SEEDS,
        "per_arm": per_arm,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
