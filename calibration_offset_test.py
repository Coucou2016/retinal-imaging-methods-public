#!/usr/bin/env python3
"""Test the mechanism behind the multi-task calibration gap.

The held-out ladder showed that a two-parameter Platt map removes almost all of
the calibration error of the shared-head arms, while a one-parameter temperature
does not. The fitted intercepts also track the logit of the endpoint prevalence.
That pair of observations suggests a specific mechanism, namely that the shared
head leaves each endpoint's logit centred near zero instead of near logit(pi).

This script turns the suggestion into a test. If the mechanism is right, then
shifting every logit by logit of the validation prevalence, with no free
parameter and no slope change, should already recover most of the gap. Three
corrections are compared on the test fold, all of them fitted on validation:

    offset_prev   z' = z + logit(pi_val)                    0 free parameters
    offset_fit    z' = z + b, b minimises validation BCE      1 free parameter
    platt         z' = a z + b                                2 free parameters
    isotonic      piecewise-constant monotone map             many

A zero-parameter correction that works is stronger evidence than a two-parameter
one, because there is nothing left to fit to the endpoint.

Writes ``results/real_rfmid/analysis_sel/stats/calibration_offset.json``.

Usage
-----
    python scripts/calibration_offset_test.py
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT
OUT = ROOT / "calibration_offset.json"
ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
RUNGS = ("none", "offset_prev", "offset_fit", "platt", "isotonic")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    n = y.size
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        k = int(m.sum())
        if k == 0:
            continue
        total += k / n * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(total)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def fit_offset(y: np.ndarray, z: np.ndarray) -> float:
    """One free additive shift on the logit, slope fixed to one."""
    grid = np.linspace(-6.0, 6.0, 1201)
    best, best_b = np.inf, 0.0
    for b in grid:
        p = np.clip(sigmoid(z + b), 1e-7, 1 - 1e-7)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if bce < best:
            best, best_b = bce, float(b)
    return best_b


def macro(v: list[float]) -> float:
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))
    per_arm = {}

    for arm in ARMS:
        if arm not in manifest:
            continue
        entry = manifest[arm]
        run_dir = Path(entry["run_dir"])
        epoch = Path(entry["ckpt"]).stem
        val_f = run_dir / "ckpt" / f"py_{epoch}_val.npz"
        test_f = ANALYSIS / f"arm_{arm}.npz"
        if not val_f.is_file() or not test_f.is_file():
            print(f"[skip] {arm}")
            continue

        v = np.load(val_f, allow_pickle=True)
        pv, yv = v["p"].astype(np.float64), v["y"].astype(np.float64)
        if pv.ndim == 1:
            pv, yv = pv.reshape(-1, 1), yv.reshape(-1, 1)
        t = np.load(test_f, allow_pickle=True)
        pt, yt = t["probs"].astype(np.float64), t["labels"].astype(np.float64)
        names = [str(x) for x in t["names"]]

        acc = {r: {"ece": [], "brier": [], "auroc": []} for r in RUNGS}
        heads = {}

        for k, ep in enumerate(names):
            if len(np.unique(yv[:, k])) < 2 or len(np.unique(yt[:, k])) < 2:
                continue
            zv, zt = logit(pv[:, k]), logit(pt[:, k])
            yvk, ytk = yv[:, k], yt[:, k]

            pi_val = float(yvk.mean())
            b_prev = float(np.log(pi_val / (1 - pi_val)))
            p_prev = sigmoid(zt + b_prev)

            b_fit = fit_offset(yvk, zv)
            p_fit = sigmoid(zt + b_fit)

            lr = LogisticRegression(solver="lbfgs", max_iter=3000, C=1e6)
            lr.fit(zv.reshape(-1, 1), yvk)
            a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
            p_platt = sigmoid(a * zt + b)

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(pv[:, k], yvk)
            p_iso = np.clip(iso.predict(pt[:, k]), 1e-6, 1 - 1e-6)

            row = {
                "prevalence_val": pi_val,
                "prevalence_test": float(ytk.mean()),
                "logit_prevalence_val": b_prev,
                "offset_from_prevalence": b_prev,
                "offset_fitted": b_fit,
                "platt_slope": a,
                "platt_intercept": b,
                "ece_none": ece(ytk, pt[:, k]),
                "ece_offset_prev": ece(ytk, p_prev),
                "ece_offset_fit": ece(ytk, p_fit),
                "ece_platt": ece(ytk, p_platt),
                "ece_isotonic": ece(ytk, p_iso),
                "brier_none": brier(ytk, pt[:, k]),
                "brier_offset_prev": brier(ytk, p_prev),
                "brier_offset_fit": brier(ytk, p_fit),
                "brier_platt": brier(ytk, p_platt),
                "brier_isotonic": brier(ytk, p_iso),
                "auroc_none": float(roc_auc_score(ytk, pt[:, k])),
                "n": int(ytk.size),
            }
            for rung, p in (
                ("none", pt[:, k]),
                ("offset_prev", p_prev),
                ("offset_fit", p_fit),
                ("platt", p_platt),
                ("isotonic", p_iso),
            ):
                acc[rung]["ece"].append(row[f"ece_{rung}"])
                acc[rung]["brier"].append(row[f"brier_{rung}"])
                acc[rung]["auroc"].append(float(roc_auc_score(ytk, p)))
            heads[ep] = row

        summary = {}
        for r in RUNGS:
            summary[f"macro_ece_{r}"] = macro(acc[r]["ece"])
            summary[f"macro_brier_{r}"] = macro(acc[r]["brier"])
            summary[f"macro_auroc_{r}"] = macro(acc[r]["auroc"])
        summary["n_heads"] = len(heads)
        summary["offset_r"] = float("nan")
        summary["offset_rmse_vs_prevalence"] = float("nan")
        if len(heads) > 2:
            xs = np.array([h["logit_prevalence_val"] for h in heads.values()])
            ys = np.array([h["offset_fitted"] for h in heads.values()])
            summary["offset_r"] = float(np.corrcoef(xs, ys)[0, 1])
            summary["offset_rmse_vs_prevalence"] = float(np.sqrt(np.mean((ys - xs) ** 2)))
            summary["offset_slope_on_prevalence"] = float(np.polyfit(xs, ys, 1)[0])
            summary["offset_intercept_on_prevalence"] = float(np.polyfit(xs, ys, 1)[1])
            summary["platt_slope_mean"] = float(
                np.mean([h["platt_slope"] for h in heads.values()])
            )

        per_arm[arm] = {"per_head": heads, "summary": summary}
        s = summary
        print(
            f"{arm}: ECE none {s['macro_ece_none']:.4f}  prev-offset {s['macro_ece_offset_prev']:.4f}  "
            f"fitted-offset {s['macro_ece_offset_fit']:.4f}  platt {s['macro_ece_platt']:.4f}  "
            f"iso {s['macro_ece_isotonic']:.4f} | AUROC {s['macro_auroc_none']:.4f} / "
            f"iso {s['macro_auroc_isotonic']:.4f} | r(offset, logit prev) {s['offset_r']:.3f}"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "per_arm": per_arm,
                "n_val": 576,
                "n_test": 640,
                "note": "every correction is fitted, or read, on the validation fold only",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
