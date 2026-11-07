#!/usr/bin/env python3
"""Held-out recalibration: fit the correction on validation, score on test.

The in-sample ladder in ``extended_analysis.py`` shows that a two-parameter
Platt map (slope and intercept) removes almost all of the multi-task calibration
gap, while a one-parameter temperature cannot. That result is optimistic by
construction, so it says nothing about deployability.

This script repeats the exercise honestly. Each correction map is fitted on the
576-image validation fold and applied unchanged to the 640-image test fold. It
covers four rungs (none, temperature, Platt, isotonic) at two levels (one map per
endpoint, and a single map shared by all endpoints), and reports ECE, Brier and
AUROC on test. The monotone maps leave AUROC exactly unchanged.

Inputs
------
validation probabilities  <run_dir>/ckpt/py_<epoch>_val.npz  (written by train.py)
test probabilities        results/real_rfmid/analysis_sel/arm_<ARM>.npz
checkpoints to use        results/real_rfmid/analysis_sel/manifest.json

Writes ``results/real_rfmid/analysis_sel/stats/recalibration_holdout.json``.

Usage
-----
    python scripts/recalibration_holdout.py
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT
OUT = ROOT / "recalibration_holdout.json"
ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
RUNGS = ("none", "temperature", "platt", "isotonic")


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


def fit_temperature(y: np.ndarray, z: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 600))
    best, best_t = np.inf, 1.0
    for t in grid:
        p = np.clip(sigmoid(z / t), 1e-7, 1 - 1e-7)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if bce < best:
            best, best_t = bce, float(t)
    return best_t


def fit_platt(y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    lr = LogisticRegression(solver="lbfgs", max_iter=3000, C=1e6)
    lr.fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def macro(v: list[float]) -> float:
    return float(np.mean(v)) if v else float("nan")


def load_manifest() -> dict:
    return json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))


def load_val(entry: dict) -> tuple[np.ndarray, np.ndarray] | None:
    run_dir = Path(entry["run_dir"])
    epoch = Path(entry["ckpt"]).stem
    f = run_dir / "ckpt" / f"py_{epoch}_val.npz"
    if not f.is_file():
        return None
    d = np.load(f, allow_pickle=True)
    return d["p"].astype(np.float64), d["y"].astype(np.float64)


def load_test(arm: str) -> dict | None:
    f = ANALYSIS / f"arm_{arm}.npz"
    if not f.is_file():
        return None
    d = np.load(f, allow_pickle=True)
    return {
        "probs": d["probs"].astype(np.float64),
        "labels": d["labels"].astype(np.float64),
        "names": [str(x) for x in d["names"]],
    }


def run() -> dict:
    manifest = load_manifest()
    per_arm = {}

    for arm in ARMS:
        if arm not in manifest:
            continue
        val = load_val(manifest[arm])
        test = load_test(arm)
        if val is None or test is None:
            print(f"[skip] {arm}: missing validation or test cache")
            continue
        pv, yv = val
        pt, yt = test["probs"], test["labels"]
        names = test["names"]

        acc = {r: {"ece": [], "brier": [], "auroc": []} for r in RUNGS}
        heads = {}

        for k, ep in enumerate(names):
            if len(np.unique(yv[:, k])) < 2 or len(np.unique(yt[:, k])) < 2:
                continue
            zv, zt = logit(pv[:, k]), logit(pt[:, k])
            yvk, ytk = yv[:, k], yt[:, k]

            t = fit_temperature(yvk, zv)
            p_temp = sigmoid(zt / t)

            a, b = fit_platt(yvk, zv)
            p_platt = sigmoid(a * zt + b)

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(pv[:, k], yvk)
            p_iso = np.clip(iso.predict(pt[:, k]), 1e-6, 1 - 1e-6)

            row = {
                "temperature": t,
                "platt_slope": a,
                "platt_intercept": b,
                "ece_none": ece(ytk, pt[:, k]),
                "ece_temperature": ece(ytk, p_temp),
                "ece_platt": ece(ytk, p_platt),
                "ece_isotonic": ece(ytk, p_iso),
                "brier_none": brier(ytk, pt[:, k]),
                "brier_temperature": brier(ytk, p_temp),
                "brier_platt": brier(ytk, p_platt),
                "brier_isotonic": brier(ytk, p_iso),
                "auroc": float(roc_auc_score(ytk, pt[:, k])),
                "n": int(ytk.size),
                "prevalence": float(ytk.mean()),
            }
            heads[ep] = row
            for rung, p in (
                ("none", pt[:, k]),
                ("temperature", p_temp),
                ("platt", p_platt),
                ("isotonic", p_iso),
            ):
                acc[rung]["ece"].append(row[f"ece_{rung}"])
                acc[rung]["brier"].append(row[f"brier_{rung}"])
                acc[rung]["auroc"].append(float(roc_auc_score(ytk, p)))

        summary = {f"macro_ece_{r}": macro(acc[r]["ece"]) for r in RUNGS}
        summary.update({f"macro_brier_{r}": macro(acc[r]["brier"]) for r in RUNGS})
        summary.update({f"macro_auroc_{r}": macro(acc[r]["auroc"]) for r in RUNGS})
        summary["ece_reduction_platt"] = summary["macro_ece_none"] - summary["macro_ece_platt"]
        summary["ece_reduction_temperature"] = (
            summary["macro_ece_none"] - summary["macro_ece_temperature"]
        )
        summary["ece_reduction_isotonic"] = (
            summary["macro_ece_none"] - summary["macro_ece_isotonic"]
        )

        # one map shared across endpoints, on pooled logits
        a_s, b_s = fit_platt(yv.ravel(), logit(pv).ravel())
        p_shared = sigmoid(a_s * logit(pt).ravel() + b_s)
        shared = {
            "platt_slope": a_s,
            "platt_intercept": b_s,
            "pooled_ece_none": ece(yt.ravel(), pt.ravel()),
            "pooled_ece_shared_platt": ece(yt.ravel(), p_shared),
        }

        per_arm[arm] = {"per_head": heads, "summary": summary, "shared_platt": shared}

    return {"per_arm": per_arm, "n_val": 576, "n_test": 640,
            "note": "correction maps fitted on the validation fold only"}


def main() -> None:
    payload = run()
    if not payload["per_arm"]:
        raise SystemExit("no arms processed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}\n")

    hdr = f"{'arm':<4} {'none':>7} {'temp':>7} {'platt':>7} {'iso':>7}  {'Δplatt':>7}  {'brier none':>10} {'platt':>7}"
    print(hdr)
    print("-" * len(hdr))
    for a, d in payload["per_arm"].items():
        s = d["summary"]
        print(
            f"{a:<4} {s['macro_ece_none']:7.3f} {s['macro_ece_temperature']:7.3f} "
            f"{s['macro_ece_platt']:7.3f} {s['macro_ece_isotonic']:7.3f}  "
            f"{s['ece_reduction_platt']:+7.3f}  {s['macro_brier_none']:10.3f} "
            f"{s['macro_brier_platt']:7.3f}"
        )
    print()
    for a, d in payload["per_arm"].items():
        s = d["shared_platt"]
        print(
            f"{a} shared Platt: slope {s['platt_slope']:.3f}, intercept {s['platt_intercept']:+.3f}, "
            f"pooled ECE {s['pooled_ece_none']:.3f} -> {s['pooled_ece_shared_platt']:.3f}"
        )


if __name__ == "__main__":
    main()
