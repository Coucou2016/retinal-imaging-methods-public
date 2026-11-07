#!/usr/bin/env python3
"""Clinical utility at the frozen 95%-specificity operating point.

Sensitivity and specificity are already reported in the manuscript. Screening
programmes budget on a different currency: how many images must be read to find
one true positive, how many cases are missed per hundred, and how many false
alarms are raised per hundred. Those follow from sensitivity, specificity and
prevalence, and they differ sharply between arms even when AUROC barely moves.

Two threshold sources are reported side by side.
    frozen   threshold fitted on the 576-image validation fold, applied to test
    matched  threshold fitted on the test fold itself (an upper bound only)

Writes ``results/real_rfmid/analysis_sel/stats/clinical_utility.json``.

Usage
-----
    python scripts/clinical_utility.py
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

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT
OUT = ROOT / "clinical_utility.json"
ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
TARGET_SPEC = 0.95


def sens_at_spec(y: np.ndarray, p: np.ndarray, target: float = TARGET_SPEC) -> dict:
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    distinct = np.where(np.diff(p))[0]
    idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[idx]
    fps = 1 + idx - tps
    spec = 1.0 - fps / max((1 - y).sum(), 1)
    sens = tps / max(y.sum(), 1)
    ok = np.where(spec >= target - 1e-9)[0]
    if ok.size == 0:
        best = int(np.argmax(spec))
        met = False
    else:
        best, met = int(ok[np.argmax(sens[ok])]), True
    return {
        "threshold": float(p[best]),
        "sensitivity": float(sens[best]),
        "specificity": float(spec[best]),
        "met_target": met,
    }


def confusion(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    prev = float(y.mean())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "threshold": thr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "prevalence": prev,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "images_per_true_positive": (1.0 / ppv) if ppv > 0 else float("inf"),
        "missed_per_100_screened": 100.0 * fn / y.size,
        "false_alarms_per_100_screened": 100.0 * fp / y.size,
        "cases_missed_of_100_diseased": 100.0 * (1.0 - sens),
    }


def main() -> None:
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))
    res = {}
    for arm in ARMS:
        if arm not in manifest:
            continue
        entry = manifest[arm]
        vf = Path(entry["run_dir"]) / "ckpt" / f"py_{Path(entry['ckpt']).stem}_val.npz"
        tf = ANALYSIS / f"arm_{arm}.npz"
        if not vf.is_file() or not tf.is_file():
            continue
        v = np.load(vf, allow_pickle=True)
        t = np.load(tf, allow_pickle=True)
        names = [str(x) for x in t["names"]]
        if "DR" not in names:
            continue
        k = names.index("DR")
        yv, pv = v["y"][:, k].astype(np.float64), v["p"][:, k].astype(np.float64)
        yt, pt = t["labels"][:, k].astype(np.float64), t["probs"][:, k].astype(np.float64)

        frozen_op = sens_at_spec(yv, pv)
        frozen = confusion(yt, pt, frozen_op["threshold"])
        frozen["val_sensitivity"] = frozen_op["sensitivity"]
        frozen["val_specificity"] = frozen_op["specificity"]

        matched_op = sens_at_spec(yt, pt)
        matched = confusion(yt, pt, matched_op["threshold"])

        res[arm] = {
            "prevalence_test": float(yt.mean()),
            "n_test": int(yt.size),
            "frozen": frozen,
            "matched": matched,
            "sensitivity_gap": matched["sensitivity"] - frozen["sensitivity"],
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}\n")
    hdr = (
        f"{'arm':<4} {'frozen sens':>11} {'spec':>6} {'PPV':>6} {'NPV':>6} "
        f"{'img/TP':>7} {'miss/100':>9} {'FA/100':>7} | {'matched':>8} {'gap':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for arm, r in res.items():
        f, m = r["frozen"], r["matched"]
        print(
            f"{arm:<4} {f['sensitivity']:11.3f} {f['specificity']:6.3f} {f['ppv']:6.3f} {f['npv']:6.3f} "
            f"{f['images_per_true_positive']:7.2f} {f['missed_per_100_screened']:9.2f} "
            f"{f['false_alarms_per_100_screened']:7.2f} | {m['sensitivity']:8.3f} {r['sensitivity_gap']:+6.3f}"
        )


if __name__ == "__main__":
    main()
