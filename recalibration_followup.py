#!/usr/bin/env python3
"""Three follow-ups that the held-out recalibration result demands.

1. Stability. The per-endpoint Platt map is fitted on 576 validation images, so
   the reduction in ECE could be an artifact of a lucky fit. Resample the
   validation fold, refit, and re-score: report the spread of the reduction.

2. Shared versus per-endpoint map. A single global Platt map improves *pooled*
   ECE, but pooled ECE is dominated by the most prevalent endpoint. Report the
   macro per-endpoint ECE under the shared map, which is the honest comparison.

3. Threshold transfer. If the multivariable calibration error is largely an
   offset, then correcting it should also reduce the loss when a 95%-specificity
   threshold is carried from validation to test. Compare frozen-threshold
   sensitivity before and after Platt correction.

Writes ``results/real_rfmid/analysis_sel/stats/recalibration_followup.json``.

Usage
-----
    python scripts/recalibration_followup.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT
OUT = ROOT / "recalibration_followup.json"
ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
TARGET_SPEC = 0.95
N_RESAMPLE = 200
SEED = 0


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ece(y, p, n_bins=10):
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


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def fit_platt(y, z):
    lr = LogisticRegression(solver="lbfgs", max_iter=3000, C=1e6)
    lr.fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def sens_at_spec(y, p, target=TARGET_SPEC):
    """Highest sensitivity whose specificity is at least the target."""
    order = np.argsort(-p, kind="stable")
    y = y[order]
    p_sorted = p[order]
    distinct = np.where(np.diff(p_sorted))[0]
    idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[idx]
    fps = 1 + idx - tps
    spec = 1.0 - fps / max((1 - y).sum(), 1)
    sens = tps / max(y.sum(), 1)
    ok = np.where(spec >= target - 1e-9)[0]
    if ok.size == 0:
        return {"threshold": float(p_sorted[-1]), "sensitivity": 0.0,
                "specificity": float(spec[-1]), "met_target": False}
    best = ok[np.argmax(sens[ok])]
    return {"threshold": float(p_sorted[best]), "sensitivity": float(sens[best]),
            "specificity": float(spec[best]), "met_target": True}


def load_manifest():
    return json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))


def load_pair(arm, manifest):
    entry = manifest[arm]
    epoch = Path(entry["ckpt"]).stem
    vf = Path(entry["run_dir"]) / "ckpt" / f"py_{epoch}_val.npz"
    tf = ANALYSIS / f"arm_{arm}.npz"
    if not vf.is_file() or not tf.is_file():
        return None
    v = np.load(vf, allow_pickle=True)
    t = np.load(tf, allow_pickle=True)
    return {
        "pv": v["p"].astype(np.float64),
        "yv": v["y"].astype(np.float64),
        "pt": t["probs"].astype(np.float64),
        "yt": t["labels"].astype(np.float64),
        "names": [str(x) for x in t["names"]],
    }


def block_stability(data, rng):
    """Refit Platt on resampled validation rows, re-score the fixed test fold."""
    yv, pv, yt, pt = data["yv"], data["pv"], data["yt"], data["pt"]
    n = yv.shape[0]
    base = {}
    for k in range(yv.shape[1]):
        a, b = fit_platt(yv[:, k], logit(pv[:, k]))
        base[data["names"][k]] = {"slope": a, "intercept": b,
                                  "ece_platt": ece(yt[:, k], sigmoid(a * logit(pt[:, k]) + b))}
    draws = {"slope": [], "intercept": [], "macro_ece": []}
    for _ in range(N_RESAMPLE):
        idx = rng.integers(0, n, n)
        slopes, icpts, eces = [], [], []
        for k in range(yv.shape[1]):
            yy = yv[idx, k]
            if len(np.unique(yy)) < 2:
                continue
            a, b = fit_platt(yy, logit(pv[idx, k]))
            slopes.append(a)
            icpts.append(b)
            eces.append(ece(yt[:, k], sigmoid(a * logit(pt[:, k]) + b)))
        if eces:
            draws["slope"].append(float(np.mean(slopes)))
            draws["intercept"].append(float(np.mean(icpts)))
            draws["macro_ece"].append(float(np.mean(eces)))
    return {
        "point": base,
        "resample": {
            key: {
                "lo": float(np.quantile(v, 0.025)),
                "hi": float(np.quantile(v, 0.975)),
                "sd": float(np.std(v)),
            }
            for key, v in draws.items()
        },
    }


def block_shared(data):
    yv, pv, yt, pt = data["yv"], data["pv"], data["yt"], data["pt"]
    a, b = fit_platt(yv.ravel(), logit(pv).ravel())
    per_head_shared, per_head_none = [], []
    for k, ep in enumerate(data["names"]):
        yy = yt[:, k]
        if len(np.unique(yy)) < 2:
            continue
        per_head_none.append(ece(yy, pt[:, k]))
        per_head_shared.append(ece(yy, sigmoid(a * logit(pt[:, k]) + b)))
    return {
        "slope": a,
        "intercept": b,
        "macro_ece_none": float(np.mean(per_head_none)),
        "macro_ece_shared": float(np.mean(per_head_shared)),
        "pooled_ece_none": ece(yt.ravel(), pt.ravel()),
        "pooled_ece_shared": ece(yt.ravel(), (sigmoid(a * logit(pt).ravel() + b))),
    }


def block_transfer(data):
    """95%-specificity threshold: fit on validation, freeze, apply to test."""
    if "DR" not in data["names"]:
        return None
    k = data["names"].index("DR")
    yv, pv = data["yv"][:, k], data["pv"][:, k]
    yt, pt = data["yt"][:, k], data["pt"][:, k]

    a, b = fit_platt(yv, logit(pv))
    pv_c = sigmoid(a * logit(pv) + b)
    pt_c = sigmoid(a * logit(pt) + b)

    out = {}
    for label, (v, t) in (("raw", (pv, pt)), ("platt", (pv_c, pt_c))):
        val_op = sens_at_spec(yv, v)
        thr = val_op["threshold"]
        pred = (t >= thr).astype(int)
        tp = int(np.sum((pred == 1) & (yt == 1)))
        fn = int(np.sum((pred == 0) & (yt == 1)))
        tn = int(np.sum((pred == 0) & (yt == 0)))
        fp = int(np.sum((pred == 1) & (yt == 0)))
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        # oracle (upper bound) on test at the same target specificity
        oracle = sens_at_spec(yt, t)
        out[label] = {
            "threshold": thr,
            "val_sensitivity": val_op["sensitivity"],
            "val_specificity": val_op["specificity"],
            "test_sensitivity": float(sens),
            "test_specificity": float(spec),
            "test_tp": tp, "test_fn": fn, "test_tn": tn, "test_fp": fp,
            "test_sensitivity_oracle": oracle["sensitivity"],
            "transfer_loss": float(oracle["sensitivity"] - sens),
            "auroc_test": float(roc_auc_score(yt, t)),
        }
    out["delta_test_sensitivity"] = out["platt"]["test_sensitivity"] - out["raw"]["test_sensitivity"]
    out["delta_transfer_loss"] = out["platt"]["transfer_loss"] - out["raw"]["transfer_loss"]
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    manifest = load_manifest()
    payload = {"config": {"target_specificity": TARGET_SPEC, "n_resample": N_RESAMPLE,
                          "seed": SEED}, "arms": {}}

    for arm in ARMS:
        if arm not in manifest:
            continue
        data = load_pair(arm, manifest)
        if data is None:
            print(f"[skip] {arm}")
            continue
        payload["arms"][arm] = {
            "stability": block_stability(data, rng),
            "shared_map": block_shared(data),
            "threshold_transfer": block_transfer(data),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}\n")

    print("-- stability of the held-out Platt reduction --")
    for a, r in payload["arms"].items():
        s = r["stability"]["resample"]
        print(f"{a}: mean slope {s['slope']['lo']:.2f}-{s['slope']['hi']:.2f}"
              f"  intercept {s['intercept']['lo']:+.2f}..{s['intercept']['hi']:+.2f}"
              f"  macro ECE after Platt {s['macro_ece']['lo']:.3f}-{s['macro_ece']['hi']:.3f}")

    print("\n-- one shared map vs one map per endpoint (test macro ECE) --")
    for a, r in payload["arms"].items():
        m = r["shared_map"]
        print(f"{a}: none {m['macro_ece_none']:.3f}  shared {m['macro_ece_shared']:.3f}"
              f"   pooled {m['pooled_ece_none']:.3f} -> {m['pooled_ece_shared']:.3f}")

    print("\n-- frozen 95%-specificity threshold, raw vs Platt-corrected --")
    for a, r in payload["arms"].items():
        t = r["threshold_transfer"]
        if not t:
            continue
        print(f"{a}: raw {t['raw']['test_sensitivity']:.3f} (loss {t['raw']['transfer_loss']:+.3f})"
              f"  platt {t['platt']['test_sensitivity']:.3f} (loss {t['platt']['transfer_loss']:+.3f})"
              f"  delta {t['delta_test_sensitivity']:+.3f}")


if __name__ == "__main__":
    main()
