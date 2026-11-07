#!/usr/bin/env python3
"""Operating-point analysis with the threshold frozen on validation.

The previous draft quoted sensitivity at 95% specificity after choosing the
threshold on the test set. That is leakage: the operating point had seen the
labels it was scored against, and the resulting number (0.766 for E5) does not
survive an honest split.

This script reports two quantities side by side, because they answer different
questions.

``frozen``
    Threshold chosen on the 576-image validation fold at 95% specificity, then
    applied unchanged to test. This is what a deployed rule would do. Its test
    specificity drifts away from 95% because the validation fold is small.

``matched``
    The best sensitivity on test among thresholds whose *test* specificity is at
    least 95%. Not deployable, because the threshold needs test labels. It is an
    upper bound on what any threshold rule could reach, and it is the fair way to
    compare discrimination between arms at a fixed false-positive budget.

Paired image-level bootstrap (1000 resamples) gives intervals for both, and for
the E5 minus E0 difference on each.

Output: ``stats/operating_point_analysis.json``

Usage:
    python scripts/operating_point_analysis.py
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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration import apply_temperature, fit_temperature  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ANALYSIS = ROOT
OUT = ROOT
EPS = 1e-6
TARGET = 0.95
N_BOOT = 1000


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sens_at_spec(y: np.ndarray, p: np.ndarray, target: float = TARGET) -> tuple[float, float, float]:
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1 - fpr
    ok = spec >= target
    if not ok.any():
        return float("nan"), float("nan"), float("nan")
    i = int(np.argmax(np.where(ok, tpr, -1.0)))
    return float(thr[i]), float(tpr[i]), float(spec[i])


def confusion(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[float, float]:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum())
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def load_val(manifest: dict, arm: str):
    e = manifest[arm]
    run_dir = Path(e["run_dir"])
    epoch = Path(e["ckpt"]).stem
    d = np.load(run_dir / "ckpt" / f"py_{epoch}_val.npz", allow_pickle=True)
    p, y = d["p"], d["y"]
    if p.ndim == 1:
        p, y = p.reshape(-1, 1), y.reshape(-1, 1)
    return p, y


def boot_ci(values: list[float]) -> tuple[float, float]:
    a = np.asarray([v for v in values if np.isfinite(v)])
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def main() -> None:
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)

    per_arm = {}
    test_scores = {}

    for arm in manifest:
        cal = np.load(ANALYSIS / f"cal_{arm}.npz", allow_pickle=True)
        m = cal["labels"] >= 0
        T = float(fit_temperature(cal["logits"][m], cal["labels"][m]))

        vp, vy = load_val(manifest, arm)
        te = np.load(ANALYSIS / f"arm_{arm}.npz", allow_pickle=True)
        names = [str(x) for x in te["names"]]
        k = names.index("DR")

        yv = vy[:, k]
        pv = apply_temperature(to_logit(vp[:, k]), T).ravel()
        yt = te["labels"][:, k]
        pt = apply_temperature(te["logits"][:, k], T).ravel()
        keep = yt >= 0
        yt, pt = yt[keep], pt[keep]
        test_scores[arm] = (yt, pt)

        thr_frozen, val_sens, val_spec = sens_at_spec(yv, pv)
        sens_f, spec_f = confusion(yt, pt, thr_frozen)
        _, sens_m, spec_m = sens_at_spec(yt, pt)

        per_arm[arm] = {
            "temperature": round(T, 4),
            "val_threshold": round(thr_frozen, 4),
            "val_sensitivity": round(val_sens, 4),
            "val_specificity": round(val_spec, 4),
            "test_sensitivity_frozen": round(sens_f, 4),
            "test_specificity_frozen": round(spec_f, 4),
            "test_sensitivity_matched": round(sens_m, 4),
            "test_specificity_matched": round(spec_m, 4),
        }
        print(
            f"{arm}: val thr={thr_frozen:.3f} (sens {val_sens:.3f}, spec {val_spec:.3f}) | "
            f"test frozen sens={sens_f:.3f} spec={spec_f:.3f} | matched sens={sens_m:.3f} spec={spec_m:.3f}",
            flush=True,
        )

    # ---- bootstrap: frozen and matched sensitivity, plus E5 - E0 ----------
    arms = list(per_arm)
    n = len(test_scores[arms[0]][0])
    boot = {a: {"frozen": [], "matched": []} for a in arms}
    boot_delta = {"frozen": [], "matched": []}

    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        this = {}
        for a in arms:
            y, p = test_scores[a]
            yb, pb = y[idx], p[idx]
            if len(np.unique(yb)) < 2:
                this[a] = None
                continue
            sf, _ = confusion(yb, pb, per_arm[a]["val_threshold"])
            _, sm, _ = sens_at_spec(yb, pb)
            boot[a]["frozen"].append(sf)
            boot[a]["matched"].append(sm)
            this[a] = (sf, sm)
        if this.get("E5") and this.get("E3") and this.get("E0"):
            boot_delta["frozen"].append(this["E5"][0] - this["E0"][0])
            boot_delta["matched"].append(this["E5"][1] - this["E0"][1])
        if this.get("E3") and this.get("E0"):
            pass

    for a in arms:
        lo, hi = boot_ci(boot[a]["frozen"])
        per_arm[a]["test_sensitivity_frozen_ci"] = [round(lo, 4), round(hi, 4)]
        lo, hi = boot_ci(boot[a]["matched"])
        per_arm[a]["test_sensitivity_matched_ci"] = [round(lo, 4), round(hi, 4)]

    deltas = {}
    for pair in (("E5", "E0"), ("E3", "E0"), ("E5", "E3"), ("E1", "E0")):
        a, b = pair
        y, pa = test_scores[a]
        _, pb = test_scores[b]
        point_f = per_arm[a]["test_sensitivity_frozen"] - per_arm[b]["test_sensitivity_frozen"]
        point_m = per_arm[a]["test_sensitivity_matched"] - per_arm[b]["test_sensitivity_matched"]
        dfs, dms = [], []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            yb = y[idx]
            if len(np.unique(yb)) < 2:
                continue
            pab, pbb = pa[idx], pb[idx]
            sf_a, _ = confusion(yb, pab, per_arm[a]["val_threshold"])
            sf_b, _ = confusion(yb, pbb, per_arm[b]["val_threshold"])
            dfs.append(sf_a - sf_b)
            _, sm_a, _ = sens_at_spec(yb, pab)
            _, sm_b, _ = sens_at_spec(yb, pbb)
            if np.isfinite(sm_a) and np.isfinite(sm_b):
                dms.append(sm_a - sm_b)
        lof, hif = boot_ci(dfs)
        lom, him = boot_ci(dms)
        deltas[f"{a}_minus_{b}"] = {
            "frozen_delta": round(point_f, 4),
            "frozen_ci": [round(lof, 4), round(hif, 4)],
            "matched_delta": round(point_m, 4),
            "matched_ci": [round(lom, 4), round(him, 4)],
        }
        print(
            f"Δ({a}-{b}) frozen {point_f:+.3f} [{lof:+.3f},{hif:+.3f}] | "
            f"matched {point_m:+.3f} [{lom:+.3f},{him:+.3f}]",
            flush=True,
        )

    payload = {"per_arm": per_arm, "paired": deltas, "n_boot": N_BOOT, "target_specificity": TARGET}
    (OUT / "operating_point_analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT / "operating_point_analysis.json")


if __name__ == "__main__":
    main()
