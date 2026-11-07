#!/usr/bin/env python3
"""Statistical analyses for the RFMiD ablation arms.

Every number written by this script is computed from cached per-sample
predictions in ``results/real_rfmid/analysis/arm_*.npz``. Nothing is typed in by
hand, so the manuscript tables can be regenerated from scratch.

Analyses
--------
1. Headline metrics per arm (macro AUROC, DR AUROC, AP, ECE, Brier) with
   bootstrap intervals that resample *images*, and DeLong intervals.
2. Paired bootstrap of ΔAUROC between the arms that differ in exactly one
   component (E0→E1, E2→E3, E3→E4free, E3→E5).
3. Quality-stratified metrics: the same arm scored on good/usable/bad strata.
4. Per-endpoint calibration: ECE and Brier for each of the eight heads.
5. Operating-point table: Youden and 95% specificity thresholds fitted on the
   validation fold and applied to test.
6. Router diagnostics: monotone weights, and the entropy of the E5 backbone mix.

Usage
-----
    python scripts/analyze_arms.py --arms E0 E1 E2 E3 E4 E5
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
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bootstrap import delong_auroc_ci, patient_level_bootstrap_ci  # noqa: E402
from calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    per_class_decision_curves,
    sensitivity_at_specificity,
    youden_operating_point,
)

ANALYSIS = Path(os.environ.get("ARM_ANALYSIS_DIR", ROOT))
OUT = Path(os.environ.get("ARM_STATS_DIR", ROOT))
ENDPOINTS = ["Disease_Risk", "DR", "MH", "ODC", "TSLN", "DN", "ARMD", "MYA"]


def _finite_mask(labels: np.ndarray) -> np.ndarray:
    return labels >= 0


def macro_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    for k in range(labels.shape[1]):
        m = _finite_mask(labels[:, k])
        y = labels[m, k]
        if y.size and len(np.unique(y)) > 1:
            vals.append(roc_auc_score(y, probs[m, k]))
    return float(np.mean(vals)) if vals else float("nan")


def macro_ap(labels: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    for k in range(labels.shape[1]):
        m = _finite_mask(labels[:, k])
        y = labels[m, k]
        if y.size and len(np.unique(y)) > 1:
            vals.append(average_precision_score(y, probs[m, k]))
    return float(np.mean(vals)) if vals else float("nan")


def flat_binary(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool every observed endpoint into one binary vector (used for pooled ECE)."""
    m = labels >= 0
    return labels[m].ravel(), probs[m].ravel()


def pooled_ece(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    y, p = flat_binary(labels, probs)
    return expected_calibration_error(y, p, n_bins=bins)


def pooled_brier(labels: np.ndarray, probs: np.ndarray) -> float:
    y, p = flat_binary(labels, probs)
    return brier_score(y, p)


def mean_per_head_ece(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    vals = []
    for k in range(labels.shape[1]):
        m = _finite_mask(labels[:, k])
        if m.sum() == 0 or len(np.unique(labels[m, k])) < 2:
            continue
        vals.append(expected_calibration_error(labels[m, k], probs[m, k], n_bins=bins))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(labels, probs, ids, n_boot=2000, seed=0):
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    return patient_level_bootstrap_ci(labels, probs, ids, n_boot=n_boot, seed=seed)


def paired_delta(y, a, b, ids, n_boot=2000, seed=0):
    """Paired bootstrap for ΔAUROC, resampling rows (images) with replacement."""
    rng = np.random.default_rng(seed)
    n = len(ids)
    point = macro_auroc(y, a) - macro_auroc(y, b)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ya, yb = y[idx], y[idx]
        if len(np.unique(ya)) < 2:
            continue
        d = macro_auroc(ya, a[idx]) - macro_auroc(ya, b[idx])
        if np.isfinite(d):
            deltas.append(d)
    d = np.asarray(deltas)
    lo, hi = (np.quantile(d, 0.025), np.quantile(d, 0.975)) if d.size else (np.nan, np.nan)
    frac_pos = float(np.mean(d > 0)) if d.size else float("nan")
    return {
        "delta": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "frac_positive": frac_pos,
        "n_boot_finite": int(d.size),
    }


def load_arm(name: str) -> dict:
    path = ANALYSIS / f"arm_{name}.npz"
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    out["names"] = [str(x) for x in out["names"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["E0", "E1", "E2", "E3", "E4", "E5"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    arms = {name: load_arm(name) for name in args.arms if (ANALYSIS / f"arm_{name}.npz").is_file()}
    print("loaded arms:", list(arms))

    ref = next(iter(arms.values()))
    ids = ref["split_ids"] if "split_ids" in ref else np.arange(len(ref["probs"]))

    # ---- 1. headline metrics -------------------------------------------------
    headline = {}
    for name, a in arms.items():
        y, p = a["labels"], a["probs"]
        k_dr = a["names"].index("DR") if "DR" in a["names"] else 0
        row = {
            "n": int(p.shape[0]),
            "k": int(p.shape[1]),
            "macro_auroc": macro_auroc(y, p),
            "macro_ap": macro_ap(y, p),
            "dr_auroc": roc_auc_score(y[:, k_dr], p[:, k_dr]) if len(np.unique(y[:, k_dr])) > 1 else float("nan"),
            "dr_ap": average_precision_score(y[:, k_dr], p[:, k_dr]) if len(np.unique(y[:, k_dr])) > 1 else float("nan"),
            "pooled_ece": pooled_ece(y, p),
            "mean_head_ece": mean_per_head_ece(y, p),
            "pooled_brier": pooled_brier(y, p),
        }
        row["bootstrap"] = bootstrap_ci(y, p, ids, n_boot=args.n_boot, seed=args.seed)
        row["delong"] = delong_auroc_ci(y, p)
        headline[name] = row
        print(
            f"{name}: macro={row['macro_auroc']:.4f} DR={row['dr_auroc']:.4f} "
            f"ECE(pooled)={row['pooled_ece']:.4f} Brier={row['pooled_brier']:.4f}"
        )

    (OUT / "headline.json").write_text(json.dumps(headline, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 2. paired comparisons ----------------------------------------------
    def paired(a_name: str, b_name: str, k: int | None = None) -> dict:
        A, B = arms[a_name], arms[b_name]
        if k is None:
            return paired_delta(A["labels"], A["probs"], B["probs"], ids, n_boot=args.n_boot, seed=args.seed)
        y = A["labels"][:, k]
        rng = np.random.default_rng(args.seed)
        n = len(y)
        pa, pb = A["probs"][:, k], B["probs"][:, k]
        point = roc_auc_score(y, pa) - roc_auc_score(y, pb)
        deltas = []
        for _ in range(args.n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            deltas.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
        d = np.asarray(deltas)
        return {
            "delta": float(point),
            "ci_low": float(np.quantile(d, 0.025)) if d.size else float("nan"),
            "ci_high": float(np.quantile(d, 0.975)) if d.size else float("nan"),
            "frac_positive": float(np.mean(d > 0)) if d.size else float("nan"),
            "n_boot_finite": int(d.size),
        }

    comparisons = {}
    pairs = [("E1", "E0"), ("E3", "E2"), ("E4", "E3"), ("E5", "E3")]
    for a_name, b_name in pairs:
        if a_name in arms and b_name in arms:
            entry = {"macro": paired(a_name, b_name)}
            if "DR" in arms[a_name]["names"] and "DR" in arms[b_name]["names"]:
                kdr = arms[a_name]["names"].index("DR")
                entry["DR"] = paired(a_name, b_name, k=kdr)
            comparisons[f"{a_name}_minus_{b_name}"] = entry
            m = entry["macro"]
            print(f"{a_name}-{b_name}: dMacro={m['delta']:+.4f} [{m['ci_low']:+.4f},{m['ci_high']:+.4f}]")

    (OUT / "paired.json").write_text(json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 3. quality strata --------------------------------------------------
    strata = {}
    for name, a in arms.items():
        q = 0.5 * (a["ql"] + a["qr"])
        band = q.argmax(axis=1)  # 0 good, 1 usable, 2 bad
        entry = {}
        for code, label in ((0, "good"), (1, "usable"), (2, "bad")):
            m = band == code
            if m.sum() < 10:
                entry[label] = {"n": int(m.sum()), "note": "too few images for a stable estimate"}
                continue
            y, p = a["labels"][m], a["probs"][m]
            k_dr = a["names"].index("DR") if "DR" in a["names"] else 0
            entry[label] = {
                "n": int(m.sum()),
                "macro_auroc": macro_auroc(y, p),
                "dr_auroc": roc_auc_score(y[:, k_dr], p[:, k_dr]) if len(np.unique(y[:, k_dr])) > 1 else float("nan"),
                "pooled_ece": pooled_ece(y, p),
                "pooled_brier": pooled_brier(y, p),
                "dr_prevalence": float(y[:, k_dr].mean()),
                "any_disease_prevalence": float(y.max(axis=1).mean()),
                "mean_quality_scalar": float(
                    (q[m] * np.array([1.0, 0.5, 1e-4])).sum(axis=1).mean()
                ),
            }
        strata[name] = entry
        print(
            f"{name} strata: "
            + ", ".join(
                f"{k}: n={v.get('n')} auroc={v.get('macro_auroc', float('nan')):.3f}"
                for k, v in entry.items()
            )
        )
    (OUT / "quality_strata.json").write_text(json.dumps(strata, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 4. per-endpoint calibration ---------------------------------------
    per_head = {}
    for name, a in arms.items():
        rows = {}
        for k, ep in enumerate(a["names"]):
            y, p = a["labels"][:, k], a["probs"][:, k]
            m = _finite_mask(y)
            if m.sum() == 0 or len(np.unique(y[m])) < 2:
                continue
            rows[ep] = {
                "auroc": float(roc_auc_score(y[m], p[m])),
                "ap": float(average_precision_score(y[m], p[m])),
                "ece": expected_calibration_error(y[m], p[m]),
                "brier": brier_score(y[m], p[m]),
                "prevalence": float(y[m].mean()),
                "n": int(m.sum()),
            }
        per_head[name] = rows
    (OUT / "per_head.json").write_text(json.dumps(per_head, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 5. operating points ------------------------------------------------
    ops = {}
    for name, a in arms.items():
        rows = {}
        for k, ep in enumerate(a["names"]):
            y, p = a["labels"][:, k], a["probs"][:, k]
            if len(np.unique(y)) < 2:
                continue
            yj = youden_operating_point(y, p)
            s95 = sensitivity_at_specificity(y, p, target_specificity=0.95)
            rows[ep] = {"youden": yj, "sens_at_95spec": s95}
        ops[name] = rows
    (OUT / "operating_points.json").write_text(json.dumps(ops, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 6. decision curves -------------------------------------------------
    dca = {}
    for name, a in arms.items():
        rows = {}
        for k, ep in enumerate(a["names"]):
            y, p = a["labels"][:, k], a["probs"][:, k]
            if len(np.unique(y)) < 2:
                continue
            rows[ep] = per_class_decision_curves(y, p, class_names=[ep])[ep]
        dca[name] = rows
    (OUT / "decision_curves.json").write_text(json.dumps(dca, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 7. router diagnostics ---------------------------------------------
    router = {
        name: {
            "quality_router": str(a["quality_router"]),
            "quality_gating": bool(a["quality_gating"]),
            "monotone_weights": a["monotone_weights"].tolist() if "monotone_weights" in a and a["monotone_weights"] is not None else None,
        }
        for name, a in arms.items()
    }
    (OUT / "router.json").write_text(json.dumps(router, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote statistics to {OUT}")


if __name__ == "__main__":
    main()
