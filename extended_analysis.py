#!/usr/bin/env python3
"""Additional analyses computed from the cached per-sample predictions.

Everything here reads ``results/real_rfmid/analysis_sel/arm_*.npz`` and writes
``results/real_rfmid/analysis_sel/stats/extended.json``. Nothing is typed by hand.

Blocks
------
1. adaptive_ece      equal-frequency (equal-mass) ECE per arm, to test whether the
                     single-task / multi-task calibration gap is a binning artifact.
2. recalibration     oracle recalibration ladder per endpoint: none, temperature
                     (1 parameter), Platt (2 parameters), isotonic (monotone,
                     nonparametric). All fitted on the test labels themselves, so
                     every rung except "none" is an optimistic upper bound.
3. clinical_utility  PPV and NPV at the frozen operating points, plus the number
                     of images that must be read to find one true positive.
4. per_endpoint      AUROC for every endpoint and every arm, with image-level
                     bootstrap intervals, and per-endpoint paired deltas for the
                     three design comparisons.
5. net_benefit       decision-curve net benefit at fixed thresholds.

Usage
-----
    python scripts/extended_analysis.py
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ANALYSIS = ROOT
OUT = ROOT / "extended.json"

ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
ENDPOINTS = ["Disease_Risk", "DR", "MH", "ODC", "TSLN", "DN", "ARMD", "MYA"]
PAIRS = [("E1", "E0"), ("E3", "E2"), ("E4", "E3"), ("E5", "E3")]
N_BOOT = 1000
SEED = 0
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
TARGET_SPEC = 0.95


# --------------------------------------------------------------------- helpers
def ece_equal_width(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
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


def ece_equal_mass(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-frequency ECE: every bin holds the same number of samples."""
    n = y.size
    if n == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, min(n_bins, n))
    total = 0.0
    for idx in chunks:
        if idx.size == 0:
            continue
        total += idx.size / n * abs(float(y[idx].mean()) - float(p[idx].mean()))
    return float(total)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(y: np.ndarray, logit: np.ndarray) -> float:
    """Scalar temperature by grid search on BCE (no scipy dependency)."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 400))
    best, best_t = np.inf, 1.0
    for t in grid:
        p = sigmoid(logit / t)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if bce < best:
            best, best_t = bce, float(t)
    return best_t


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def macro(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


# ------------------------------------------------------------------- load arms
def load() -> dict[str, dict]:
    out = {}
    for a in ARMS:
        f = ANALYSIS / f"arm_{a}.npz"
        if not f.is_file():
            continue
        d = np.load(f, allow_pickle=True)
        out[a] = {
            "probs": d["probs"].astype(np.float64),
            "logits": d["logits"].astype(np.float64),
            "labels": d["labels"].astype(np.float64),
            "names": [str(x) for x in d["names"]],
            "ids": d["split_ids"],
            "monotone_weights": d["monotone_weights"],
        }
    return out


# ------------------------------------------------- 1. equal-width vs equal-mass
def block_adaptive(arms: dict) -> dict:
    res = {}
    for name, a in arms.items():
        y, p = a["labels"], a["probs"]
        per_ep_w, per_ep_m = [], []
        for k in range(y.shape[1]):
            per_ep_w.append(ece_equal_width(y[:, k], p[:, k]))
            per_ep_m.append(ece_equal_mass(y[:, k], p[:, k]))
        res[name] = {
            "mean_head_ece_equal_width": macro(per_ep_w),
            "mean_head_ece_equal_mass": macro(per_ep_m),
            "per_head_equal_width": dict(zip(a["names"], per_ep_w)),
            "per_head_equal_mass": dict(zip(a["names"], per_ep_m)),
            "n": int(y.shape[0]),
            "k": int(y.shape[1]),
        }
    return res


# --------------------------------------------------------- 2. recalibration ladder
def block_recalibration(arms: dict) -> dict:
    res = {}
    for name, a in arms.items():
        y, p, z = a["labels"], a["probs"], a["logits"]
        lvl = {"none": [], "temperature": [], "platt": [], "isotonic": []}
        per_head = {}
        slopes, intercepts = [], []
        for k, ep in enumerate(a["names"]):
            yk, pk, zk = y[:, k], p[:, k], z[:, k]
            if len(np.unique(yk)) < 2:
                continue
            t = fit_temperature(yk, zk)
            p_t = sigmoid(zk / t)

            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(zk.reshape(-1, 1), yk)
            p_p = lr.predict_proba(zk.reshape(-1, 1))[:, 1]
            slope = float(lr.coef_[0, 0])
            icpt = float(lr.intercept_[0])

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(pk, yk)
            p_i = np.clip(iso.predict(pk), 1e-6, 1 - 1e-6)

            row = {
                "temperature": t,
                "platt_slope": slope,
                "platt_intercept": icpt,
                "ece_none": ece_equal_width(yk, pk),
                "ece_temperature": ece_equal_width(yk, p_t),
                "ece_platt": ece_equal_width(yk, p_p),
                "ece_isotonic": ece_equal_width(yk, p_i),
                "brier_none": brier(yk, pk),
                "brier_platt": brier(yk, p_p),
                "brier_isotonic": brier(yk, p_i),
            }
            per_head[ep] = row
            slopes.append(slope)
            intercepts.append(icpt)
            for key in lvl:
                lvl[key].append(row[f"ece_{key}"] if key != "none" else row["ece_none"])

        res[name] = {
            "per_head": per_head,
            "macro_ece_none": macro(lvl["none"]),
            "macro_ece_temperature": macro(lvl["temperature"]),
            "macro_ece_platt": macro(lvl["platt"]),
            "macro_ece_isotonic": macro(lvl["isotonic"]),
            "mean_platt_slope": macro(slopes),
            "mean_platt_intercept": macro(intercepts),
            "n_heads": len(per_head),
        }
    return res


# ------------------------------------------------------- 3. clinical utility / PPV
def ppv(sens: float, spec: float, prev: float) -> float:
    num = sens * prev
    den = num + (1.0 - spec) * (1.0 - prev)
    return float(num / den) if den > 0 else float("nan")


def npv(sens: float, spec: float, prev: float) -> float:
    num = spec * (1.0 - prev)
    den = num + (1.0 - sens) * prev
    return float(num / den) if den > 0 else float("nan")


def operating_points(y: np.ndarray, p: np.ndarray) -> dict:
    """Youden and 95%-specificity thresholds, plus sensitivity/specificity at each."""
    fpr, tpr, thr = roc_curve_local(y, p)
    spec = 1.0 - fpr
    j = tpr - fpr
    i = int(np.argmax(j))
    yj = {"threshold": float(thr[i]), "sensitivity": float(tpr[i]), "specificity": float(spec[i])}

    ok = np.where(spec >= TARGET_SPEC - 1e-9)[0]
    if ok.size:
        cand = ok[np.argmax(tpr[ok])]
    else:
        cand = int(np.argmax(spec))
    s95 = {
        "threshold": float(thr[cand]),
        "sensitivity": float(tpr[cand]),
        "specificity": float(spec[cand]),
    }
    return {"youden": yj, "sens_at_95spec": s95}


def roc_curve_local(y: np.ndarray, p: np.ndarray):
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    distinct = np.where(np.diff(p))[0]
    thr_idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[thr_idx]
    fps = 1 + thr_idx - tps
    tpr = tps / max(y.sum(), 1)
    fpr = fps / max((1 - y).sum(), 1)
    thr = p[thr_idx]
    return np.r_[0.0, fpr], np.r_[0.0, tpr], np.r_[np.inf, thr]


def block_utility(arms: dict) -> dict:
    res = {}
    for name, a in arms.items():
        y, p = a["labels"], a["probs"]
        if "DR" not in a["names"]:
            continue
        k = a["names"].index("DR")
        yk, pk = y[:, k], p[:, k]
        prev = float(yk.mean())
        op = operating_points(yk, pk)
        row = {"prevalence": prev, "n": int(yk.size)}
        for label, key in (("youden", "youden"), ("spec95", "sens_at_95spec")):
            s, c = op[key]["sensitivity"], op[key]["specificity"]
            row[label] = {
                "threshold": op[key]["threshold"],
                "sensitivity": s,
                "specificity": c,
                "ppv": ppv(s, c, prev),
                "npv": npv(s, c, prev),
                "images_per_true_positive": float(1.0 / ppv(s, c, prev)) if ppv(s, c, prev) > 0 else float("nan"),
                "missed_per_100": float(100.0 * (1.0 - s) * prev),
                "false_alarms_per_100": float(100.0 * (1.0 - c) * (1.0 - prev)),
            }
        res[name] = row
    return res


# ------------------------------------------------------------- 4. per endpoint
def auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def boot_auroc_ci(y, p, ids, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = y.size
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, pp))
    v = np.asarray(vals)
    if v.size == 0:
        return {"estimate": auroc(y, p), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    return {
        "estimate": auroc(y, p),
        "ci_low": float(np.quantile(v, 0.025)),
        "ci_high": float(np.quantile(v, 0.975)),
        "n_boot": int(v.size),
    }


def block_per_endpoint(arms: dict) -> dict:
    res = {"auroc": {}, "delta": {}}
    for name, a in arms.items():
        rows = {}
        for k, ep in enumerate(a["names"]):
            y, p = a["labels"][:, k], a["probs"][:, k]
            if len(np.unique(y)) < 2:
                continue
            rows[ep] = boot_auroc_ci(y, p, a["ids"])
        res["auroc"][name] = rows

    rng_master = np.random.default_rng(SEED)
    for hi, lo in PAIRS:
        if hi not in arms or lo not in arms:
            continue
        A, B = arms[hi], arms[lo]
        ids = A["ids"]
        n = ids.size
        base = rng_master.integers(0, n, (N_BOOT, n))
        entry = {}
        for k, ep in enumerate(A["names"]):
            y = A["labels"][:, k]
            if len(np.unique(y)) < 2:
                continue
            pa, pb = A["probs"][:, k], B["probs"][:, k]
            point = roc_auc_score(y, pa) - roc_auc_score(y, pb)
            vals = []
            for b in range(N_BOOT):
                idx = base[b]
                yy = y[idx]
                if len(np.unique(yy)) < 2:
                    continue
                vals.append(roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx]))
            v = np.asarray(vals)
            entry[ep] = {
                "delta": float(point),
                "ci_low": float(np.quantile(v, 0.025)) if v.size else float("nan"),
                "ci_high": float(np.quantile(v, 0.975)) if v.size else float("nan"),
                "frac_positive": float(np.mean(v > 0)) if v.size else float("nan"),
            }
        res["delta"][f"{hi}_minus_{lo}"] = entry
    return res


# ---------------------------------------------------------------- 5. net benefit
def block_net_benefit(arms: dict) -> dict:
    res = {}
    for name, a in arms.items():
        rows = {}
        for k, ep in enumerate(a["names"]):
            y, p = a["labels"][:, k], a["probs"][:, k]
            prev = float(y.mean())
            curve = []
            for t in THRESHOLDS:
                tp = float(np.sum((p >= t) & (y == 1)))
                fp = float(np.sum((p >= t) & (y == 0)))
                nb = tp / y.size - fp / y.size * (t / (1 - t))
                treat_all = prev - (1 - prev) * (t / (1 - t))
                curve.append({"threshold": t, "net_benefit": nb, "treat_all": treat_all})
            rows[ep] = {"prevalence": prev, "curve": curve}
        res[name] = rows
    return res


def main() -> None:
    arms = load()
    payload = {
        "config": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "target_specificity": TARGET_SPEC,
            "thresholds": THRESHOLDS,
            "source": "results/real_rfmid/analysis_sel/arm_*.npz",
        },
        "adaptive_ece": block_adaptive(arms),
        "recalibration": block_recalibration(arms),
        "clinical_utility": block_utility(arms),
        "per_endpoint": block_per_endpoint(arms),
        "net_benefit": block_net_benefit(arms),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")

    print("\n-- equal-width vs equal-mass ECE --")
    for a, r in payload["adaptive_ece"].items():
        print(f"{a}: width {r['mean_head_ece_equal_width']:.3f}  mass {r['mean_head_ece_equal_mass']:.3f}")

    print("\n-- oracle recalibration (macro ECE) --")
    for a, r in payload["recalibration"].items():
        print(
            f"{a}: none {r['macro_ece_none']:.3f}  T {r['macro_ece_temperature']:.3f}  "
            f"Platt {r['macro_ece_platt']:.3f}  isotonic {r['macro_ece_isotonic']:.3f}  "
            f"slope {r['mean_platt_slope']:.2f}"
        )

    print("\n-- PPV at frozen 95% specificity (DR) --")
    for a, r in payload["clinical_utility"].items():
        s = r["spec95"]
        print(f"{a}: sens {s['sensitivity']:.3f} PPV {s['ppv']:.3f} NPV {s['npv']:.3f} imgs/TP {s['images_per_true_positive']:.1f}")

    print("\n-- per-endpoint AUROC --")
    for a, rows in payload["per_endpoint"]["auroc"].items():
        print(a, {k: round(v["estimate"], 3) for k, v in rows.items()})

    print("\n-- per-endpoint paired delta E5-E3 --")
    for ep, r in payload["per_endpoint"]["delta"].get("E5_minus_E3", {}).items():
        print(f"{ep}: {r['delta']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] share {r['frac_positive']:.3f}")


if __name__ == "__main__":
    main()
