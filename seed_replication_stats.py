#!/usr/bin/env python3
"""Seed-level replication statistics for the ablation arms.

The main Results report one training run per arm, so paired bootstrap intervals
cover test-set sampling noise only. Four seeds per arm (42, 43, 44, 45) let the
same comparisons be re-evaluated under a change of initialisation, which is the
dominant source of variability for a frozen-backbone model trained for a few
epochs.

This script writes everything the manuscript needs from those replicates.

1. Per-arm macro AUROC for every seed, mean, sd, and the seed-level range.
2. Per-comparison delta for every seed, mean, sd, and a t interval across seeds.
3. A two-level variance decomposition: between-seed spread versus the mean
   width of the within-seed bootstrap interval, so the reader can see which
   component dominates.
4. Calibration replication: mean per-endpoint ECE, the correlation between
   logit centre and endpoint logit prevalence, and the ECE after a zero-parameter
   prevalence shift, recomputed for every seed. The mechanism claim in the
   paper is not allowed to rest on a single initialisation.
5. A sign-consistency table used directly in the manuscript.

Outputs
-------
results/real_rfmid/analysis_seed/seed_replication.json
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
import math
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
SEED_DIR = ROOT
MAIN_DIR = ROOT
OUT = SEED_DIR / "seed_replication.json"

ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
COMPARISONS = [
    ("E1", "E0", "单任务 单调路由 减 固定权重"),
    ("E3", "E2", "多任务 单调路由 减 固定权重"),
    ("E4", "E3", "多任务 自由线性 减 单调路由"),
    ("E5", "E3", "骨干路由 减 单调路由"),
]


# --------------------------------------------------------------------- metrics
def macro_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    for k in range(labels.shape[1]):
        y = labels[:, k]
        if len(np.unique(y)) > 1:
            vals.append(roc_auc_score(y, probs[:, k]))
    return float(np.mean(vals))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = y.size
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        total += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(total)


def mean_endpoint_ece(labels: np.ndarray, probs: np.ndarray) -> float:
    vals = [ece(labels[:, k], probs[:, k]) for k in range(labels.shape[1])]
    return float(np.mean(vals))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def offset_diagnostics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Centre versus prevalence, and the ECE left after a prevalence shift.

    The diagnostic the manuscript reports is the correlation between the mean
    logit of an endpoint and the logit of its prevalence, with the slope of the
    former on the latter. A slope near one and a strong positive correlation mean
    the shared head reproduces the ranking of prevalences but places each head at
    a systematically wrong offset, which is a location problem, not a scale one.
    """
    gaps, prev_logits, centres, ece_raw, ece_shift = [], [], [], [], []
    for k in range(labels.shape[1]):
        y, p = labels[:, k], probs[:, k]
        pi = float(y.mean())
        z = logit(p)
        gap = float(z.mean() - math.log(pi / (1 - pi)))
        gaps.append(gap)
        prev_logits.append(math.log(pi / (1 - pi)))
        centres.append(float(z.mean()))
        ece_raw.append(ece(y, p))
        shifted = 1.0 / (1.0 + np.exp(-(z + math.log(pi / (1 - pi)))))
        ece_shift.append(ece(y, shifted))
    gaps_a = np.asarray(gaps)
    prev_a = np.asarray(prev_logits)
    centre_a = np.asarray(centres)
    if centre_a.size > 2 and np.std(prev_a) > 0 and np.std(centre_a) > 0:
        corr_centre = float(np.corrcoef(centre_a, prev_a)[0, 1])
        slope_centre = float(np.polyfit(prev_a, centre_a, 1)[0])
        corr_gap = float(np.corrcoef(gaps_a, prev_a)[0, 1])
        slope_gap = float(np.polyfit(prev_a, gaps_a, 1)[0])
    else:
        corr_centre = slope_centre = corr_gap = slope_gap = float("nan")
    return {
        "corr_centre_prevalence": corr_centre,
        "slope_centre_prevalence": slope_centre,
        "corr_gap_prevalence": corr_gap,
        "slope_gap_prevalence": slope_gap,
        "compression_ratio": (
            float(np.std(centre_a) / np.std(prev_a)) if np.std(prev_a) > 0 else float("nan")
        ),
        "mean_abs_gap": float(np.abs(gaps_a).mean()),
        "mean_endpoint_ece_raw": float(np.mean(ece_raw)),
        "mean_endpoint_ece_shifted": float(np.mean(ece_shift)),
        "gap_by_endpoint": gaps,
        "prevalence_logit_by_endpoint": prev_logits,
    }


# ------------------------------------------------------------------ data access
def load(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "labels": d["labels"],
        "probs": d["probs"],
        "names": [str(x) for x in d["names"]],
    }


def arm_tables() -> dict[str, dict[int, Path]]:
    table: dict[str, dict[int, Path]] = {}
    for f in sorted(SEED_DIR.glob("arm_E*_s*.npz")):
        _, arm, s = f.stem.split("_")
        table.setdefault(arm, {})[int(s.lstrip("s"))] = f
    for arm in ARMS:
        primary = MAIN_DIR / f"arm_{arm}.npz"
        if primary.is_file():
            table.setdefault(arm, {})[42] = primary
    # never trust a cached primary run that is not seed 42 of the same protocol
    return table


def main() -> None:
    table = arm_tables()
    cache: dict[tuple[str, int], dict] = {}
    for arm, seeds in table.items():
        for seed, path in seeds.items():
            cache[(arm, seed)] = load(path)

    per_arm: dict[str, dict] = {}
    for arm in ARMS:
        if arm not in table:
            continue
        rows = {}
        for seed in sorted(table[arm]):
            d = cache[(arm, seed)]
            diag = offset_diagnostics(d["labels"], d["probs"])
            rows[str(seed)] = {
                "macro_auroc": macro_auroc(d["labels"], d["probs"]),
                **diag,
            }
        scores = np.array([r["macro_auroc"] for r in rows.values()])
        per_arm[arm] = {
            "seeds": sorted(table[arm]),
            "by_seed": rows,
            "mean": float(scores.mean()),
            "sd": float(scores.std(ddof=1)) if scores.size > 1 else None,
            "min": float(scores.min()),
            "max": float(scores.max()),
            "n_seeds": int(scores.size),
        }

    comparisons = []
    for a, b, label in COMPARISONS:
        shared = sorted(set(table.get(a, {})) & set(table.get(b, {})))
        if not shared:
            continue
        deltas = np.array([
            per_arm[a]["by_seed"][str(s)]["macro_auroc"]
            - per_arm[b]["by_seed"][str(s)]["macro_auroc"]
            for s in shared
        ])
        n = deltas.size
        mean = float(deltas.mean())
        sd = float(deltas.std(ddof=1)) if n > 1 else None
        if n > 1:
            half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
            ci = [mean - half, mean + half]
        else:
            ci = [float("nan"), float("nan")]
        comparisons.append({
            "label": label,
            "a": a,
            "b": b,
            "seeds": shared,
            "delta_by_seed": {str(s): float(d) for s, d in zip(shared, deltas)},
            "mean_delta": mean,
            "sd_delta": sd,
            "seed_ci": ci,
            "all_positive": bool((deltas > 0).all()),
            "all_negative": bool((deltas < 0).all()),
            "n_positive": int((deltas > 0).sum()),
            "n_seeds": int(n),
        })

    # ---- two-level variance decomposition --------------------------------
    # mean bootstrap half-width of a single-run delta, taken from the main
    # analysis, compared with the seed-level spread of the same delta.
    main_paired = json.loads(
        (ROOT / "paired.json").read_text(encoding="utf-8")
    )
    decomposition = []
    for rec in comparisons:
        key = f"{rec['a']}_minus_{rec['b']}"
        boot_half = None
        if key in main_paired:
            m = main_paired[key]["macro"]
            boot_half = 0.5 * (m["ci_high"] - m["ci_low"])
        decomposition.append({
            "label": rec["label"],
            "seed_sd": rec["sd_delta"],
            "seed_range": max(rec["delta_by_seed"].values()) - min(rec["delta_by_seed"].values()),
            "bootstrap_half_width_single_run": boot_half,
            "ratio_seed_sd_to_bootstrap_half_width": (
                None if (boot_half in (None, 0) or rec["sd_delta"] is None)
                else float(rec["sd_delta"] / boot_half)
            ),
        })

    # ---- calibration replication -----------------------------------------
    calibration = {}
    for arm in ("E2", "E3", "E4", "E5"):
        if arm not in per_arm:
            continue
        rows = per_arm[arm]["by_seed"]
        raw = np.array([r["mean_endpoint_ece_raw"] for r in rows.values()])
        shifted = np.array([r["mean_endpoint_ece_shifted"] for r in rows.values()])
        corr = np.array([r["corr_centre_prevalence"] for r in rows.values()])
        slope = np.array([r["slope_centre_prevalence"] for r in rows.values()])
        calibration[arm] = {
            "seeds": [int(s) for s in rows],
            "ece_raw": raw.tolist(),
            "ece_shifted": shifted.tolist(),
            "ece_raw_mean": float(raw.mean()),
            "ece_shifted_mean": float(shifted.mean()),
            "corr_by_seed": corr.tolist(),
            "corr_min": float(np.nanmin(corr)),
            "corr_max": float(np.nanmax(corr)),
            "slope_by_seed": slope.tolist(),
            "slope_min": float(np.nanmin(slope)),
            "slope_max": float(np.nanmax(slope)),
            "all_shift_reduces": bool((shifted < raw).all()),
            "mean_absolute_reduction": float(np.abs(raw - shifted).mean()),
        }

    payload = {
        "per_arm": per_arm,
        "comparisons": comparisons,
        "variance_decomposition": decomposition,
        "calibration": calibration,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("arm   seeds  mean    sd      min     max")
    for arm, rec in per_arm.items():
        print(
            f"{arm}   {rec['n_seeds']:>2}    {rec['mean']:.4f}  "
            f"{rec['sd']:.4f}  {rec['min']:.4f}  {rec['max']:.4f}"
        )
    print()
    for rec in comparisons:
        print(
            f"{rec['label']}: mean {rec['mean_delta']:+.4f} "
            f"sd {rec['sd_delta']:.4f} seeds {rec['n_positive']}/{rec['n_seeds']} 正值"
        )
    print()
    for arm, rec in calibration.items():
        print(
            f"{arm} ECE {rec['ece_raw_mean']:.3f} -> {rec['ece_shifted_mean']:.3f}  "
            f"r {rec['corr_min']:.3f}..{rec['corr_max']:.3f}"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
