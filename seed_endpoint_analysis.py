#!/usr/bin/env python3
"""Seed-level comparison on the one endpoint every arm shares.

The single-task arms (E0, E1) only have a diabetic-retinopathy head, so their
macro AUROC is that endpoint and cannot be compared with the eight-endpoint
macro average of E2-E5. Diabetic retinopathy is the only endpoint present in all
six arms, which makes it the only like-for-like axis for a six-arm comparison.

This script reports, per seed:

1. DR AUROC for every arm, mean and sd across the four training runs.
2. DR-level paired deltas for the design comparisons, so the manuscript can say
   whether a conclusion depends on the eight-endpoint average.
3. A variance ratio test between arms, because the backbone-routing arm is
   visibly more stable across seeds and that needs a number behind it.
4. Per-endpoint seed-level deltas for E5 minus E3, to show where the robust
   gain is concentrated.

Outputs
-------
results/real_rfmid/analysis_seed/seed_endpoint_analysis.json
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
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
SEED_DIR = ROOT
MAIN_DIR = ROOT
OUT = SEED_DIR / "seed_endpoint_analysis.json"

ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
CN = {
    "Disease_Risk": "任意病变",
    "DR": "糖尿病视网膜病变",
    "MH": "屈光介质混浊",
    "ODC": "视盘杯盘比异常",
    "TSLN": "豹纹状眼底",
    "DN": "玻璃膜疣",
    "ARMD": "年龄相关性黄斑变性",
    "MYA": "近视性改变",
}


def load_curves() -> dict[str, dict[int, dict]]:
    table: dict[str, dict[int, dict]] = {}
    for arm in ARMS:
        for p in sorted(SEED_DIR.glob(f"arm_{arm}_s*.npz")):
            seed = int(p.stem.split("_s")[1])
            d = np.load(p, allow_pickle=True)
            table.setdefault(arm, {})[seed] = {
                "names": [str(x) for x in d["names"]],
                "labels": d["labels"],
                "probs": d["probs"],
            }
        primary = MAIN_DIR / f"arm_{arm}.npz"
        if primary.is_file():
            d = np.load(primary, allow_pickle=True)
            table.setdefault(arm, {})[42] = {
                "names": [str(x) for x in d["names"]],
                "labels": d["labels"],
                "probs": d["probs"],
            }
    return table


def auroc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def main() -> None:
    table = load_curves()

    dr: dict[str, dict] = {}
    for arm in ARMS:
        if arm not in table:
            continue
        by_seed = {}
        for seed, rec in sorted(table[arm].items()):
            k = rec["names"].index("DR")
            by_seed[str(seed)] = auroc(rec["labels"][:, k], rec["probs"][:, k])
        vals = np.array(list(by_seed.values()))
        dr[arm] = {
            "by_seed": by_seed,
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n_seeds": int(vals.size),
        }

    comparisons = []
    for a, b in [("E1", "E0"), ("E3", "E2"), ("E4", "E3"), ("E5", "E3"), ("E5", "E0")]:
        shared = sorted(set(table[a]) & set(table[b]))
        deltas = []
        for s in shared:
            ka = table[a][s]["names"].index("DR")
            kb = table[b][s]["names"].index("DR")
            ya, yb = table[a][s]["labels"], table[b][s]["labels"]
            deltas.append(
                auroc(ya[:, ka], table[a][s]["probs"][:, ka])
                - auroc(yb[:, kb], table[b][s]["probs"][:, kb])
            )
        d = np.array(deltas)
        n = d.size
        sd = float(d.std(ddof=1)) if n > 1 else float("nan")
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
        comparisons.append({
            "a": a,
            "b": b,
            "endpoint": "DR",
            "seeds": shared,
            "delta_by_seed": {str(s): float(v) for s, v in zip(shared, deltas)},
            "mean_delta": float(d.mean()),
            "sd_delta": sd,
            "seed_ci": [float(d.mean() - half), float(d.mean() + half)],
            "n_positive": int((d > 0).sum()),
            "n_seeds": int(n),
        })

    # variance stability: is the backbone-routing arm tighter across seeds?
    variance = []
    for a, b in [("E3", "E5"), ("E2", "E5"), ("E0", "E5"), ("E3", "E0")]:
        va = np.array(list(dr[a]["by_seed"].values())) if a in dr else None
        vb = np.array(list(dr[b]["by_seed"].values())) if b in dr else None
        if va is None or vb is None or va.size < 3 or vb.size < 3:
            continue
        lev = stats.levene(va, vb, center="median")
        variance.append({
            "a": a,
            "b": b,
            "sd_a": float(va.std(ddof=1)),
            "sd_b": float(vb.std(ddof=1)),
            "ratio": float(va.std(ddof=1) / vb.std(ddof=1)) if vb.std(ddof=1) > 0 else None,
            "levene_p": float(lev.pvalue),
        })

    # per-endpoint E5 - E3 across seeds
    endpoints = {}
    shared = sorted(set(table["E5"]) & set(table["E3"]))
    names = table["E5"][shared[0]]["names"]
    for k, ep in enumerate(names):
        vals = []
        for s in shared:
            vals.append(
                auroc(table["E5"][s]["labels"][:, k], table["E5"][s]["probs"][:, k])
                - auroc(table["E3"][s]["labels"][:, k], table["E3"][s]["probs"][:, k])
            )
        v = np.array(vals)
        endpoints[ep] = {
            "label": CN.get(ep, ep),
            "by_seed": {str(s): float(x) for s, x in zip(shared, vals)},
            "mean_delta": float(v.mean()),
            "sd_delta": float(v.std(ddof=1)),
            "n_positive": int((v > 0).sum()),
            "n_seeds": int(v.size),
            "all_positive": bool((v > 0).all()),
        }

    payload = {
        "dr_auroc": dr,
        "dr_comparisons": comparisons,
        "variance_stability": variance,
        "e5_minus_e3_by_endpoint": endpoints,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("DR AUROC by arm (4 seeds)")
    for arm, rec in dr.items():
        print(f"  {arm}  mean {rec['mean']:.4f}  sd {rec['sd']:.4f}  "
              f"range {rec['min']:.4f}-{rec['max']:.4f}")
    print("\nDR-level deltas")
    for rec in comparisons:
        print(f"  {rec['a']}-{rec['b']}: {rec['mean_delta']:+.4f} "
              f"sd {rec['sd_delta']:.4f} ci {rec['seed_ci'][0]:+.4f}..{rec['seed_ci'][1]:+.4f} "
              f"pos {rec['n_positive']}/{rec['n_seeds']}")
    print("\nvariance stability (DR AUROC)")
    for rec in variance:
        print(f"  sd({rec['a']}) {rec['sd_a']:.4f} vs sd({rec['b']}) {rec['sd_b']:.4f} "
              f"ratio {rec['ratio']:.2f}  Levene p {rec['levene_p']:.3f}")
    print("\nE5 - E3 by endpoint")
    for ep, rec in endpoints.items():
        print(f"  {rec['label']:12s} {rec['mean_delta']:+.4f} sd {rec['sd_delta']:.4f} "
              f"pos {rec['n_positive']}/{rec['n_seeds']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
