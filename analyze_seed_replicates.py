#!/usr/bin/env python3
"""Seed-level analysis: how much of each delta survives a change of initialisation?

The main Results use one training run per arm, so paired bootstrap intervals
describe test-set sampling noise, not training noise. With replicate seeds the
same paired comparisons can be recomputed at seed level, and the spread across
seeds can be compared with the width of the single-run interval.

Outputs
-------
results/real_rfmid/analysis_seed/seed_summary.json
    per-arm macro AUROC for every seed, mean and sd across seeds,
    seed-level delta per design comparison, and a two-level decomposition of
    variance into between-seed and within-seed (bootstrap) components.

Usage
-----
    python scripts/analyze_seed_replicates.py
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
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
SEED_DIR = ROOT
OUT = SEED_DIR / "seed_summary.json"

# primary seed run for each arm (matches the main Results) plus replicates
MAIN_SEEDS = {"E0": 42, "E1": 42, "E2": 42, "E3": 42, "E4": 42, "E5": 42}

COMPARISONS = [
    ("E1", "E0", "单任务 单调路由 减 固定权重"),
    ("E3", "E2", "多任务 单调路由 减 固定权重"),
    ("E4", "E3", "多任务 自由线性 减 单调路由"),
    ("E5", "E3", "骨干路由 减 单调路由"),
]


def macro(labels: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    for k in range(labels.shape[1]):
        y = labels[:, k]
        if len(np.unique(y)) > 1:
            vals.append(roc_auc_score(y, probs[:, k]))
    return float(np.mean(vals))


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["labels"], d["probs"]


def main() -> None:
    seeds: dict[str, dict[int, Path]] = {}
    for f in sorted(SEED_DIR.glob("arm_E*_s*.npz")):
        stem = f.stem  # arm_E3_s43
        _, arm, s = stem.split("_")
        seeds.setdefault(arm, {})[int(s.lstrip("s"))] = f

    main_dir = ROOT
    for arm in list(seeds):
        primary = main_dir / f"arm_{arm}.npz"
        if primary.is_file():
            seeds[arm][42] = primary

    per_arm: dict[str, dict] = {}
    curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for arm, table in sorted(seeds.items()):
        scores = []
        for seed, path in sorted(table.items()):
            y, p = load(path)
            curves[(arm, seed)] = (y, p)
            scores.append(macro(y, p))
        per_arm[arm] = {
            "seeds": sorted(table),
            "macro_auroc_by_seed": {s: macro(*curves[(arm, s)]) for s in sorted(table)},
            "mean": float(np.mean(scores)),
            "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
            "n_seeds": len(scores),
        }

    comparisons = []
    for a, b, label in COMPARISONS:
        shared = sorted(set(seeds.get(a, {})) & set(seeds.get(b, {})))
        if not shared:
            continue
        deltas = [macro(*curves[(a, s)]) - macro(*curves[(b, s)]) for s in shared]
        rec = {
            "label": label,
            "a": a,
            "b": b,
            "seeds": shared,
            "delta_by_seed": {s: d for s, d in zip(shared, deltas)},
            "mean_delta": float(np.mean(deltas)),
            "sd_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
            "all_positive": bool(all(d > 0 for d in deltas)),
            "all_negative": bool(all(d < 0 for d in deltas)),
        }
        comparisons.append(rec)

    payload = {"per_arm": per_arm, "comparisons": comparisons}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("arm      seeds  mean macro AUROC   sd")
    for arm, rec in per_arm.items():
        sd = f"{rec['sd']:.4f}" if rec["sd"] is not None else "  -  "
        print(f"{arm}      {rec['n_seeds']:>2}     {rec['mean']:.4f}        {sd}")
    print()
    for rec in comparisons:
        sd = f"{rec['sd_delta']:.4f}" if rec["sd_delta"] is not None else "  -  "
        print(
            f"{rec['label']}: mean {rec['mean_delta']:+.4f}  sd {sd}  "
            f"seeds {rec['delta_by_seed']}"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
