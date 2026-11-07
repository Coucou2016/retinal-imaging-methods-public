#!/usr/bin/env python3
"""追加证据：配对差值的自助法分布，以及三种温度拟合口径的对照。

配对差值此前只报告点估计与区间端点。把它们背后的重采样分布画出来，读者可以
直接看到支持零假设的那部分质量有多大，这比两个端点数字直观得多。温度口径的
对照把正文第 5.3 节的三次拟合从叙述变成一张可核对的表。

输出
----
results/real_rfmid/analysis_sel/stats/extra_evidence.json
results/real_rfmid/analysis_sel/extra_bootstrap.npz

用法
----
    python scripts/extra_evidence.py
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ANALYSIS = ROOT
STATS = ROOT
OUT_JSON = ROOT / "extra_evidence.json"
OUT_NPZ = ANALYSIS / "extra_bootstrap.npz"

N_BOOT = 1000
SEED = 0

PAIRS = [
    ("E1_minus_E0", "E1", "E0", "DR"),
    ("E3_minus_E2", "E3", "E2", "macro"),
    ("E4_minus_E3", "E4", "E3", "macro"),
    ("E5_minus_E3", "E5", "E3", "macro"),
]


def load(arm: str) -> dict:
    d = np.load(ANALYSIS / f"arm_{arm}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for k in range(y.shape[1]):
        m = y[:, k] >= 0
        if m.sum() == 0 or len(np.unique(y[m, k])) < 2:
            continue
        vals.append(roc_auc_score(y[m, k], p[m, k]))
    return float(np.mean(vals))


def dr_column(arm: dict) -> int:
    names = [str(x) for x in arm["names"]]
    return names.index("DR") if "DR" in names else 0


def score(arm: dict, rows: np.ndarray, which: str) -> float:
    y, p = arm["labels"][rows], arm["probs"][rows]
    if which == "DR":
        k = dr_column(arm)
        m = y[:, k] >= 0
        return float(roc_auc_score(y[m, k], p[m, k]))
    return macro_auroc(y, p)


def main() -> int:
    rng = np.random.default_rng(SEED)
    arms = {a: load(a) for a in ("E0", "E1", "E2", "E3", "E4", "E5")}
    names = [str(x) for x in arms["E5"]["names"]]
    dr_index = names.index("DR")
    n = arms["E5"]["probs"].shape[0]

    idx = rng.integers(0, n, size=(N_BOOT, n))
    dists = {}
    summary = {}
    for key, hi, lo, which in PAIRS:
        a, b = arms[hi], arms[lo]
        if a["probs"].shape[1] != b["probs"].shape[1]:
            # 单任务臂与多任务臂的终点集合不同，只比较同一终点
            which_eff = "DR"
        else:
            which_eff = which
        d = np.empty(N_BOOT, dtype=np.float64)
        for t in range(N_BOOT):
            rows = idx[t]
            d[t] = score(a, rows, which_eff) - score(b, rows, which_eff)
        dists[key] = d
        summary[key] = {
            "delta": float(d.mean()),
            "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)),
            "frac_negative": float((d < 0).mean()),
            "frac_positive": float((d > 0).mean()),
            "frac_within_pm_0p002": float((np.abs(d) < 0.002).mean()),
            "frac_within_pm_0p005": float((np.abs(d) < 0.005).mean()),
            "n_boot": int(N_BOOT),
        }

    np.savez_compressed(OUT_NPZ, **{k: v for k, v in dists.items()})

    schemes = json.loads((ROOT / "temperature_schemes.json").read_text(encoding="utf-8"))
    oracle = json.loads((ROOT / "calibration_oracle.json").read_text(encoding="utf-8"))
    temp = {}
    for arm in ("E0", "E1", "E2", "E3", "E4", "E5"):
        s = schemes[arm]
        temp[arm] = {
            "true_single": float(s["temperature_single"]),
            "raw": float(s["none"]["mean_head_ece"]),
            "single": float(s["single"]["mean_head_ece"]),
            "per_head": float(s["per_head"]["mean_head_ece"]),
            "worst_after_single": float(s["single"]["worst_head_ece"]),
            "oracle": float(oracle[arm]["mean_head_ece_oracle"]),
            "oracle_gain": float(oracle[arm]["mean_head_ece_gain"]),
        }

    payload = {"config": {"n_boot": N_BOOT, "seed": SEED}, "pairs": summary, "temperature": temp}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print("pair deltas")
    for k, v in summary.items():
        print(
            f"  {k}: delta {v['delta']:+.4f} ci [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] "
            f"P(<0) {v['frac_negative']:.3f} |d|<0.002 {v['frac_within_pm_0p002']:.3f}"
        )
    print("temperature schemes")
    for a, v in temp.items():
        print(
            f"  {a}: T={v['true_single']:.3f} raw {v['raw']:.4f} single {v['single']:.4f} "
            f"per-head {v['per_head']:.4f} oracle {v['oracle']:.4f}"
        )
    print("wrote", OUT_JSON.name, OUT_NPZ.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
