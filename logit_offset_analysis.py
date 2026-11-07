#!/usr/bin/env python3
"""Where the shared head puts its logits, endpoint by endpoint.

The held-out ladder says the multi-task calibration error behaves like a shifted
logit rather than an over-confident one. A zero-parameter prevalence offset
removes most of it, a two-parameter Platt map removes nearly all of it, and a
one-parameter temperature removes none of it.

This script checks the premise directly on the cached logits. For every endpoint
it reports the mean logit, the logit of the endpoint prevalence, and their gap.
If the shared head merely mis-centres each endpoint, the per-endpoint gaps should
be small and should not follow prevalence; if the head compresses the range, the
slope of mean logit on logit prevalence should be well below one.

Writes ``results/real_rfmid/analysis_sel/stats/logit_offset.json``.

Usage
-----
    python scripts/logit_offset_analysis.py
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
OUT = ROOT / "logit_offset.json"
ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def main() -> None:
    per_arm: dict[str, dict] = {}

    for arm in ARMS:
        f = ANALYSIS / f"arm_{arm}.npz"
        if not f.is_file():
            print(f"[skip] {arm}")
            continue
        d = np.load(f, allow_pickle=True)
        names = [str(x) for x in d["names"]]
        z = d["logits"].astype(np.float64)
        y = d["labels"].astype(np.float64)

        heads = {}
        for k, ep in enumerate(names):
            pi = float(y[:, k].mean())
            lp = logit(pi)
            mz = float(z[:, k].mean())
            sd = float(z[:, k].std())
            heads[ep] = {
                "prevalence": pi,
                "logit_prevalence": lp,
                "mean_logit": mz,
                "gap": mz - lp,
                "logit_sd": sd,
                "n": int(y.shape[0]),
            }

        summary: dict = {"n_heads": len(heads)}
        if len(heads) > 2:
            xs = np.array([h["logit_prevalence"] for h in heads.values()])
            ys = np.array([h["mean_logit"] for h in heads.values()])
            gaps = np.abs(ys - xs)
            ece = []
            for _k, _ep in enumerate(heads):
                p = 1.0 / (1.0 + np.exp(-z[:, _k]))
                edges = np.linspace(0.0, 1.0, 11)
                tot = 0.0
                for i in range(10):
                    lo, hi = edges[i], edges[i + 1]
                    m = (p >= lo) & (p <= hi) if i == 9 else (p >= lo) & (p < hi)
                    if m.sum():
                        tot += m.sum() / p.size * abs(y[m, _k].mean() - p[m].mean())
                ece.append(tot)
            ece = np.array(ece)
            slope, intercept = np.polyfit(xs, ys, 1)
            summary.update(
                {
                    "r_mean_logit_vs_logit_prevalence": float(np.corrcoef(xs, ys)[0, 1]),
                    "slope_mean_logit_on_logit_prevalence": float(slope),
                    "intercept_mean_logit_on_logit_prevalence": float(intercept),
                    "mean_abs_gap": float(gaps.mean()),
                    "max_abs_gap": float(gaps.max()),
                    "sd_mean_logit": float(ys.std()),
                    "sd_logit_prevalence": float(xs.std()),
                    "compression_ratio": float(ys.std() / xs.std()),
                    "r_abs_gap_vs_ece": float(np.corrcoef(gaps, ece)[0, 1]),
                    "mean_head_ece": float(ece.mean()),
                }
            )
        per_arm[arm] = {"per_head": heads, "summary": summary}

        s = summary
        print(f"\n{arm}")
        for ep, h in heads.items():
            print(
                f"  {ep:14s} prev {h['prevalence']:.3f}  logit(pi) {h['logit_prevalence']:+.3f}  "
                f"mean z {h['mean_logit']:+.3f}  gap {h['gap']:+.3f}  sd {h['logit_sd']:.3f}"
            )
        if len(heads) > 2:
            print(
                f"  r {s['r_mean_logit_vs_logit_prevalence']:+.3f}  slope "
                f"{s['slope_mean_logit_on_logit_prevalence']:+.3f}  mean|gap| {s['mean_abs_gap']:.3f}  "
                f"r(|gap|, ECE) {s['r_abs_gap_vs_ece']:+.3f}"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "per_arm": per_arm,
                "note": "logits and labels read from the cached test predictions; no refitting",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
