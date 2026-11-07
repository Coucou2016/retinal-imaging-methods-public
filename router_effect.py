#!/usr/bin/env python3
"""How much does the learned quality router actually move the fusion scalar?

All six arms end training with a router within a few thousandths of its
initialisation. That is worth quantifying rather than describing, because it
determines what the ablation can and cannot show.

For a quality vector q and weight triple w the fusion scalar is s = q . w. The
fixed baseline uses w0 = (1, 0.5, 0). The displacement induced by the learned
router is therefore q . (w - w0), which is bounded by max|w - w0| because q lies
on the simplex. This script reports that displacement per image for every arm,
alongside the spread of the scalar itself, so the reader can see that the
router's contribution is two orders of magnitude smaller than the variation in
quality across images.

Writes ``results/real_rfmid/analysis_sel/stats/router_effect.json``.

Usage
-----
    python scripts/router_effect.py
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
STATS = ROOT
OUT = ROOT / "router_effect.json"
ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
W0 = np.array([1.0, 0.5, 0.0])  # good, usable, bad


def main() -> None:
    routers = json.loads((ROOT / "router_coefficients.json").read_text(encoding="utf-8"))
    res = {}

    for arm in ARMS:
        f = ANALYSIS / f"arm_{arm}.npz"
        if not f.is_file() or arm not in routers:
            continue
        d = np.load(f, allow_pickle=True)
        q = 0.5 * (d["ql"].astype(np.float64) + d["qr"].astype(np.float64))
        q = q / q.sum(axis=1, keepdims=True).clip(min=1e-9)

        w = routers[arm].get("mean_weights_good_usable_bad")
        if w is None:
            continue
        w = np.asarray(w, dtype=np.float64)

        s = q @ w
        s0 = q @ W0
        delta = s - s0

        res[arm] = {
            "weights": w.tolist(),
            "weights_minus_fixed": (w - W0).tolist(),
            "scalar_mean": float(s.mean()),
            "scalar_sd": float(s.std()),
            "scalar_range": [float(s.min()), float(s.max())],
            "scalar_baseline_sd": float(s0.std()),
            "induced_shift_mean": float(delta.mean()),
            "induced_shift_max_abs": float(np.abs(delta).max()),
            "induced_shift_sd": float(delta.std()),
            "ratio_shift_to_spread": float(np.abs(delta).std() / max(s0.std(), 1e-12)),
            "corr_shift_with_scalar": float(np.corrcoef(delta, s0)[0, 1])
            if delta.std() > 1e-12
            else 0.0,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}\n")

    hdr = (
        f"{'arm':<4} {'w good':>8} {'w usable':>9} {'w bad':>9} "
        f"{'scalar mean':>12} {'scalar sd':>10} {'max|Δs|':>9} {'Δs sd':>8} {'ratio':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for arm, r in res.items():
        w = r["weights"]
        print(
            f"{arm:<4} {w[0]:8.4f} {w[1]:9.4f} {w[2]:9.4f} {r['scalar_mean']:12.4f} "
            f"{r['scalar_sd']:10.4f} {r['induced_shift_max_abs']:9.5f} "
            f"{r['induced_shift_sd']:8.5f} {r['ratio_shift_to_spread']:7.4f}"
        )


if __name__ == "__main__":
    main()
