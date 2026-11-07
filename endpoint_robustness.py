#!/usr/bin/env python3
"""How much of each macro difference rests on a single endpoint.

The macro AUROC averages eight endpoints with very different prevalence and very
different separability.  A macro difference of +0.010 can therefore be produced
by one strong endpoint and seven flat ones.  This script recomputes every design
comparison after dropping one endpoint at a time, which shows whether the sign
of the difference survives the removal of any single endpoint.

Writes ``results/real_rfmid/analysis_sel/stats/endpoint_robustness.json``.

Usage
-----
    python scripts/endpoint_robustness.py
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

ROOT = Path(__file__).resolve().parent
STATS = ROOT
OUT = ROOT / "endpoint_robustness.json"

EP = ["Disease_Risk", "DR", "MH", "ODC", "TSLN", "DN", "ARMD", "MYA"]
PAIRS = {"E1_minus_E0": ("E1", "E0"), "E3_minus_E2": ("E3", "E2"),
         "E4_minus_E3": ("E4", "E3"), "E5_minus_E3": ("E5", "E3")}


def main() -> None:
    per_head = json.loads((ROOT / "per_head.json").read_text(encoding="utf-8"))
    out = {}

    for name, (x, y) in PAIRS.items():
        shared = [e for e in EP if e in per_head[x] and e in per_head[y]]
        if len(shared) < 3:
            continue
        full_x = sum(per_head[x][e]["auroc"] for e in shared) / len(shared)
        full_y = sum(per_head[y][e]["auroc"] for e in shared) / len(shared)
        rows = {}
        for drop in shared:
            keep = [e for e in shared if e != drop]
            mx = sum(per_head[x][e]["auroc"] for e in keep) / len(keep)
            my = sum(per_head[y][e]["auroc"] for e in keep) / len(keep)
            rows[drop] = {"delta": mx - my, "n_endpoints": len(keep)}
        deltas = [v["delta"] for v in rows.values()]
        out[name] = {
            "endpoints": shared,
            "full_delta": full_x - full_y,
            "leave_one_out": rows,
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "sign_stable": all(d > 0 for d in deltas) or all(d < 0 for d in deltas),
        }
        o = out[name]
        print(f"{name}: full {o['full_delta']:+.4f}  range [{o['min_delta']:+.4f}, "
              f"{o['max_delta']:+.4f}]  sign stable {o['sign_stable']}")

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
