#!/usr/bin/env python3
"""Read the learned quality-router coefficients out of the checkpoints.

The manuscript originally said that E4's free-linear router could not be put in
the same form as the monotone weights. That was wrong. Both parameterisations
are stored in the checkpoint, and once the column order is respected they are
directly comparable.

Column order matters and is easy to get wrong. ``quality_scalar`` feeds the
quality vector to the router as ``(good, usable, bad)``, so ``q_fc.weight`` is in
that order. ``monotone_weights`` internally builds ``(bad, usable, good)`` and
then reverses it. This script prints both in the shared ``(good, usable, bad)``
order used by the model, and also normalises each map so that the good-quality
weight is 1, which is the scale the fixed baseline uses.

Writes ``results/real_rfmid/analysis_sel/stats/router_coefficients.json``.

Usage
-----
    python scripts/router_coefficients.py
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

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT
OUT = ROOT / "router_coefficients.json"
ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
BASELINE = [1.0, 0.5, 0.0]  # good, usable, bad


def monotone_from_delta(delta: torch.Tensor) -> list[float]:
    inc = F.softplus(delta) + 1e-6
    cum = torch.cumsum(inc, dim=0)
    w = cum / cum[-1].clamp_min(1e-6)  # bad, usable, good
    return [float(w[2]), float(w[1]), float(w[0])]  # -> good, usable, bad


def main() -> None:
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))
    res = {}

    for arm in ARMS:
        if arm not in manifest:
            continue
        entry = manifest[arm]
        ckpt = Path(entry["ckpt"])
        if not ckpt.is_file():
            alt = Path(entry["run_dir"]) / "ckpt" / ckpt.name
            ckpt = alt if alt.is_file() else ckpt
        if not ckpt.is_file():
            print(f"[skip] {arm}: checkpoint not found")
            continue

        sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        row = {"router": entry.get("quality_router"), "ckpt": str(ckpt), "backbones": []}

        for idx in range(3):
            prefix = f"models.{idx}.0.quality_aware."
            if prefix + "q_fc.weight" in sd:
                w = [float(x) for x in sd[prefix + "q_fc.weight"].flatten()]
                b = float(sd[prefix + "q_fc.bias"].flatten()[0])
                row["backbones"].append({"backbone": idx, "weights_good_usable_bad": w, "bias": b})
            elif prefix + "mono_delta" in sd:
                w = monotone_from_delta(sd[prefix + "mono_delta"])
                row["backbones"].append({"backbone": idx, "weights_good_usable_bad": w, "bias": 0.0})

        if not row["backbones"]:
            res[arm] = row
            continue

        ws = [b["weights_good_usable_bad"] for b in row["backbones"]]
        mean = [sum(w[i] for w in ws) / len(ws) for i in range(3)]
        row["mean_weights_good_usable_bad"] = mean
        good = mean[0]
        row["normalised_to_good"] = [x / good for x in mean] if abs(good) > 1e-9 else None
        row["ordering_ok"] = bool(mean[0] >= mean[1] >= mean[2])
        row["deviation_from_baseline"] = [mean[i] - BASELINE[i] for i in range(3)]
        row["max_abs_deviation"] = max(abs(x) for x in row["deviation_from_baseline"])

        res[arm] = row

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}\n")

    print(f"{'arm':<4} {'router':<12} {'good':>9} {'usable':>9} {'bad':>9}  {'ordered':>8}  {'max|dev|':>9}")
    print("-" * 74)
    for arm, r in res.items():
        if "mean_weights_good_usable_bad" not in r:
            print(f"{arm:<4} {r['router']:<12}  (no router head)")
            continue
        m = r["mean_weights_good_usable_bad"]
        print(
            f"{arm:<4} {str(r['router']):<12} {m[0]:9.4f} {m[1]:9.4f} {m[2]:9.4f}"
            f"  {str(r['ordering_ok']):>8}  {r['max_abs_deviation']:9.4f}"
        )

    print("\nper-backbone spread (q_fc / mono_delta solved independently per backbone)")
    for arm, r in res.items():
        for b in r.get("backbones", []):
            w = b["weights_good_usable_bad"]
            print(f"  {arm} backbone {b['backbone']}: good {w[0]:.4f} usable {w[1]:.4f} bad {w[2]:.6f}")


if __name__ == "__main__":
    main()
