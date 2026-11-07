"""Recompute the prose claims that tables do not carry, and print checks."""

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
STATS = ROOT
NUM = ROOT / "manuscript_numbers.json"

num = json.loads(NUM.read_text(encoding="utf-8"))
offs = json.loads((ROOT / "logit_offset.json").read_text(encoding="utf-8"))["per_arm"]
hold = json.loads((ROOT / "recalibration_holdout.json").read_text(encoding="utf-8"))["per_arm"]

print("== ECE after temperature: rise per arm ==")
rises = {}
for arm, row in num["arms"].items():
    before = row["mean_head_ece"]
    after = row["mean_head_ece_after_T"]
    rises[arm] = after - before
    print(f"  {arm}: {before:.4f} -> {after:.4f}  rise {after - before:+.4f}")
print("  rise range: %.4f to %.4f" % (min(rises.values()), max(rises.values())))
print("  all positive:", all(v > 0 for v in rises.values()))

print()
print("== Brier change after temperature ==")
bs = {}
for arm, row in num["arms"].items():
    bs[arm] = row["brier_after_T"] - row["pooled_brier"]
    print(f"  {arm}: {row['pooled_brier']:.4f} -> {row['brier_after_T']:.4f}  {bs[arm]:+.4f}")
print("  max abs change: %.4f" % max(abs(v) for v in bs.values()))

print()
print("== logit centre vs logit prevalence: r and slope ==")
for arm in ("E2", "E3", "E4", "E5"):
    rows = offs[arm]["per_head"]
    x = np.array([v["logit_prevalence"] for v in rows.values()])
    y = np.array([v["mean_logit"] for v in rows.values()])
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    gaps = np.abs(y - x)
    print(
        f"  {arm}: n={len(x)} r={r:.3f} slope={slope:.3f} intercept={intercept:+.3f} "
        f"mean|gap|={gaps.mean():.3f} max|gap|={gaps.max():.3f} meanECE={num['arms'][arm]['mean_head_ece']:.4f}"
    )

print()
print("== isotonic AUROC change (holdout) ==")
for arm, row in hold.items():
    s = row["summary"]
    print(
        f"  {arm}: none {s['macro_auroc_none']:.4f} isotonic {s['macro_auroc_isotonic']:.4f} "
        f"delta {s['macro_auroc_isotonic'] - s['macro_auroc_none']:+.4f} "
        f"platt {s['macro_auroc_platt']:.4f} temp {s['macro_auroc_temperature']:.4f}"
    )

print()
print("== prevalence-offset ECE vs raw, multitask arms ==")
for arm in ("E2", "E3", "E4", "E5"):
    r = num["arms"][arm]
    print(
        f"  {arm}: raw {r['mean_head_ece']:.4f} -> prev {r['holdout_ece_none']:.4f} "
        f"platt {r['holdout_ece_platt']:.4f} iso {r['holdout_ece_isotonic']:.4f}"
    )

print()
print("== router induced shifts ==")
re_ = json.loads((ROOT / "router_effect.json").read_text(encoding="utf-8"))
for arm, r in re_.items():
    print(
        f"  {arm}: mean {r['induced_shift_mean']:.5f} max {r['induced_shift_max_abs']:.5f} "
        f"ratio {r['ratio_shift_to_spread']:.5f}"
    )

print()
print("== threshold transfer loss list ==")
for arm, r in num["threshold_transfer"].items():
    dr = r["DR"]
    print(
        f"  {arm}: val sens {dr['val_sensitivity_at_95spec']:.3f} frozen {dr['test_sensitivity_frozen']:.3f} "
        f"oracle {dr['test_sensitivity_oracle']:.3f} gap {dr['test_sensitivity_oracle'] - dr['test_sensitivity_frozen']:+.3f}"
    )
