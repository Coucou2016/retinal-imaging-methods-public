"""Audit the remaining manuscript numbers against the statistics files."""

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
NUM = json.loads((ROOT / "manuscript_numbers.json").read_text(encoding="utf-8"))

print("== prevalence (test) ==")
for k, v in NUM["prevalence"].items():
    print(f"  {k}: {v}")

print("\n== quality counts ==")
print(" ", NUM["quality_counts"])

print("\n== epoch selection ==")
for k, v in NUM["epoch_selection"].items():
    print(f"  {k}: epoch {v['epoch']} of {v['n_epochs']} val {v['val_macro_auroc']}")

print("\n== E5 per head ==")
ph = json.loads((ROOT / "per_head.json").read_text(encoding="utf-8"))
print(" keys:", list(ph.keys())[:3])
e5 = ph["E5"]
for ep, row in e5.items():
    if isinstance(row, dict):
        print(f"  {ep}: { {k: (round(v,3) if isinstance(v,(int,float)) else v) for k,v in row.items() if k in ('auroc','ap','ece','brier','prevalence','n_pos','n')} }")

print("\n== strata ==")
st = json.loads((ROOT / "quality_strata.json").read_text(encoding="utf-8"))
print(json.dumps(st, indent=1, ensure_ascii=False)[:2200])

print("\n== fitted offset vs logit prevalence correlation ==")
co = json.loads((ROOT / "calibration_offset.json").read_text(encoding="utf-8"))["per_arm"]
for arm in ("E2", "E3", "E4", "E5"):
    rows = co[arm]["per_head"]
    x = np.array([v["logit_prevalence_val"] for v in rows.values()])
    y = np.array([v["offset_fitted"] for v in rows.values()])
    print(f"  {arm}: r={np.corrcoef(x,y)[0,1]:.3f} slope={np.polyfit(x,y,1)[0]:.3f}")

print("\n== E5 table 10 rows ==")
rows = co["E5"]["per_head"]
off = json.loads((ROOT / "logit_offset.json").read_text(encoding="utf-8"))["per_arm"]["E5"]["per_head"]
for ep in rows:
    r = rows[ep]
    o = off[ep]
    print(
        f"  {ep}: prev {r['prevalence_test']:.3f} logitprev {o['logit_prevalence']:+.2f} meanlogit {o['mean_logit']:+.2f} "
        f"gap {o['gap']:+.2f} fit {r['offset_fitted']:+.2f} ece {r['ece_none']:.3f} prev {r['ece_offset_prev']:.3f} platt {r['ece_platt']:.3f}"
    )

print("\n== decision curves @ DR ==")
dc = json.loads((ROOT / "decision_curves.json").read_text(encoding="utf-8"))
print(" keys:", list(dc.keys())[:6])
print(json.dumps(dc, indent=1, ensure_ascii=False)[:1200])
