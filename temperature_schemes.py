#!/usr/bin/env python3
"""Per-endpoint versus single-temperature calibration for the shared-head arms.

The shared-head arms carry a large calibration cost. The question is whether that
cost comes from the scale of the scores (which one temperature can fix) or from
the endpoints disagreeing with each other (which one temperature cannot fix).

Three schemes are compared, all fitted without touching test:

``none``      raw sigmoid output
``single``    one temperature fitted on the 640-image validation fold
``per_head``  one temperature per endpoint, fitted on the same 640 images

Reported on test: pooled ECE, mean per-endpoint ECE, worst-endpoint ECE, and
Brier. AUROC is unchanged by any monotone rescaling and is not reported again.

Output: ``stats/temperature_schemes.json``

Usage:
    python scripts/temperature_schemes.py
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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration import (  # noqa: E402
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_temperature,
)

ANALYSIS = ROOT
OUT = ROOT
BINS = 10
EPS = 1e-6


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ece10(a, b):
    return expected_calibration_error(a, b, n_bins=BINS)


def load_val(manifest: dict, arm: str):
    e = manifest[arm]
    run_dir = Path(e["run_dir"])
    epoch = Path(e["ckpt"]).stem
    d = np.load(run_dir / "ckpt" / f"py_{epoch}_val.npz", allow_pickle=True)
    p, y = d["p"], d["y"]
    if p.ndim == 1:
        p, y = p.reshape(-1, 1), y.reshape(-1, 1)
    return p, y


def summary(y: np.ndarray, p: np.ndarray, names: list[str]) -> dict:
    per_head, keep = {}, []
    for k, ep in enumerate(names):
        m = y[:, k] >= 0
        if m.sum() == 0 or len(np.unique(y[m, k])) < 2:
            continue
        v = ece10(y[m, k], p[m, k])
        per_head[ep] = round(float(v), 4)
        keep.append(ep)
    m = y >= 0
    return {
        "pooled_ece": round(float(ece10(y[m].ravel(), p[m].ravel())), 4),
        "mean_head_ece": round(float(np.mean(list(per_head.values()))), 4),
        "worst_head_ece": round(float(np.max(list(per_head.values()))), 4),
        "pooled_brier": round(float(brier_score(y[m].ravel(), p[m].ravel())), 4),
        "per_head_ece": per_head,
    }


def main() -> None:
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))
    out = {}

    for arm in manifest:
        cal = np.load(ANALYSIS / f"cal_{arm}.npz", allow_pickle=True)
        vp, vy = load_val(manifest, arm)
        te = np.load(ANALYSIS / f"arm_{arm}.npz", allow_pickle=True)
        names = [str(x) for x in te["names"]]
        K = len(names)

        # validation fold in logit space: 64 calibration + 576 validation
        val_p = np.concatenate([cal["probs"], vp], axis=0)
        val_y = np.concatenate([cal["labels"], vy], axis=0)
        val_z = np.concatenate([cal["logits"], to_logit(vp)], axis=0)

        t_y, t_z, t_p = te["labels"], te["logits"], te["probs"]

        m_all = val_y >= 0
        T_single = float(fit_temperature(val_z[m_all], val_y[m_all]))

        T_head = {}
        for k, ep in enumerate(names):
            mk = val_y[:, k] >= 0
            if mk.sum() < 10 or len(np.unique(val_y[mk, k])) < 2:
                T_head[ep] = 1.0
                continue
            T_head[ep] = float(fit_temperature(val_z[mk, k], val_y[mk, k]))

        p_single = apply_temperature(t_z, T_single)
        p_head = np.stack(
            [apply_temperature(t_z[:, k], T_head[names[k]]) for k in range(K)], axis=1
        )

        out[arm] = {
            "temperature_single": round(T_single, 4),
            "temperature_per_head": {k: round(v, 4) for k, v in T_head.items()},
            "n_val_total": int(val_y.shape[0]),
            "none": summary(t_y, t_p, names),
            "single": summary(t_y, p_single, names),
            "per_head": summary(t_y, p_head, names),
        }

        r = out[arm]
        print(
            f"{arm} (T_single={T_single:.3f}, T range {min(T_head.values()):.2f}-{max(T_head.values()):.2f})\n"
            f"   mean-head ECE  none {r['none']['mean_head_ece']:.4f} | "
            f"single {r['single']['mean_head_ece']:.4f} | per-head {r['per_head']['mean_head_ece']:.4f}\n"
            f"   pooled ECE     none {r['none']['pooled_ece']:.4f} | "
            f"single {r['single']['pooled_ece']:.4f} | per-head {r['per_head']['pooled_ece']:.4f}\n"
            f"   worst-head ECE none {r['none']['worst_head_ece']:.4f} | "
            f"single {r['single']['worst_head_ece']:.4f} | per-head {r['per_head']['worst_head_ece']:.4f}\n"
            f"   Brier          none {r['none']['pooled_brier']:.4f} | "
            f"single {r['single']['pooled_brier']:.4f} | per-head {r['per_head']['pooled_brier']:.4f}",
            flush=True,
        )

    (OUT / "temperature_schemes.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT / "temperature_schemes.json")


if __name__ == "__main__":
    main()
