#!/usr/bin/env python3
"""Upper bound on what temperature scaling could achieve, per endpoint.

For every arm and every endpoint, fit a temperature on the *test* labels and
rescale. This is not a legitimate calibration procedure, because it uses the
labels it is scored against. It answers one question only: if the scale were
chosen perfectly, how much of the calibration error would disappear?

The answer decides how the manuscript should describe the shared-head
calibration cost. If the oracle barely moves the number, the error is in the
shape of the score distribution, and no scalar rescaling will fix it.

Output: ``stats/calibration_oracle.json``

Usage:
    python scripts/calibration_oracle.py
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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def main() -> None:
    out = {}
    for arm in ["E0", "E1", "E2", "E3", "E4", "E5"]:
        d = np.load(ANALYSIS / f"arm_{arm}.npz", allow_pickle=True)
        names = [str(x) for x in d["names"]]
        y, z = d["labels"], d["logits"]

        raw_ece, or_ece, raw_br, or_br, temps = [], [], [], [], {}
        for k, ep in enumerate(names):
            m = y[:, k] >= 0
            if m.sum() == 0 or len(np.unique(y[m, k])) < 2:
                continue
            yk, zk = y[m, k], z[m, k]
            T = float(fit_temperature(zk, yk))
            temps[ep] = round(T, 4)
            raw_ece.append(expected_calibration_error(yk, sigmoid(zk), n_bins=BINS))
            or_ece.append(expected_calibration_error(yk, apply_temperature(zk, T), n_bins=BINS))
            raw_br.append(brier_score(yk, sigmoid(zk)))
            or_br.append(brier_score(yk, apply_temperature(zk, T)))

        out[arm] = {
            "temperature_per_endpoint_oracle": temps,
            "mean_head_ece_raw": round(float(np.mean(raw_ece)), 4),
            "mean_head_ece_oracle": round(float(np.mean(or_ece)), 4),
            "mean_head_ece_gain": round(float(np.mean(raw_ece) - np.mean(or_ece)), 4),
            "mean_head_brier_raw": round(float(np.mean(raw_br)), 4),
            "mean_head_brier_oracle": round(float(np.mean(or_br)), 4),
            "note": "temperature fitted on test labels; an upper bound, not a procedure",
        }
        r = out[arm]
        print(
            f"{arm}: mean-head ECE raw {r['mean_head_ece_raw']:.4f} -> oracle {r['mean_head_ece_oracle']:.4f} "
            f"(gain {r['mean_head_ece_gain']:+.4f}) | Brier {r['mean_head_brier_raw']:.4f} -> {r['mean_head_brier_oracle']:.4f}",
            flush=True,
        )

    (OUT / "calibration_oracle.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT / "calibration_oracle.json")


if __name__ == "__main__":
    main()
