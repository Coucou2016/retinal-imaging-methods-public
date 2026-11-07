#!/usr/bin/env python3
"""Rebuild only the per-endpoint decision curves from cached predictions."""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ANALYSIS = Path(os.environ.get("ARM_ANALYSIS_DIR", ROOT))
OUT = Path(os.environ.get("ARM_STATS_DIR", ROOT))

from calibration import per_class_decision_curves  # noqa: E402


def main() -> None:
    rows = {}
    for name in ("E0", "E1", "E2", "E3", "E4", "E5"):
        f = ANALYSIS / f"arm_{name}.npz"
        if not f.is_file():
            continue
        d = np.load(f, allow_pickle=True)
        names = [str(x) for x in d["names"]]
        per = {}
        for k, ep in enumerate(names):
            y, p = d["labels"][:, k], d["probs"][:, k]
            if len(np.unique(y)) < 2:
                continue
            per[ep] = per_class_decision_curves(y, p, class_names=[ep])[ep]
        rows[name] = per
        print(f"{name}: {len(per)} endpoints")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision_curves.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT / "decision_curves.json")


if __name__ == "__main__":
    main()
