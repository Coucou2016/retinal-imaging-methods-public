"""UKB-compressed tensor layout checks."""

from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import os

REQUIRED_NPZ = [
    "UKB_RETF.npz",
    "UKB_swin.npz",
    "UKB_vim.npz",
    "UKB_mqd.npz",
    "UKB_y0.npz",
    "UKB_y5.npz",
    "UKB_y10.npz",
]


def ukb_compressed_ready(data_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(data_dir, name)) for name in REQUIRED_NPZ)
