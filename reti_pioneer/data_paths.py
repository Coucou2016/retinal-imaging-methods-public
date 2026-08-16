"""UKB-compressed tensor layout checks."""

from __future__ import annotations

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
