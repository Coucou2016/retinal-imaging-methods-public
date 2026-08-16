#!/usr/bin/env python
"""Generate synthetic UKB-compressed tensors for smoke tests and local training demos.

The official Reti-Pioneer release trains on pre-extracted foundation features
(RETFound, Swin-B, Vision Mamba) plus quality scores and clinical metadata.
This script creates minimal compatible .npz files when UK Biobank access is unavailable.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from reti_pioneer.constants import DISEASE_NAMES
from reti_pioneer.data_paths import ukb_compressed_ready


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate(
    out_dir: str,
    n_samples: int = 256,
    seed: int = 42,
    positive_rate: float = 0.25,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = _rng(seed)
    n = n_samples
    d = len(DISEASE_NAMES)

    # Feature dims: RETFound/Swin 1024, Vim-S 384 (paper architecture)
    for name, dim in [("UKB_RETF", 1024), ("UKB_swin", 1024), ("UKB_vim", 384)]:
        left = rng.standard_normal((n, dim)).astype(np.float32)
        right = rng.standard_normal((n, dim)).astype(np.float32)
        np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), left=left, right=right)

    # Clinical + quality (good, usable, bad probabilities)
    m = np.column_stack(
        [
            rng.normal(56, 8, n),  # age
            rng.integers(0, 2, n),  # gender
            rng.normal(75, 15, n),  # weight
            rng.integers(0, 7, n),  # ethnicity index 0-6 (one-hot dim 7)
        ]
    ).astype(np.float32)
    mn = np.array(["baselineage", "gender", "weight", "ethnicity"])
    ql = rng.dirichlet([8, 2, 1], size=n).astype(np.float32)
    qr = rng.dirichlet([8, 2, 1], size=n).astype(np.float32)
    center = rng.integers(0, 6, size=n)
    pid = np.arange(n, dtype=np.int64)
    np.savez_compressed(
        os.path.join(out_dir, "UKB_mqd.npz"),
        m=m,
        mn=mn,
        ql=ql,
        qr=qr,
        center=center,
        pid=pid,
    )

    yn = np.array(DISEASE_NAMES)
    for horizon in [0, 5, 10]:
        y = (rng.random((n, d)) < positive_rate).astype(np.float32)
        np.savez_compressed(os.path.join(out_dir, f"UKB_y{horizon}.npz"), y=y, yn=yn)

    print(f"Wrote demo dataset ({n} samples) to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic Reti-Pioneer training tensors")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(ROOT, "data", "UKBCompressed"),
        help="Output directory for UKB_*.npz files",
    )
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-rate", type=float, default=0.25)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing UKB_*.npz files",
    )
    args = parser.parse_args()
    if not args.force and ukb_compressed_ready(args.out_dir):
        print(f"Dataset already present in {args.out_dir} (use --force to regenerate)")
        return
    generate(args.out_dir, args.n_samples, args.seed, args.positive_rate)


if __name__ == "__main__":
    main()
