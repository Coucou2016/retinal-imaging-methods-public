#!/usr/bin/env python3
"""Pick the reporting epoch per arm from the per-epoch validation predictions.

Background
----------
Training writes ``py_<epoch>_val.npz`` next to each checkpoint. The earlier
pipeline chose the checkpoint by lexicographic filename order, so ``9.pt``
sorted after ``15.pt`` and every arm silently reported epoch 9. This script
ranks epochs by validation macro AUROC (single-task arms: AUROC of the single
head) and prints the selection table, so the reported epoch is a documented,
data-driven choice rather than an artefact of string sorting.

Usage
-----
    python scripts/select_epochs.py --runs E0=path E1=path ...
"""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    vals = []
    for k in range(y.shape[1]):
        col = y[:, k]
        if len(np.unique(col)) > 1:
            vals.append(roc_auc_score(col, p[:, k]))
    return float(np.mean(vals)) if vals else float("nan")


def scan(run_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(run_dir.rglob("py_*_val.npz")):
        m = re.match(r"py_(\d+)_val\.npz$", f.name)
        if not m:
            continue
        epoch = int(m.group(1))
        d = np.load(f)
        p, y = d["p"], d["y"]
        if p.ndim == 1:
            p = p.reshape(-1, 1)
            y = y.reshape(-1, 1)
        rows.append({"epoch": epoch, "val_macro_auroc": macro_auroc(y, p), "n": int(p.shape[0]), "k": int(p.shape[1])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=RUN_DIR")
    ap.add_argument("--out", default=str(ROOT / "epoch_selection.json"))
    args = ap.parse_args()

    table = {}
    for spec in args.runs:
        name, _, run = spec.partition("=")
        run_dir = Path(run)
        rows = scan(run_dir)
        if not rows:
            print(f"{name}: no per-epoch validation predictions under {run_dir}")
            continue
        best = max(rows, key=lambda r: r["val_macro_auroc"])
        table[name] = {
            "run_dir": str(run_dir),
            "n_epochs_scored": len(rows),
            "selected_epoch": best["epoch"],
            "selected_val_macro_auroc": best["val_macro_auroc"],
            "per_epoch": rows,
        }
        print(
            f"{name}: {len(rows)} epochs, best epoch {best['epoch']} "
            f"(val macro AUROC {best['val_macro_auroc']:.4f}); "
           f"last epoch {rows[-1]['epoch']} scored {rows[-1]['val_macro_auroc']:.4f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
