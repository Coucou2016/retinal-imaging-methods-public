#!/usr/bin/env python3
"""Calibration replication across seeds, fitted on validation and scored on test.

The main Results fit one scalar temperature on a 64-image carve of the
validation fold. A single run cannot tell whether "temperature makes it worse"
is a property of the shared head or of one initialisation. This script repeats
the post-processing ladder for every replicate seed with the same split of
duties every time: parameters come from the validation predictions of the
reported epoch, scores are read from the test fold, and nothing is fitted on
test labels.

Rules compared
--------------
1. identity, the raw score
2. one temperature shared by all endpoints, fitted on validation
3. one temperature per endpoint, fitted on validation
4. a zero-parameter prevalence shift, z + log(pi_val / (1 - pi_val))
5. a one-parameter shift per endpoint, fitted on validation by minimising BCE

Outputs
-------
results/real_rfmid/analysis_seed/calibration_replication.json
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

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
SEED_DIR = ROOT
MAIN_DIR = ROOT
CKPT_SEED = ROOT / "ckpts_seed"
OUT = SEED_DIR / "calibration_replication.json"

ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
PRIOR_EPOCHS = {"E0": 12, "E1": 14, "E2": 15, "E3": 15, "E4": 18, "E5": 15}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = y.size
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        total += mask.sum() / n * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def mean_ece(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([ece(y[:, k], p[:, k]) for k in range(y.shape[1])]))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(
        lambda lt: bce(y, sigmoid(z / np.exp(lt))), bounds=(-3.0, 3.0), method="bounded"
    )
    return float(np.exp(res.x))


def fit_shift(z: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(
        lambda b: bce(y, sigmoid(z + b)), bounds=(-8.0, 8.0), method="bounded"
    )
    return float(res.x)


def val_path(arm: str, seed: int, epoch: int) -> Path | None:
    """Validation per-sample predictions of the reported epoch."""
    if seed == 42:
        base = None
        for cand in (MAIN_DIR / f"arm_{arm}.npz",):
            if cand.is_file():
                d = np.load(cand, allow_pickle=True)
                if "ckpt" in d.files:
                    base = Path(str(d["ckpt"])).parent
    else:
        base = None
        for run in CKPT_SEED.glob(f"{arm}_s{seed}/*/*/y0"):
            cand = run / "ckpt" / f"{epoch}.pt"
            if cand.is_file():
                base = cand.parent
                break
    if base is None:
        return None
    p = base / f"py_{epoch}_val.npz"
    return p if p.is_file() else None


def main() -> None:
    manifest_path = SEED_DIR / "replicate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    summary: dict[str, dict] = {}
    for arm in ARMS:
        test_files: list[tuple[int, Path]] = []
        for p in SEED_DIR.glob(f"arm_{arm}_s*.npz"):
            test_files.append((int(p.stem.split("_s")[1]), p))
        primary = MAIN_DIR / f"arm_{arm}.npz"
        if primary.is_file():
            test_files.append((42, primary))

        per_seed = []
        for seed, tpath in sorted(test_files):
            epoch = (manifest.get(f"{arm}_s{seed}") or {}).get("epoch") or PRIOR_EPOCHS[arm]
            vpath = val_path(arm, seed, int(epoch))
            if vpath is None:
                print(f"[skip] {arm} s{seed}: no validation predictions at epoch {epoch}")
                continue
            t = np.load(tpath, allow_pickle=True)
            v = np.load(vpath, allow_pickle=True)
            y_te, z_te = t["labels"].astype(np.float64), t["logits"].astype(np.float64)
            y_va, z_va = v["y"].astype(np.float64), logit(v["p"].astype(np.float64))

            prev_va = np.array([y_va[:, k].mean() for k in range(y_va.shape[1])])
            pi_va = np.log(prev_va / (1 - prev_va))
            t_global = fit_temperature(z_va.ravel(), y_va.ravel())
            t_head = [fit_temperature(z_va[:, k], y_va[:, k]) for k in range(y_va.shape[1])]
            b_head = [fit_shift(z_va[:, k], y_va[:, k]) for k in range(y_va.shape[1])]

            entry = {
                "seed": seed,
                "epoch": int(epoch),
                "n_val": int(y_va.shape[0]),
                "ece_identity": mean_ece(y_te, sigmoid(z_te)),
                "ece_temperature_global": mean_ece(
                    y_te, np.stack([sigmoid(z_te[:, k] / t_global) for k in range(z_te.shape[1])], axis=1)
                ),
                "ece_temperature_per_head": mean_ece(
                    y_te, np.stack([sigmoid(z_te[:, k] / t_head[k]) for k in range(z_te.shape[1])], axis=1)
                ),
                "ece_prevalence_shift": mean_ece(
                    y_te, np.stack([sigmoid(z_te[:, k] + pi_va[k]) for k in range(z_te.shape[1])], axis=1)
                ),
                "ece_fitted_shift": mean_ece(
                    y_te, np.stack([sigmoid(z_te[:, k] + b_head[k]) for k in range(z_te.shape[1])], axis=1)
                ),
                "temperature_global": t_global,
                "temperature_per_head": t_head,
                "fitted_shift": b_head,
                "validation_prevalence": prev_va.tolist(),
            }
            per_seed.append(entry)

        if not per_seed:
            continue

        agg: dict = {"n_seeds": len(per_seed)}
        for key in ("ece_identity", "ece_temperature_global", "ece_temperature_per_head",
                    "ece_prevalence_shift", "ece_fitted_shift"):
            vals = np.array([e[key] for e in per_seed])
            agg[f"{key}_mean"] = float(vals.mean())
            agg[f"{key}_sd"] = float(vals.std(ddof=1)) if vals.size > 1 else None
            agg[f"{key}_min"] = float(vals.min())
            agg[f"{key}_max"] = float(vals.max())
        agg["temperature_worse_in_all_seeds"] = bool(
            all(e["ece_temperature_global"] > e["ece_identity"] for e in per_seed)
        )
        agg["per_head_temperature_worse_in_all_seeds"] = bool(
            all(e["ece_temperature_per_head"] > e["ece_identity"] for e in per_seed)
        )
        agg["prevalence_shift_better_in_all_seeds"] = bool(
            all(e["ece_prevalence_shift"] < e["ece_identity"] for e in per_seed)
        )
        summary[arm] = {"per_seed": per_seed, "aggregate": agg}

    OUT.write_text(
        json.dumps(
            {
                "per_arm": summary,
                "protocol": (
                    "post-processing fitted on validation predictions of the reported "
                    "epoch; test labels used only for scoring"
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"{'arm':4s} {'seeds':>5s} {'ident':>7s} {'temp':>7s} {'T/head':>7s} {'prev':>7s} {'fit':>7s}")
    for arm, rec in summary.items():
        a = rec["aggregate"]
        print(
            f"{arm:4s} {a['n_seeds']:>5d} {a['ece_identity_mean']:.3f}  "
            f"{a['ece_temperature_global_mean']:.3f}  {a['ece_temperature_per_head_mean']:.3f}  "
            f"{a['ece_prevalence_shift_mean']:.3f}  {a['ece_fitted_shift_mean']:.3f}"
        )
        print(
            f"     温度变差全部种子 {a['temperature_worse_in_all_seeds']}  "
            f"逐终点温度变差 {a['per_head_temperature_worse_in_all_seeds']}  "
            f"流行率平移改善全部种子 {a['prevalence_shift_better_in_all_seeds']}"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
