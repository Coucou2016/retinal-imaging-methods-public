#!/usr/bin/env python3
"""Collect per-sample test predictions for every RFMiD ablation arm.

Why this exists
---------------
``scripts/evaluate.py`` recomputes metrics from a checkpoint each time it runs,
which costs roughly two minutes per arm on this host. The manuscript needs
paired comparisons, quality-stratified metrics, per-head intervals and router
weights for the same fold, so we cache the raw probabilities once and compute
every statistic downstream from the cache.

Outputs
-------
``results/real_rfmid/analysis/arm_<NAME>.npz`` with
    probs   (n, K)  sigmoid probabilities on the test split
    logits  (n, K)  raw logits (needed to refit temperature)
    labels  (n, K)  binary labels, -1 where the endpoint was not observed
    ql      (n, 3)  left-eye soft quality (good, usable, bad)
    qr      (n, 3)  right-eye soft quality
    qfuse   (n, 3)  the quality vector actually consumed by the fusion block
    names   (K,)    endpoint names
    split_ids (n,)  row indices into the source cache
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
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UKBDataset import UKBDatasetFast  # noqa: E402
from RetiPioneer import get_reti_pioneer, normalize_ensemble  # noqa: E402
from evaluate import (  # noqa: E402
    collect_probs,
    load_run_meta,
    resolve_ckpt_path,
    resolve_eval_split,
    resolve_run_dir,
)


def build_model(meta: dict):
    num_classes = int(meta.get("num_classes", len(meta.get("diseases") or []) or 1))
    model = get_reti_pioneer(
        True,
        num_classes=num_classes,
        learnable_q=bool(meta.get("learnable_q", False)),
        enable_q=bool(meta.get("enable_q", True)),
        ensemble=normalize_ensemble(meta.get("ensemble", "released_code")),
        quality_router=meta.get("quality_router", "fixed"),
        quality_gating=bool(meta.get("quality_gating", False)),
        quality_aux=False,
        lambda_q=float(meta.get("lambda_q", 0) or 0),
    )
    state = torch.load(
        meta.get("_ckpt") or resolve_ckpt_path(str(meta["_run_dir"])),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def collect_arm(run_dir: Path, data_dir: Path, split: str, device: torch.device, ckpt_override: str | None = None) -> dict:
    meta = load_run_meta(str(run_dir))
    meta["_run_dir"] = str(run_dir)
    if ckpt_override:
        meta["_ckpt"] = ckpt_override
    diseases = list(meta.get("diseases") or [])

    model = build_model(meta).to(device)

    base_ds = UKBDatasetFast(
        str(data_dir),
        meta=["baselineage", "gender", "weight", "ethnicity"],
        y=0,
        disease=[0],
        use_pretrain=["RETF", "SwinB", "VimS"],
        incident_exclude_prior=False,
    )
    if not diseases:
        diseases = list(base_ds.disease_names)
    base_ds.set_target(0, diseases, incident_exclude_prior=False)

    eval_ds, split_name, bundle = resolve_eval_split(str(run_dir), base_ds, split, 0.25, 42)
    probs, labels, logits = collect_probs(model, eval_ds, device, batch_size=128, return_logits=True)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
        labels = labels.reshape(-1, 1)
        logits = logits.reshape(-1, 1)

    ql, qr = [], []
    for i in range(len(eval_ds)):
        _lr, _m, (q_l, q_r), _y = eval_ds[i]
        ql.append(np.asarray(q_l, dtype=np.float32).reshape(-1))
        qr.append(np.asarray(q_r, dtype=np.float32).reshape(-1))

    return {
        "probs": probs.astype(np.float32),
        "logits": logits.astype(np.float32),
        "labels": labels.astype(np.float32),
        "ql": np.stack(ql),
        "qr": np.stack(qr),
        "names": np.asarray(diseases),
        "split_ids": np.asarray(bundle.get("test_idx") if split == "test" else bundle.get("val_idx"), dtype=np.int64),
        "split": split_name,
        "run_dir": str(run_dir),
        "quality_router": meta.get("quality_router"),
        "quality_gating": bool(meta.get("quality_gating", False)),
        "monotone_weights": _router_weights(model),
    }


def _router_weights(model) -> list[float] | None:
    """Return (bad, usable, good) router weights when a monotone router exists."""
    for module in model.modules():
        fn = getattr(module, "monotone_weights", None)
        if callable(fn) and getattr(module, "mono_delta", None) is not None:
            with torch.no_grad():
                return [float(v) for v in fn().reshape(-1)]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="NAME=RUN_DIR or NAME=RUN_DIR::CKPT")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "rfmid"))
    ap.add_argument("--out-dir", default=str(ROOT))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    for spec in args.arms:
        name, _, rest = spec.partition("=")
        run, _, ckpt = rest.partition("::")
        run_dir = Path(resolve_run_dir(str(ROOT / run if not Path(run).is_absolute() else run)))
        payload = collect_arm(run_dir, Path(args.data_dir), args.split, device, ckpt or None)
        payload["ckpt"] = np.asarray(ckpt or "")
        dst = out_dir / f"arm_{name}.npz"
        np.savez_compressed(dst, **payload)
        manifest[name] = {k: payload[k] for k in ("split", "run_dir", "quality_router", "quality_gating")}
        manifest[name]["monotone_weights"] = payload["monotone_weights"]
        manifest[name]["n"] = int(payload["probs"].shape[0])
        manifest[name]["ckpt"] = str(payload["ckpt"])
        print(
            f"{name}: n={payload['probs'].shape[0]} K={payload['probs'].shape[1]} "
            f"router={payload['quality_router']} ckpt={ckpt or 'auto'} weights={payload['monotone_weights']} -> {dst.name}",
            flush=True,
        )

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
