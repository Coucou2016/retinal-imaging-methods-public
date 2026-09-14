"""Full inference path for methods-extension checkpoints (not upstream inference.py)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset.UKBDataset import UKBDatasetFast
from model.RetiPioneer import get_reti_pioneer, normalize_ensemble
from reti_pioneer.config import load_config
from reti_pioneer.split import load_calibration_idx, load_split, load_test_idx
from utils.calibration import (
    apply_operating_point_thresholds,
    fit_operating_point_thresholds,
    fit_temperature,
)


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_ckpt_dir(ckpt: str | Path) -> Path:
    p = Path(ckpt)
    if p.is_file():
        return p.parent
    if p.is_dir():
        return p
    raise FileNotFoundError(f"Checkpoint path not found: {ckpt}")


def _find_weights(ckpt_dir: Path) -> Path:
    for name in ("best.pt", "model.pt", "checkpoint.pt", "last.pt"):
        cand = ckpt_dir / name
        if cand.is_file():
            return cand
    pts = sorted(ckpt_dir.glob("*.pt"))
    if pts:
        return pts[0]
    raise FileNotFoundError(f"No .pt weights under {ckpt_dir}")


def _load_state(model: torch.nn.Module, weights: Path, device: torch.device) -> None:
    blob = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(blob, dict):
        if "model" in blob:
            state = blob["model"]
        elif "state_dict" in blob:
            state = blob["state_dict"]
        else:
            state = blob
    else:
        state = blob
    model.load_state_dict(state, strict=False)


@torch.no_grad()
def _score_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        x, y = batch[0], batch[-1]
        # prepare like train prepare_batch
        (l, r), m, qs = x if isinstance(x, tuple) and len(x) == 3 else batch[:3]
        if isinstance(l, list):
            l = [t.to(device).float() for t in l]
            r = [t.to(device).float() for t in r]
        else:
            l = l.to(device).float()
            r = r.to(device).float()
        m = m.to(device).float()
        ql, qr = qs
        ql = ql.to(device).float()
        qr = qr.to(device).float()
        out = model(((l, r), m, (ql, qr)))
        logits_all.append(out.detach().cpu().numpy())
        y_all.append(np.asarray(y, dtype=np.float32))
    return np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Predict with methods-extension checkpoints: load config + run_meta, "
            "optional temperature T from calibration fold, frozen operating-point "
            "thresholds, endpoint names in JSON. "
            "Upstream inference.py is reference-only (UKB TorchScript)."
        )
    )
    parser.add_argument("--ckpt", required=True, help="Checkpoint dir or .pt file")
    parser.add_argument("--config", default=None, help="YAML used for training (optional)")
    parser.add_argument("--data-dir", default=None, help="Feature cache override")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which frozen split to score (default test)",
    )
    parser.add_argument("--calibrate", action="store_true", help="Fit T on calibration_idx")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", required=True, help="Where to write predictions/metrics")
    parser.add_argument(
        "--thresholds-json",
        default=None,
        help="Optional frozen thresholds JSON (youden / sens95spec); else fit on val",
    )
    args = parser.parse_args()

    ckpt_dir = _resolve_ckpt_dir(args.ckpt)
    meta_path = ckpt_dir / "run_meta.json"
    meta = _read_json(meta_path) if meta_path.is_file() else {}
    cfg_path = args.config
    if cfg_path is None:
        # Prefer sibling config pointer if present.
        for cand in (ckpt_dir / "config.yaml", Path(ROOT) / "configs" / "default.yaml"):
            if cand.is_file():
                cfg_path = str(cand)
                break
    cfg = load_config(cfg_path) if cfg_path else load_config()
    data_dir = args.data_dir or meta.get("data_dir") or cfg["data_dir"]
    diseases = list(meta.get("diseases") or cfg.get("diseases") or ["t2dm"])
    num_classes = int(meta.get("num_classes") or len(diseases))
    learnable_q = bool(meta.get("learnable_q", cfg["training"].get("learnable_q", False)))
    quality_router = meta.get(
        "quality_router",
        cfg["training"].get("quality_router", "monotone" if learnable_q else "fixed"),
    )
    ensemble = normalize_ensemble(meta.get("ensemble", cfg["training"].get("ensemble", "released_code")))
    enable_q = bool(meta.get("enable_q", cfg["training"].get("enable_q", True)))
    quality_gating = bool(meta.get("quality_gating", cfg["training"].get("quality_gating", False)))
    lambda_q = float(meta.get("lambda_q", cfg["training"].get("lambda_q", 0.0)) or 0.0)
    horizon = int(meta.get("horizon", 0))
    fast = bool(meta.get("fast_mode", cfg["training"].get("fast_mode", True)))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = get_reti_pioneer(
        fast,
        num_classes=num_classes,
        learnable_q=learnable_q,
        enable_q=enable_q,
        ensemble=ensemble,
        quality_router=quality_router,
        quality_gating=quality_gating,
        quality_aux=lambda_q > 0,
        lambda_q=lambda_q,
    ).to(device)
    weights = _find_weights(ckpt_dir)
    _load_state(model, weights, device)
    print(f"Loaded {weights} on {device}")

    ds = UKBDatasetFast(
        data_dir,
        None,
        cfg["clinic_variables"],
        disease=[0],
        use_pretrain=cfg["pretrain_features"],
        incident_exclude_prior=False,
    )
    ds.set_target(horizon, diseases, incident_exclude_prior=False)

    split_path = ckpt_dir / "split.npz"
    if not split_path.is_file():
        split_path = Path(data_dir) / "split.npz"
    ids: list[int] | None = None
    if split_path.is_file() and args.split != "all":
        train_idx, val_idx = load_split(str(split_path))
        test_idx = load_test_idx(str(split_path))
        cal_idx = load_calibration_idx(str(split_path))
        mapping = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx if test_idx is not None else val_idx,
        }
        ids = mapping[args.split]
        if ids is None:
            raise ValueError(f"Split {args.split!r} missing in {split_path}")
        score_ds: Subset | UKBDatasetFast = Subset(ds, ids)
    else:
        score_ds = ds
        train_idx = val_idx = test_idx = cal_idx = None

    loader = DataLoader(score_ds, batch_size=min(args.batch_size, len(score_ds)), shuffle=False)
    logits, labels = _score_loader(model, loader, device)

    temperature = 1.0
    if args.calibrate and split_path.is_file():
        cal_ids = load_calibration_idx(str(split_path))
        if not cal_ids and val_idx:
            cal_ids = val_idx
        if cal_ids:
            cal_loader = DataLoader(
                Subset(ds, cal_ids),
                batch_size=min(args.batch_size, len(cal_ids)),
                shuffle=False,
            )
            cal_logits, cal_y = _score_loader(model, cal_loader, device)
            temperature = float(fit_temperature(cal_logits, cal_y))
            print(f"Fitted temperature T={temperature:.4f} on calibration fold")

    probs = 1.0 / (1.0 + np.exp(-np.clip(logits / max(temperature, 1e-6), -60, 60)))

    thr_payload: dict | None = None
    if args.thresholds_json and Path(args.thresholds_json).is_file():
        thr_payload = _read_json(Path(args.thresholds_json))
    elif split_path.is_file() and val_idx:
        val_loader = DataLoader(
            Subset(ds, val_idx),
            batch_size=min(args.batch_size, len(val_idx)),
            shuffle=False,
        )
        v_logits, v_y = _score_loader(model, val_loader, device)
        v_probs = 1.0 / (1.0 + np.exp(-np.clip(v_logits / max(temperature, 1e-6), -60, 60)))
        thr_payload = fit_operating_point_thresholds(v_y, v_probs)
        thr_payload["source"] = "val_frozen"

    applied = None
    if thr_payload is not None:
        applied = apply_operating_point_thresholds(labels, probs, thr_payload)

    out = {
        "ckpt": str(weights),
        "data_dir": str(data_dir),
        "split": args.split,
        "diseases": diseases,
        "endpoints": diseases,
        "num_classes": num_classes,
        "temperature": temperature,
        "n": int(labels.shape[0]),
        "probs": probs.tolist(),
        "labels": labels.tolist(),
        "thresholds": thr_payload,
        "operating_points_applied": applied,
        "quality_gating": quality_gating,
        "ensemble": ensemble,
        "quality_router": quality_router,
        "note": (
            "Extension inference path. Upstream inference.py is UKB TorchScript "
            "reference only — do not use it for methods-extension checkpoints."
        ),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
