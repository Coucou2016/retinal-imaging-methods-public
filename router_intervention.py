#!/usr/bin/env python3
"""Inference-time intervention on the quality router.

Table 9 shows that every arm ends training with router weights within a few
thousandths of the fixed baseline. That makes the ablation hard to interpret: if
the learned router is numerically almost the identity, then a difference between
an arm with a router and an arm without one cannot be credited to routing.

This script settles the question by intervention rather than inference. It loads
each multitask checkpoint twice, once as trained and once with the router forced
back to the fixed baseline weights (good 1, usable 0.5, bad 0), and scores the
same test images both ways. Whatever performance remains after forcing the
router to the baseline did not come from the router.

Only the router parameters are touched. Encoder features are frozen, the heads
are unchanged, and the image order is identical, so the paired difference is
attributable to the router alone.

Writes ``results/real_rfmid/analysis_sel/stats/router_intervention.json``.

Usage
-----
    python scripts/router_intervention.py --arms E1 E3 E4 E5
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
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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

ANALYSIS = ROOT
OUT = ROOT / "router_intervention.json"
TARGET = np.array([0.0, 0.5, 1.0])  # bad, usable, good; good == 1 by construction


def inv_softplus(x: float) -> float:
    x = max(float(x), 1e-8)
    return float(np.log(np.expm1(x)))


def build_model(meta: dict, ckpt: str):
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
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def force_fixed_router(model) -> int:
    """Overwrite every quality head so the fusion scalar ignores quality ordering."""
    touched = 0
    for module in model.modules():
        if not hasattr(module, "quality_router"):
            continue
        if getattr(module, "mono_delta", None) is not None:
            with torch.no_grad():
                # weights (bad, usable, good) = (0, 0.5, 1)
                module.mono_delta.copy_(
                    torch.tensor(
                        [inv_softplus(1e-6), inv_softplus(0.5), inv_softplus(0.5)],
                        dtype=module.mono_delta.dtype,
                    )
                )
            touched += 1
        elif getattr(module, "q_fc", None) is not None:
            with torch.no_grad():
                module.q_fc.weight.copy_(torch.tensor([[1.0, 0.5, 0.0]]))
                module.q_fc.bias.copy_(torch.tensor([0.0]))
            touched += 1
    return touched


def macro_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    for k in range(labels.shape[1]):
        y = labels[:, k]
        if y.size and len(np.unique(y)) > 1:
            vals.append(roc_auc_score(y, probs[:, k]))
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["E1", "E3", "E4", "E5"])
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "rfmid"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))
    res = {}

    for arm in args.arms:
        if arm not in manifest:
            continue
        entry = manifest[arm]
        run_dir = Path(resolve_run_dir(str(entry["run_dir"])))
        meta = load_run_meta(str(run_dir))
        ckpt = entry["ckpt"]
        if not Path(ckpt).is_file():
            ckpt = str(Path(entry["run_dir"]) / "ckpt" / Path(ckpt).name)

        base = build_model(meta, ckpt).to(device)
        ds = UKBDatasetFast(
            args.data_dir,
            meta=["baselineage", "gender", "weight", "ethnicity"],
            y=0,
            disease=[0],
            use_pretrain=["RETF", "SwinB", "VimS"],
            incident_exclude_prior=False,
        )
        diseases = list(meta.get("diseases") or []) or list(ds.disease_names)
        ds.set_target(0, diseases, incident_exclude_prior=False)
        eval_ds, _, _ = resolve_eval_split(str(run_dir), ds, args.split, 0.25, 42)

        p_raw, y, _ = collect_probs(base, eval_ds, device, batch_size=128, return_logits=True)
        if p_raw.ndim == 1:
            p_raw = p_raw.reshape(-1, 1)
            y = y.reshape(-1, 1)

        forced = copy.deepcopy(base)
        n_touched = force_fixed_router(forced)
        p_fix, y2, _ = collect_probs(forced, eval_ds, device, batch_size=128, return_logits=True)
        if p_fix.ndim == 1:
            p_fix = p_fix.reshape(-1, 1)
        y2 = np.asarray(y2).reshape(y.shape)

        assert np.array_equal(y, y2), f"label order changed between passes: {y.shape} vs {y2.shape}"

        a_raw = macro_auroc(y, p_raw)
        a_fix = macro_auroc(y, p_fix)
        max_abs = float(np.abs(p_raw - p_fix).max())
        res[arm] = {
            "router": meta.get("quality_router"),
            "heads_overwritten": n_touched,
            "n": int(y.shape[0]),
            "macro_auroc_as_trained": a_raw,
            "macro_auroc_router_forced_to_fixed": a_fix,
            "delta": a_raw - a_fix,
            "max_abs_prob_change": max_abs,
            "identical_within_1e-6": bool(max_abs < 1e-6),
        }
        print(
            f"{arm}: as-trained {a_raw:.4f}  router forced to baseline {a_fix:.4f}  "
            f"delta {a_raw - a_fix:+.6f}  max|Δp| {max_abs:.2e}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
