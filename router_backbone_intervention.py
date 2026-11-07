#!/usr/bin/env python3
"""Inference-time interventions on both router layers of E5.

Two questions that the first version of the paper could not answer, because it
only intervened on the quality router.

1. Quality router. Forcing the learned weights back to (bad 0, usable 0.5,
   good 1) changes almost nothing, so the monotone gain comes from the training
   trajectory rather than from re-weighting at run time.

2. Backbone router. E5 adds a two-layer perceptron that maps the quality vector
   to a softmax over the three backbones. That layer also adds parameters, so a
   gain over E3 could come from quality conditioning or simply from an extra
   trainable block. Three interventions separate the two readings.

   uniform  replace the softmax output with (1/3, 1/3, 1/3), which keeps the
            architecture but removes any quality dependence. A large drop here
            means the quality signal is being used. Equality with ``as trained``
            means the module is inert.
   frozen   evaluate a copy whose backbone-router parameters were reset to their
            initial zeros, which is the same uniform average as ``uniform`` but
            applied to a freshly built module.
   shuffled quality  keep the learned router but permute the quality vector
            across images. If performance survives, the router cannot be
            reading quality at all.

Outputs ``results/real_rfmid/analysis_sel/stats/router_backbone_intervention.json``.

Usage
-----
    python scripts/router_backbone_intervention.py --arm E5
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))

from UKBDataset import UKBDatasetFast  # noqa: E402
from evaluate import (  # noqa: E402
    collect_probs,
    load_run_meta,
    resolve_eval_split,
    resolve_run_dir,
)
from router_intervention import build_model  # noqa: E402

ANALYSIS = ROOT
OUT = ROOT / "router_backbone_intervention.json"


def macro_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    for k in range(labels.shape[1]):
        y = labels[:, k]
        if y.size and len(np.unique(y)) > 1:
            vals.append(roc_auc_score(y, probs[:, k]))
    return float(np.mean(vals))


class UniformBackbone(torch.nn.Module):
    """Wrap a trained model so its backbone softmax is replaced by a uniform mix."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, batch):
        router = getattr(self.inner, "backbone_router", None)
        if router is None:
            return self.inner(batch)
        original = router.forward

        def uniform_forward(q, _n=router.n_backbones):
            if q.dim() == 1:
                q = q.unsqueeze(0)
            return torch.full(
                (q.shape[0], _n), 1.0 / float(_n), device=q.device, dtype=q.dtype
            )

        router.forward = uniform_forward  # type: ignore[assignment]
        try:
            return self.inner(batch)
        finally:
            router.forward = original  # type: ignore[assignment]


def backbone_weights(model, dataset, device, batch_size: int = 128) -> np.ndarray:
    """Collect the (n, 3) softmax weights the router assigns on the eval split."""
    model.eval()
    chunks: list[np.ndarray] = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            from evaluate import prepare_batch as eval_prepare_batch

            batch_in, _y = eval_prepare_batch(batch, device, False)
            _ = model(batch_in)
            w = getattr(model, "_last_backbone_weights", None)
            if w is None:
                break
            chunks.append(w.detach().cpu().numpy())
    if not chunks:
        return np.zeros((0, 3))
    return np.concatenate(chunks, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="E5")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "rfmid"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--shuffles", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest[args.arm]
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
        p_raw, y = p_raw.reshape(-1, 1), y.reshape(-1, 1)
    a_raw = macro_auroc(y, p_raw)

    # 1. uniform backbone mixture, architecture otherwise untouched
    p_uni, y_uni, _ = collect_probs(
        UniformBackbone(base).to(device), eval_ds, device, batch_size=128, return_logits=True
    )
    if p_uni.ndim == 1:
        p_uni, y_uni = p_uni.reshape(-1, 1), y_uni.reshape(-1, 1)
    assert np.array_equal(np.asarray(y), np.asarray(y_uni))
    a_uni = macro_auroc(y, p_uni)

    # 2. zeroed router parameters == initial state
    frozen = copy.deepcopy(base)
    for name, module in frozen.named_modules():
        if name == "backbone_router" and getattr(module, "mlp", None) is not None:
            with torch.no_grad():
                module.mlp[-1].weight.zero_()
                module.mlp[-1].bias.zero_()
    p_frz, _y, _ = collect_probs(frozen.to(device), eval_ds, device, batch_size=128, return_logits=True)
    if p_frz.ndim == 1:
        p_frz = p_frz.reshape(-1, 1)
    a_frz = macro_auroc(y, p_frz)

    # 3. permute the quality vector so the router reads the wrong image's quality
    w = backbone_weights(base, eval_ds, device)
    rng = np.random.default_rng(0)
    shuffled_scores = []
    for _ in range(args.shuffles):
        perm = rng.permutation(len(eval_ds))

        class PermutedQuality(torch.utils.data.Dataset):
            def __init__(self, inner, perm):
                self.inner = inner
                self.perm = perm

            def __len__(self):
                return len(self.inner)

            def __getitem__(self, i):
                (l, r), m, _q, yb = self.inner[i]
                _l2, _r2, (ql2, qr2), _ = self.inner[int(self.perm[i])]
                return (l, r), m, (ql2, qr2), yb

        wrapped = PermutedQuality(eval_ds, perm)
        p_pm, _y, _ = collect_probs(base, wrapped, device, batch_size=128, return_logits=True)
        if p_pm.ndim == 1:
            p_pm = p_pm.reshape(-1, 1)
        shuffled_scores.append(macro_auroc(y, p_pm))
    shuffled_scores = np.asarray(shuffled_scores)

    payload = {
        "arm": args.arm,
        "n": int(y.shape[0]),
        "macro_auroc_as_trained": a_raw,
        "macro_auroc_uniform_backbone_mix": a_uni,
        "delta_uniform": a_uni - a_raw,
        "macro_auroc_zeroed_router": a_frz,
        "delta_zeroed": a_frz - a_raw,
        "macro_auroc_permuted_quality_mean": float(shuffled_scores.mean()),
        "macro_auroc_permuted_quality_sd": float(shuffled_scores.std(ddof=1))
        if shuffled_scores.size > 1
        else 0.0,
        "delta_permuted": float(shuffled_scores.mean()) - a_raw,
        "n_shuffles": int(len(shuffled_scores)),
        "backbone_weight_mean": w.mean(axis=0).tolist() if w.size else None,
        "backbone_weight_sd": w.std(axis=0).tolist() if w.size else None,
        "backbone_weight_min": w.min(axis=0).tolist() if w.size else None,
        "backbone_weight_max": w.max(axis=0).tolist() if w.size else None,
        "max_abs_prob_change_uniform": float(np.abs(p_raw - p_uni).max()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
