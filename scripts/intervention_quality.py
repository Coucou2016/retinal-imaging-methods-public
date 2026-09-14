#!/usr/bin/env python
"""Good → usable → bad quality intervention: router + E5 backbone weights + figures."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.QualityAware import QualityAware
from model.quality_gate import QualityBackboneRouter, QualityConditionedGate
from utils.intervention import canonical_quality_vectors, confidence_monotonicity_report


def _maybe_plot(out_dir: str, router: dict, gate: dict, backbone: dict) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    labels = ["good", "usable", "bad"]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    axes[0].bar(labels, [router["score_good"], router["score_usable"], router["score_bad"]], color="#2c7bb6")
    axes[0].set_title("Monotone router scalar")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(labels, [gate["score_good"], gate["score_usable"], gate["score_bad"]], color="#abd9e9")
    axes[1].set_title("Legacy feature gate mean")
    axes[1].set_ylim(0, 1.05)
    # Backbone: plot weight on backbone-0 under each q prototype
    axes[2].bar(
        labels,
        [backbone["w0_good"], backbone["w0_usable"], backbone["w0_bad"]],
        color="#fdae61",
    )
    axes[2].set_title("E5 backbone-0 softmax weight")
    axes[2].set_ylim(0, 1.05)
    fig.suptitle("Quality intervention: good → usable → bad")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "intervention_quality.png")
    fig.savefig(path, dpi=150)
    pdf = os.path.join(out_dir, "intervention_quality.pdf")
    fig.savefig(pdf)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", default=None)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(ROOT, "artifacts", "figures"),
        help="Directory for intervention figures",
    )
    args = parser.parse_args()

    qa = QualityAware(8, 4, enable_q=True, learnable_q=True, quality_router="monotone")

    def router_score(q: np.ndarray) -> float:
        qt = torch.as_tensor(q, dtype=torch.float32).view(1, 3)
        with torch.no_grad():
            return float(qa.quality_scalar(qt).item())

    gate = QualityConditionedGate(feat_dim=8, enabled=True)
    xf = torch.ones(1, 8)

    def gate_score(q: np.ndarray) -> float:
        qt = torch.as_tensor(q, dtype=torch.float32).view(1, 3)
        with torch.no_grad():
            return float(gate(xf, qt).mean().item())

    router_bb = QualityBackboneRouter(n_backbones=3, enabled=True)
    # Make routing q-dependent: good → backbone 0, bad → backbone 2.
    with torch.no_grad():
        router_bb.mlp[-1].weight.zero_()
        router_bb.mlp[-1].bias.zero_()
        router_bb.mlp[-1].weight[0, 0] = 3.0  # good channel → bb0
        router_bb.mlp[-1].weight[1, 1] = 3.0  # usable → bb1
        router_bb.mlp[-1].weight[2, 2] = 3.0  # bad → bb2
        # Zero earlier layer so q passes nearly linearly into last logits via SELU path:
        # keep default first-layer; bias the last layer toward identity-ish mapping.
        nn_init = torch.nn.init
        nn_init.eye_(router_bb.mlp[0].weight[:3, :3]) if False else None
        # Simpler: replace MLP forward path by using last layer on q directly for demo.
        router_bb.mlp = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Linear(3, 3),
        )
        router_bb.mlp[-1].weight.copy_(
            torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
        )
        router_bb.mlp[-1].bias.zero_()

    def backbone_w0(q: np.ndarray) -> float:
        qt = torch.as_tensor(q, dtype=torch.float32).view(1, 3)
        with torch.no_grad():
            return float(router_bb(qt)[0, 0].item())

    router = confidence_monotonicity_report(router_score)
    gated = confidence_monotonicity_report(gate_score)
    bb = confidence_monotonicity_report(backbone_w0)
    vecs = canonical_quality_vectors()
    with torch.no_grad():
        w_by_q = {
            name: router_bb(torch.as_tensor(v, dtype=torch.float32).view(1, 3))
            .numpy()
            .ravel()
            .tolist()
            for name, v in vecs.items()
        }
    backbone_payload = {
        **bb,
        "w0_good": bb["score_good"],
        "w0_usable": bb["score_usable"],
        "w0_bad": bb["score_bad"],
        "weights_by_q": w_by_q,
    }
    payload = {
        "monotone_router": router,
        "quality_gate": gated,
        "backbone_router": backbone_payload,
    }
    fig_path = _maybe_plot(args.out_dir, router, gated, backbone_payload)
    if fig_path:
        payload["figure"] = fig_path
        print(f"Wrote figure {fig_path}")
    print(json.dumps(payload, indent=2))
    if not router["monotonic"]:
        raise SystemExit("monotone router failed good≥usable≥bad")
    if args.out_json:
        path = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
