#!/usr/bin/env python
"""Smoke: good → usable → bad quality intervention on router / gate scalars."""

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
from model.quality_gate import QualityConditionedGate
from utils.intervention import confidence_monotonicity_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", default=None)
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

    router = confidence_monotonicity_report(router_score)
    gated = confidence_monotonicity_report(gate_score)
    payload = {"monotone_router": router, "quality_gate": gated}
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
