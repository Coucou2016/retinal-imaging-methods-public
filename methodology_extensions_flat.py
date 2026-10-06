#!/usr/bin/env python
"""The three methods modifications, isolated in one flat file.

A *reading aid* for cross-review, not new science: each item below is the same
parametrisation used by ``QualityAware.py`` / ``quality_gate.py`` in this
snapshot, reduced to its essential maths so a reviewer can check the
monotonicity and routing claims without reading the whole model.

Nothing here is imported by the pipeline.  Run it directly for its assertions:

    python methodology_extensions_flat.py
"""

from __future__ import annotations

import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    """Inverse of ``softplus``; mirrors ``QualityAware._inv_softplus``."""
    y = max(float(y), 1e-8)
    return float(math.log(math.expm1(y)))


class MonotoneQualityRouter(nn.Module):
    """B1: monotone bounded quality fusion (mirror of ``QualityAware`` monotone).

    Learns three *global* stratum weights and orders them by construction,
    instead of learning three free logits and hoping the ordering emerges.

    Parametrisation (identical to ``mono_delta`` in ``QualityAware.py``)
    -------------------------------------------------------------------
    ``inc = softplus(delta) + 1e-6`` with ``delta`` the free 3-vector, then::

        cum = cumsum(inc)        # strictly positive and increasing
        w   = cum / cum[-1]      # w[-1] is exactly 1

    so ``w = (bad, usable, good)`` satisfies ``0 < bad < usable < good = 1``.
    :meth:`quality_scalar` then returns the dot product of a three-way quality
    distribution ``q = (good, usable, bad)`` with the aligned weights: a bounded
    scalar in ``[w_bad, 1]``.

    Normalising ``good`` to exactly 1 means a perfect-quality sample is never
    down-weighted.  The ordering ``bad <= usable <= good`` is structural, not a
    penalty term.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialised to approximately (0, 0.5, 1), as in QualityAware.
        self.mono_delta = nn.Parameter(
            torch.tensor(
                [_inv_softplus(1e-4), _inv_softplus(0.5), _inv_softplus(0.5)],
                dtype=torch.float32,
            )
        )

    def monotone_weights(self) -> torch.Tensor:
        """Return ``(bad, usable, good)`` with ``good == 1`` and no inversions."""
        inc = F.softplus(self.mono_delta) + 1e-6
        cum = torch.cumsum(inc, dim=0)
        return cum / cum[-1].clamp_min(1e-6)

    def quality_scalar(self, q: torch.Tensor) -> torch.Tensor:
        """q: (N, 3) as (good, usable, bad) -> (N, 1) bounded monotone weight."""
        w_bug = self.monotone_weights()                      # (bad, usable, good)
        w = torch.stack([w_bug[2], w_bug[1], w_bug[0]])      # align to (good, usable, bad)
        return (q * w.view(1, 3)).sum(dim=-1, keepdim=True)


class QualityConditionedBackboneRouter(nn.Module):
    """E5: softmax over backbone heads conditioned on the quality vector q.

    Mirrors ``quality_gate.QualityBackboneRouter``: ``w = softmax(MLP(q))`` in
    the simplex over H backbones, and ``QualityAware`` uses those weights in
    place of the fixed ensemble mix.  The last layer is zero-initialised, so at
    initialisation the weights are exactly uniform and the router reproduces the
    released mean/soft-vote behaviour before it learns anything.
    """

    def __init__(self, n_quality: int = 3, n_backbones: int = 3, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_quality, hidden),
            nn.SELU(True),
            nn.Linear(hidden, n_backbones),
        )
        # Near-uniform init: early training ~= mean ensemble (as in the repo).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def weights(self, q: torch.Tensor) -> torch.Tensor:
        """q: (N, 3) -> w: (N, H), rows summing to 1."""
        return F.softmax(self.net(q), dim=1)

    def forward(self, features: list[torch.Tensor], q: torch.Tensor) -> torch.Tensor:
        """features: H tensors of (N, d) -> fused (N, d)."""
        w = self.weights(q)
        stacked = torch.stack(features, dim=1)          # (N, H, d)
        return (stacked * w.unsqueeze(-1)).sum(dim=1)   # (N, d)


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """B2: partial-label multi-task loss (mirror of ``utils/run.py``).

    A dataset that annotates only a subset of endpoints supplies ``-1`` (or NaN)
    for the labels it never observes.  Those entries are masked out, so
    co-training cannot invent supervision it does not have.
    """
    targets = targets.float()
    mask = torch.isfinite(targets) & (targets >= 0)
    safe = torch.where(mask, targets.clamp(0.0, 1.0), torch.zeros_like(targets))
    per = F.binary_cross_entropy_with_logits(
        logits, safe, reduction="none", pos_weight=pos_weight
    )
    denom = mask.float().sum().clamp_min(1.0)
    return (per * mask.float()).sum() / denom


if __name__ == "__main__":
    torch.manual_seed(0)

    router = MonotoneQualityRouter()
    w = router.monotone_weights()
    bad, usable, good = w[0].item(), w[1].item(), w[2].item()
    assert 0.0 < bad < usable < good, f"ordering violated: {(bad, usable, good)}"
    assert abs(good - 1.0) < 1e-6, "good must normalise to exactly 1"

    q = torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.1, 0.9]])
    scalar = router.quality_scalar(q)
    assert scalar.shape == (2, 1)
    # Better quality must never yield a smaller weight.
    assert scalar[0, 0] > scalar[1, 0], "quality weight is not monotone in q"

    # A distribution that puts all mass on `good` gets the full weight of 1.
    assert abs(router.quality_scalar(torch.tensor([[1.0, 0.0, 0.0]]))[0, 0].item() - 1.0) < 1e-5

    backbone = QualityConditionedBackboneRouter()
    feats = [torch.randn(4, 16) for _ in range(3)]
    q4 = torch.randn(4, 3).softmax(dim=1)
    wb = backbone.weights(q4)
    assert torch.allclose(wb.sum(dim=1), torch.ones(4), atol=1e-5)
    assert backbone(feats, q4).shape == (4, 16)

    logits = torch.randn(4, 2)
    targets = torch.tensor([[1.0, -1.0], [0.0, 1.0], [1.0, 0.0], [0.0, float("nan")]])
    loss = masked_bce_with_logits(logits, targets)
    assert float(loss) > 0.0
    # Missing labels must not contribute: all-missing -> zero loss.
    all_missing = torch.full((4, 2), -1.0)
    assert float(masked_bce_with_logits(logits, all_missing)) == 0.0

    print("methodology_extensions_flat.py: all structural assertions passed")
    print(f"  monotone weights (bad, usable, good) = ({bad:.4f}, {usable:.4f}, {good:.4f})")
