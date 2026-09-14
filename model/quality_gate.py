"""Optional quality-conditioned backbone routing (E5) and soft quality auxiliary loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_quality_ce(pred_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a soft 3-way quality target (good/usable/bad).

    ``pred_logits``: (B, 3), ``target_probs``: (B, 3) non-negative, row-normalized preferred.
    """
    target = target_probs.clamp_min(0.0)
    denom = target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    target = target / denom
    log_p = F.log_softmax(pred_logits, dim=-1)
    return -(target * log_p).sum(dim=-1).mean()


class QualityBackboneRouter(nn.Module):
    """E5 primary: softmax over backbone heads conditioned on quality vector q.

    Maps a 3-way (good, usable, bad) quality distribution to mixture weights over
    ``n_backbones`` foundation heads. Prefer this over feature attenuation.
    """

    def __init__(self, n_backbones: int = 3, hidden: int = 32, enabled: bool = False):
        super().__init__()
        self.enabled = bool(enabled)
        self.n_backbones = int(n_backbones)
        if self.enabled:
            self.mlp = nn.Sequential(
                nn.Linear(3, hidden),
                nn.SELU(True),
                nn.Linear(hidden, self.n_backbones),
            )
            # Near-uniform init so early training ≈ mean ensemble.
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            self.mlp = None

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Return (B, H) softmax weights. Identity uniform when disabled."""
        if q.dim() == 1:
            q = q.unsqueeze(0)
        b = q.shape[0]
        if not self.enabled or self.mlp is None:
            return torch.full(
                (b, self.n_backbones),
                1.0 / float(self.n_backbones),
                device=q.device,
                dtype=q.dtype,
            )
        return F.softmax(self.mlp(q), dim=-1)


class QualityConditionedGate(nn.Module):
    """Legacy E5 attenuation: multiply backbone features by a gate in (0, 1].

    Kept for ablation / backward compatibility. Prefer ``QualityBackboneRouter``.
    """

    def __init__(self, feat_dim: int, hidden: int = 32, enabled: bool = False):
        super().__init__()
        self.enabled = bool(enabled)
        if self.enabled:
            self.mlp = nn.Sequential(
                nn.Linear(3, hidden),
                nn.SELU(True),
                nn.Linear(hidden, feat_dim),
                nn.Sigmoid(),
            )
            # Near-identity init: bias so sigmoid ≈ 1
            nn.init.zeros_(self.mlp[-2].weight)
            nn.init.constant_(self.mlp[-2].bias, 4.0)
        else:
            self.mlp = None

    def forward(self, xf: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.mlp is None:
            return xf
        gate = self.mlp(q)
        return xf * gate


class QualityAuxHead(nn.Module):
    """Predict 3-way quality logits from backbone features (BRSET λ_q hook)."""

    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SELU(True),
            nn.Linear(hidden, 3),
        )

    def forward(self, xf: torch.Tensor) -> torch.Tensor:
        return self.net(xf)
