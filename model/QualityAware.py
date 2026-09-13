import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

QualityRouter = Literal["fixed", "free_linear", "monotone"]


def _inv_softplus(y: float) -> float:
    y = max(float(y), 1e-8)
    return float(math.log(math.expm1(y)))


class QualityAware(nn.Module):
    """Bilinear quality fusion with fixed, free-linear, or monotone routing.

    Fixed paper weights: good=1, usable=0.5, bad=0.
    Monotone (default learnable path): bad ≤ usable ≤ good in [0, 1].
    Free linear: unconstrained ``nn.Linear(3, 1)`` ablation only.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        enable_q: bool,
        learnable_q: bool,
        quality_router: QualityRouter | str = "fixed",
    ):
        super().__init__()

        self.enable_q = enable_q
        self.learnable_q = bool(learnable_q)
        router = str(quality_router)
        if not self.learnable_q:
            router = "fixed"
        elif router in ("fixed",) and self.learnable_q:
            # learnable_q=True with unspecified/fixed → monotone default
            router = "monotone"
        if router not in ("fixed", "free_linear", "monotone"):
            raise ValueError(
                f"quality_router must be fixed|free_linear|monotone; got {quality_router!r}"
            )
        self.quality_router: QualityRouter = router  # type: ignore[assignment]

        if self.enable_q:
            self.q_fc: nn.Linear | None = None
            self.mono_delta: nn.Parameter | None = None
            if self.quality_router == "monotone":
                # Increments softplus → cumsum / max so good=1; init ≈ (0, 0.5, 1).
                self.mono_delta = nn.Parameter(
                    torch.tensor(
                        [
                            _inv_softplus(1e-4),
                            _inv_softplus(0.5),
                            _inv_softplus(0.5),
                        ],
                        dtype=torch.float32,
                    )
                )
            else:
                self.q_fc = nn.Linear(3, 1)
                with torch.no_grad():
                    self.q_fc.weight[0, 0] = 1
                    self.q_fc.weight[0, 1] = 0.5
                    self.q_fc.weight[0, 2] = 0
                    self.q_fc.bias[0] = 0
                if self.quality_router == "fixed":
                    self.q_fc.requires_grad_(False)
            self.q_fuse = nn.Bilinear(in_dim + 1, 2, out_dim, False)
            self.post_q = nn.Sequential(nn.SELU(True), nn.Linear(out_dim, out_dim))
        else:
            self.pre_m = nn.Sequential(nn.SELU(True), nn.Linear(in_dim, out_dim))

    def monotone_weights(self) -> torch.Tensor:
        """Return (bad, usable, good) with bad ≤ usable ≤ good and good == 1."""
        if self.mono_delta is None:
            raise RuntimeError("monotone_weights requires quality_router='monotone'")
        inc = F.softplus(self.mono_delta) + 1e-6
        cum = torch.cumsum(inc, dim=0)
        return cum / cum[-1].clamp_min(1e-6)

    def quality_scalar(self, q: torch.Tensor) -> torch.Tensor:
        """Map (B, 3) quality probs → (B, 1) scalar weight."""
        if self.quality_router == "monotone":
            w = self.monotone_weights()
            return (q * w.view(1, 3)).sum(dim=-1, keepdim=True)
        assert self.q_fc is not None
        return self.q_fc(q)

    def forward(self, xf: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        dev = q.device
        bat = q.shape[0]
        if self.enable_q:
            xf = torch.cat([torch.ones(bat, 1, device=dev), xf], dim=1)
            q = torch.cat([torch.ones(bat, 1, device=dev), self.quality_scalar(q)], dim=1)
            xqf = self.q_fuse(xf, q)
            return self.post_q(xqf)
        return self.pre_m(xf)
