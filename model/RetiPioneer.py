from typing import List
import warnings

import torch
import torch.nn as nn

from model.QualityAware import QualityAware, QualityRouter

# Ensemble over the three backbone heads:
#   released_code       — train: softmax-weighted mix (T=0.1); eval: max
#                         (upstream Reti-Pioneer released training code)
#   published_soft_vote — soft mix at train and eval
#   mean                — arithmetic mean of heads (train and eval)
#   temp_mean           — temperature-scaled softmax mix at train and eval
# Legacy alias: ensemble="paper" → released_code (DeprecationWarning).
# For num_classes>1 the mix/max is applied independently per class.
VALID_ENSEMBLES = ("released_code", "published_soft_vote", "mean", "temp_mean", "paper")


def normalize_ensemble(ensemble: str) -> str:
    name = str(ensemble)
    if name == "paper":
        warnings.warn(
            'ensemble="paper" is deprecated; use ensemble="released_code" '
            "(train soft / eval max, matching released upstream code).",
            DeprecationWarning,
            stacklevel=2,
        )
        return "released_code"
    if name not in VALID_ENSEMBLES:
        raise ValueError(f"ensemble must be one of {VALID_ENSEMBLES}; got {ensemble!r}")
    return name


class FuseBase(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        base_out_size: int,
        meta_size: int,
        num_classes: int,
        fuse_dim: int,
        enable_q: bool = True,
        learnable_q: bool = False,
        quality_router: QualityRouter | str = "fixed",
        flip_r: bool = True,
    ) -> None:
        super().__init__()

        self.base = base
        self.fuse_dim = fuse_dim
        self.quality_aware = QualityAware(
            base_out_size,
            self.fuse_dim,
            enable_q,
            learnable_q,
            quality_router=quality_router,
        )
        self.m_fuse = nn.Bilinear(self.fuse_dim * 2 + 1, meta_size + 1, num_classes, False)
        self.flip_r = flip_r

    def forward(self, xmq: List[torch.Tensor]) -> torch.Tensor:
        lr, m, qs = xmq
        dev = m.device
        bat = m.shape[0]

        if self.flip_r:
            lr = [lr[0], lr[1].flip([-1])]
        else:
            lr = [lr[0], lr[1]]

        xqf2 = [torch.ones(bat, 1, device=dev)]
        for x, q in zip(lr, qs):
            xf = self.base(x)
            xqf2.append(self.quality_aware(xf, q))
        xqf2 = torch.cat(xqf2, dim=1)
        m = torch.cat([torch.ones(bat, 1, device=dev), m], dim=1)
        xqmf = self.m_fuse(xqf2, m)

        return xqmf


class ComplexModel(nn.Module):
    def __init__(
        self,
        backbones: List[nn.Module],
        base_out_sizes: List[int],
        mid_size: int,
        fuse_dim: int,
        n_meta: int,
        num_classes: int,
        learnable_q: bool = False,
        enable_q: bool = True,
        ensemble: str = "released_code",
        quality_router: QualityRouter | str = "fixed",
    ):
        super().__init__()
        ensemble = normalize_ensemble(ensemble)
        self.num_classes = num_classes
        self.learnable_q = learnable_q
        self.enable_q = enable_q
        self.ensemble = ensemble
        self.quality_router = quality_router
        if learnable_q and quality_router == "fixed":
            quality_router = "monotone"
            self.quality_router = quality_router
        self.models = nn.ModuleList(
            [
                nn.Sequential(
                    FuseBase(
                        backbone,
                        size,
                        n_meta,
                        mid_size,
                        fuse_dim,
                        enable_q=enable_q,
                        learnable_q=learnable_q,
                        quality_router=quality_router,
                        flip_r=False,
                    ),
                    nn.SELU(True),
                    nn.Linear(mid_size, num_classes),
                )
                for backbone, size in zip(backbones, base_out_sizes)
            ]
        )

    def forward(self, batch):
        (l, r), m, qs = batch
        head_logits = []
        for i in range(len(self.models)):
            head_logits.append(self.models[i](((l[i], r[i]), m, qs)))
        # (batch, n_heads, num_classes) — K=1 stays (B, H, 1) → (B, 1) after reduce
        heads = torch.stack(head_logits, dim=1)

        if self.ensemble == "mean":
            return heads.mean(dim=1)

        temp = 0.1
        # released_code: soft in train, max in eval (upstream main.py behavior)
        # published_soft_vote / temp_mean: soft always
        if self.ensemble == "released_code":
            use_soft = self.training
        else:
            use_soft = True  # published_soft_vote, temp_mean
        if use_soft:
            weights = torch.softmax(heads / temp, dim=1)
            return (weights * heads).sum(dim=1)
        return heads.max(dim=1).values


def get_reti_pioneer(
    fast: bool,
    num_classes: int = 1,
    learnable_q: bool = False,
    enable_q: bool = True,
    ensemble: str = "released_code",
    quality_router: QualityRouter | str = "fixed",
    n_meta: int = 3 + 7,
):
    """Build Reti-Pioneer. K=1 is the paper clone; K>1 is a joint multi-label head."""
    ensemble = normalize_ensemble(ensemble)
    if learnable_q and quality_router == "fixed":
        quality_router = "monotone"
    if fast:
        backbones = [nn.Identity(), nn.Identity(), nn.Identity()]
    else:
        from model.base import get_RETFound, get_SwinB, get_VimS

        backbones = [get_RETFound(), get_SwinB(), get_VimS()]
    base_out_sizes = [1024, 1024, 384]
    mid_size = 256
    fuse_dim = 256
    return ComplexModel(
        backbones,
        base_out_sizes,
        mid_size,
        fuse_dim,
        n_meta,
        num_classes,
        learnable_q=learnable_q,
        enable_q=enable_q,
        ensemble=ensemble,
        quality_router=quality_router,
    )
