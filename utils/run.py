import os
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.contrib.handlers.tensorboard_logger import OptimizerParamsHandler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.metrics.roc_auc import ROC_AUC
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import CosineAnnealingScheduler, create_lr_scheduler_with_warmup
from ignite.handlers.stores import EpochOutputStore
from ignite.metrics import Accuracy, Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from utils.metrics import AveragePrecision, SensitivityScore, SpecificityScore
from model.quality_gate import soft_quality_ce


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE that supervises only present labels (target >= 0); -1/NaN = missing."""
    targets = targets.float()
    mask = torch.isfinite(targets) & (targets >= 0)
    # Clamp missing entries so BCE does not NaN; they are zeroed by the mask.
    safe = torch.where(mask, targets.clamp(0.0, 1.0), torch.zeros_like(targets))
    per = F.binary_cross_entropy_with_logits(
        logits, safe, reduction="none", pos_weight=pos_weight
    )
    denom = mask.float().sum().clamp_min(1.0)
    return (per * mask.float()).sum() / denom


class MaskedBCEWithLogitsLoss(nn.Module):
    """Partial-label multitask loss: ignore targets marked missing (<0 or NaN)."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight
        if pw is not None and pw.device != logits.device:
            pw = pw.to(logits.device)
        return masked_bce_with_logits(logits, targets, pos_weight=pw)


def save_py(py: List[Tuple[np.ndarray, np.ndarray]] , path: str):
    def to_cpu_numpy(t: torch.Tensor): return t.cpu().numpy()
    p, y = zip(*py)
    p = np.concatenate(list(map(to_cpu_numpy, p)))
    y = np.concatenate(list(map(to_cpu_numpy, y)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, p=p, y=y)


def single_fastds_run(model: nn.Module,
                      lr: float, epochs: int, epochs_factor: int,
                      warmup_lr: float, warmup_epochs: int,
                      tds, vds, xdss: list,
                      pb: Callable, bs: int, device, tbdir: str,
                      save: bool = False,
                      save_prob: bool = False,
                      eval_train: bool = False,
                      **args):
    pos_weight = args.get("pos_weight")
    use_masked = bool(args.get("masked_bce", True))
    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    else:
        pw = None
    if use_masked:
        loss = MaskedBCEWithLogitsLoss(pos_weight=pw)
    elif pw is not None:
        loss = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        loss = nn.BCEWithLogitsLoss()

    optim = AdamW(model.parameters(), lr, weight_decay=args.get("weight_decay", 0.01))
    scheduler = CosineAnnealingScheduler(optim, "lr", lr, 0.0, epochs // epochs_factor)
    scheduler = create_lr_scheduler_with_warmup(scheduler, warmup_lr, warmup_epochs)

    if not isinstance(xdss, list):
        xdss = [xdss]

    def _label_positive(y) -> float:
        if hasattr(y, "numel"):
            yy = y.float()
            present = yy[yy >= 0] if yy.numel() > 1 else yy
            return float(present.sum().item() if present.numel() > 1 else (present.item() if present.numel() else 0.0))
        arr = np.asarray(y, dtype=np.float64).reshape(-1)
        present = arr[np.isfinite(arr) & (arr >= 0)]
        return float(present.sum() if present.size else 0.0)

    sampler = None
    balance_sampler = args.get("balance_sampler", False)
    if balance_sampler:
        ys = []
        for i in range(len(tds)):
            ys.append(_label_positive(tds[i][-1]))
        pos = sum(1 for y in ys if y >= 0.5)
        neg = len(ys) - pos
        if pos <= 0 or neg <= 0:
            raise ValueError("balance_sampler requires both positive and negative labels")
        ws = [(1.0 / pos) if y >= 0.5 else (1.0 / neg) for y in ys]
        sampler = WeightedRandomSampler(ws, len(ys), generator=torch.Generator().manual_seed(430))
    train_loader = DataLoader(tds, batch_size=bs, shuffle=not balance_sampler, sampler=sampler, pin_memory=False)
    valid_loader = DataLoader(vds, batch_size=bs, shuffle=False, pin_memory=False) if vds is not None else None
    test_loaders = [DataLoader(xds, batch_size=bs, shuffle=False, pin_memory=False) for xds in xdss]

    def _val_has_positive() -> bool:
        if vds is None or len(vds) == 0:
            return False
        for i in range(len(vds)):
            if _label_positive(vds[i][-1]) >= 0.5:
                return True
        return False

    skip_vds = vds is None or not _val_has_positive()

    use_amp = str(device).startswith("cuda")
    lambda_q = float(args.get("lambda_q", 0.0) or 0.0)

    if lambda_q > 0:

        def _train_step(engine: Engine, batch):
            model.train()
            optim.zero_grad()
            x, y = pb(batch, device, False)
            logits = model(x)
            loss_d = loss(logits, y)
            total = loss_d
            if hasattr(model, "collect_quality_aux_logits"):
                q_logits = model.collect_quality_aux_logits()
                if q_logits is not None:
                    ql, qr = x[2]
                    q_tgt = 0.5 * (ql.float() + qr.float())
                    total = loss_d + lambda_q * soft_quality_ce(q_logits, q_tgt)
            total.backward()
            optim.step()
            return total.detach()

        trainer = Engine(_train_step)
    else:
        trainer = create_supervised_trainer(
            model, optim, loss, device,
            prepare_batch=pb, amp_mode="amp" if use_amp else False
        )

    def binize(py): return (torch.round(F.sigmoid(py[0])), py[1])
    def sig(py): return ((F.sigmoid(py[0]), py[1]))
    # Multi-label (K>1) logits break binary ignite AUROC/sen/spe; keep loss only.
    if args.get("simple_metrics", False):
        metrics = {"loss": Loss(loss)}
    else:
        metrics = {
            "loss": Loss(loss),
            "acc": Accuracy(binize),
            "sen": SensitivityScore(binize),
            "spe": SpecificityScore(binize),
            "auc": ROC_AUC(sig),
            "ap": AveragePrecision(sig),
        }

    evaluator = create_supervised_evaluator(model, metrics, device, prepare_batch=pb)
    validator = create_supervised_evaluator(model, metrics, device, prepare_batch=pb)
    testers = [create_supervised_evaluator(model, metrics, device, prepare_batch=pb) for _ in range(len(xdss))]

    if save_prob:
        eos_eval = EpochOutputStore(sig)
        eos_eval.attach(evaluator, "py")
        eos_vald = EpochOutputStore(sig)
        eos_vald.attach(validator, "py")
        eos_tests = [EpochOutputStore(sig) for _ in range(len(xdss))]
        for i in range(len(xdss)):
            eos_tests[i].attach(testers[i], "py")

    pbar = ProgressBar()
    pbar.attach(trainer)
    pbar.attach(validator)
    for tester in testers:
        pbar.attach(tester)

    trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

    @trainer.on(Events.EPOCH_COMPLETED)
    def train_epoch_completed(engine: Engine):
        if save:
            torch.save(model.state_dict(), os.path.join(tbdir, "ckpt", f"{engine.state.epoch}.pt"))
        if eval_train:
            evaluator.run(train_loader) # too slow
        if not skip_vds:
            validator.run(valid_loader)
            if save_prob:
                save_py(validator.state.py, os.path.join(tbdir, "ckpt", f"py_{engine.state.epoch}_val.npz"))
        if engine.state.epoch >= warmup_epochs + epochs - 2:
            for i, (tester, test_loader) in enumerate(zip(testers, test_loaders)):
                tester.run(test_loader)
                if save_prob:
                    save_py(tester.state.py, os.path.join(tbdir, "ckpt", f"py_{engine.state.epoch}_test{i}.npz"))

    os.makedirs(tbdir, exist_ok=True)
    if args.get("tensorboard", True):
        tb_logger = TensorboardLogger(log_dir=tbdir)
        tb_logger.attach(trainer, OptimizerParamsHandler(optim), Events.EPOCH_STARTED)
        for tag, engine in [("train", evaluator), ("valid", validator)] + [
            (f"test_{i}", tester) for i, tester in enumerate(testers)
        ]:
            tb_logger.attach_output_handler(
                engine,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names="all",
                global_step_transform=global_step_from_engine(trainer),
            )

    if save:
        os.makedirs(os.path.join(tbdir, "ckpt"), exist_ok=True)

    trainer.run(train_loader, warmup_epochs + epochs - 1)
