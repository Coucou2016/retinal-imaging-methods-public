#!/usr/bin/env python
"""Evaluate saved checkpoints on UKB-compressed tensors."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from ignite.utils import to_onehot
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset.UKBDataset import UKBDatasetFast
from model.RetiPioneer import get_reti_pioneer, normalize_ensemble
from reti_pioneer.config import load_config
from reti_pioneer.constants import PUBLIC_DATASETS
from reti_pioneer.data_paths import ukb_compressed_ready
from reti_pioneer.label_map import (
    DEFAULT_CROSS_HEADS,
    MappedHead,
    assert_clinical_alignment,
    resolve_cross_heads,
    slice_mapped,
)
from reti_pioneer.split import (
    assert_calibration_disjoint,
    dataset_labels,
    dataset_meta_matrix,
    dataset_meta_names,
    dataset_patient_ids,
    load_calibration_idx,
    load_split,
    load_test_idx,
    nested_calibration_from_val,
    patient_level_train_val_indices,
    resolve_split_indices,
    subset_for_split,
)
from utils.calibration import (
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_calibration_intercept_slope,
    fit_temperature,
    net_benefit_at_threshold,
    per_class_decision_curves,
    sensitivity_at_specificity,
    treat_all_net_benefit,
    treat_none_net_benefit,
    youden_operating_point,
)
from utils.functional import pb_l_r_m_q_y


def resolve_ckpt_path(ckpt: str) -> str:
    """Resolve a run directory, ckpt/ folder, or direct .pt path."""
    ckpt = os.path.abspath(ckpt)
    if ckpt.endswith(".pt") and os.path.isfile(ckpt):
        return ckpt
    if os.path.isdir(ckpt):
        nested = os.path.join(ckpt, "ckpt")
        search_dir = nested if os.path.isdir(nested) else ckpt
        pts = sorted(glob.glob(os.path.join(search_dir, "*.pt")))
        if pts:
            return pts[-1]
        pts = sorted(glob.glob(os.path.join(ckpt, "**", "ckpt", "*.pt"), recursive=True))
        if pts:
            return pts[-1]
    raise FileNotFoundError(f"No checkpoint at {ckpt}")


def resolve_run_dir(ckpt: str) -> str:
    ckpt = os.path.abspath(ckpt)
    if ckpt.endswith(".pt"):
        return os.path.dirname(os.path.dirname(ckpt))
    split_path = os.path.join(ckpt, "split.npz")
    if os.path.isfile(split_path):
        return ckpt
    parent = os.path.dirname(ckpt.rstrip(os.sep))
    if os.path.isfile(os.path.join(parent, "split.npz")):
        return parent
    meta = os.path.join(ckpt, "run_meta.json")
    if os.path.isfile(meta):
        return ckpt
    return ckpt


def load_run_meta(run_dir: str) -> dict:
    path = os.path.join(run_dir, "run_meta.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def prepare_batch(batch, device, non_blocking):
    (l, r), m, (ql, qr), y = batch
    eth = m[:, -1]
    e = to_onehot(eth.long(), 7)
    m = torch.cat([m[:, :-1], e], 1)
    if y.dim() == 1:
        y = y.view(-1, 1)
    batch = (l, r), m, (ql, qr), y
    return pb_l_r_m_q_y(batch, device, non_blocking)


@torch.no_grad()
def collect_probs(model, dataset, device, batch_size: int = 64, return_logits: bool = False):
    model.eval()
    probs, labels, logits = [], [], []
    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        lr_pairs, ms, qs, ys = zip(*batch)
        l, r = zip(*lr_pairs)
        ql, qr = zip(*qs)

        to_t = lambda x: x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        if isinstance(l[0], list):
            l = [torch.stack([to_t(sample[k]) for sample in l]) for k in range(len(l[0]))]
            r = [torch.stack([to_t(sample[k]) for sample in r]) for k in range(len(r[0]))]
        else:
            l, r = torch.stack([to_t(x) for x in l]), torch.stack([to_t(x) for x in r])
        m = torch.stack([to_t(x) for x in ms])
        ql = torch.stack([to_t(x) for x in ql])
        qr = torch.stack([to_t(x) for x in qr])
        y = torch.stack([to_t(x).reshape(-1) for x in ys])
        xmq, y = prepare_batch(((l, r), m, (ql, qr), y), device, False)
        logit = model(xmq)
        logits.append(logit.cpu().numpy())
        probs.append(F.sigmoid(logit).cpu().numpy())
        labels.append(y.cpu().numpy())
    logits_np = np.concatenate(logits, axis=0)
    probs_np = np.concatenate(probs, axis=0)
    labels_np = np.concatenate(labels, axis=0)
    # Squeeze singleton class dims independently. Do NOT flatten multi-column
    # labels just because the model is K=1 (cross-dataset eval keeps test K).
    if probs_np.ndim == 2 and probs_np.shape[1] == 1:
        logits_np = logits_np.reshape(-1)
        probs_np = probs_np.reshape(-1)
    elif probs_np.ndim > 2:
        raise ValueError(f"Unexpected probs shape {probs_np.shape}")
    if labels_np.ndim == 2 and labels_np.shape[1] == 1:
        labels_np = labels_np.reshape(-1)
    if return_logits:
        return probs_np, labels_np, logits_np
    return probs_np, labels_np


def resolve_eval_split(
    run_dir: str,
    base_dataset: UKBDatasetFast,
    split: str,
    val_fraction: float,
    split_seed: int,
) -> tuple[object, str, dict]:
    """Return (subset, split_name, index_bundle).

    ``index_bundle`` carries train/val/test/calibration lists for leakage checks.
    """
    split_path = os.path.join(run_dir, "split.npz")
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx = None
    calibration_idx = None
    if os.path.isfile(split_path):
        train_idx, val_idx = load_split(split_path)
        test_idx = load_test_idx(split_path)
        calibration_idx = load_calibration_idx(split_path)
    elif split == "all" or val_fraction <= 0:
        return base_dataset, "all", {
            "train_idx": [],
            "val_idx": [],
            "test_idx": None,
            "calibration_idx": None,
        }
    else:
        labels = dataset_labels(base_dataset)
        pids = dataset_patient_ids(base_dataset)
        if len(np.unique(pids)) >= 4:
            train_idx, val_idx = patient_level_train_val_indices(
                pids, labels, val_fraction, split_seed
            )
        else:
            from reti_pioneer.split import stratified_train_val_indices

            train_idx, val_idx = stratified_train_val_indices(labels, val_fraction, split_seed)
    if split not in ("train", "val", "test", "all", "calibration", "cal"):
        raise ValueError(f"Unknown split {split!r}; use train, val, test, calibration, or all")
    if split == "test" and not test_idx:
        raise ValueError("split='test' needs test_idx in split.npz")
    if split in ("calibration", "cal") and not calibration_idx:
        raise ValueError("split='calibration' needs calibration_idx in split.npz")
    bundle = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "calibration_idx": calibration_idx,
        "patient_ids": dataset_patient_ids(base_dataset) if len(base_dataset) else None,
        "labels": dataset_labels(base_dataset) if len(base_dataset) else None,
    }
    return (
        subset_for_split(
            base_dataset, split, train_idx, val_idx, test_idx, calibration_idx
        ),
        split,
        bundle,
    )


def resolve_calibration_fit_ids(
    bundle: dict,
    eval_split: str,
    *,
    paper_mode: bool,
    split_seed: int,
) -> tuple[list[int], str]:
    """Pick calibration indices disjoint from the evaluation fold.

    If ``--split val --calibrate`` would fit and score the same set, nest a held-out
    calibration fold (or error in ``paper_mode``).
    """
    cal = bundle.get("calibration_idx")
    val = list(bundle.get("val_idx") or [])
    test = list(bundle.get("test_idx") or [])
    train = list(bundle.get("train_idx") or [])
    split_key = "calibration" if eval_split == "cal" else eval_split
    if split_key == "all":
        eval_ids = train + val + (test or []) + (cal or [])
    else:
        eval_ids = resolve_split_indices(split_key, train, val, test, cal)

    if cal:
        assert_calibration_disjoint(cal, eval_ids)
        return list(cal), "calibration_idx"

    if eval_split in ("test", "train") and val:
        assert_calibration_disjoint(val, eval_ids)
        return val, "val"

    if eval_split == "val":
        msg = (
            "--split val --calibrate would fit temperature on the same fold used for scoring. "
            "Provide calibration_idx in split.npz, score --split test, or allow nested calibration."
        )
        if paper_mode:
            raise ValueError(msg + " (paper_mode refuses silent leakage.)")
        pids = bundle.get("patient_ids")
        labels = bundle.get("labels")
        if pids is None or labels is None or len(val) < 4:
            raise ValueError(msg + " Nested calibration unavailable (too few val rows).")
        cal_ids, eval_only = nested_calibration_from_val(
            pids, val, labels, cal_fraction=0.4, seed=split_seed + 99
        )
        bundle["_nested_eval_idx"] = eval_only
        print(
            f"WARNING: nested calibration on val "
            f"(cal={len(cal_ids)} eval={len(eval_only)}); prefer explicit calibration_idx."
        )
        return cal_ids, "nested_val"

    if val:
        return val, "val"
    if train:
        return train, "train"
    raise ValueError("No fold available to fit temperature")


def _binary_scores(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, probs)), float(average_precision_score(labels, probs))


def _merge_operating_points(out: dict[str, float], labels: np.ndarray, probs: np.ndarray) -> None:
    """Attach Youden and sens@95%spec keys (binary)."""
    youden = youden_operating_point(labels, probs)
    out["youden_threshold"] = youden["threshold"]
    out["youden_sensitivity"] = youden["sensitivity"]
    out["youden_specificity"] = youden["specificity"]
    out["youden_j"] = youden["youden_j"]
    s95 = sensitivity_at_specificity(labels, probs, target_specificity=0.95)
    out["sens@95%spec"] = s95["sensitivity"]
    out["sens@95%spec_threshold"] = s95["threshold"]
    out["sens@95%spec_specificity"] = s95["specificity"]
    out["sens@95%spec_met_target"] = s95["met_target"]


def score_predictions(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    out: dict[str, float] = {}
    if labels.ndim == 1:
        auc, ap = _binary_scores(labels, probs)
        out["auroc"] = auc
        out["ap"] = ap
        out["ece"] = expected_calibration_error(labels, probs)
        out["brier"] = brier_score(labels, probs)
        out["nb@0.10"] = net_benefit_at_threshold(labels, probs, 0.10)
        out["treat_all_nb@0.10"] = treat_all_net_benefit(labels, 0.10)
        out["treat_none_nb@0.10"] = treat_none_net_benefit(labels, 0.10)
        _merge_operating_points(out, labels, probs)
        if class_names:
            out[f"class_{class_names[0]}_auroc"] = auc
        return out
    aucs, aps, eces, briers, nbs = [], [], [], [], []
    youden_js, youden_sens, youden_specs, youden_thrs = [], [], [], []
    s95_sens, s95_specs, s95_thrs, s95_mets = [], [], [], []
    for k in range(labels.shape[1]):
        auc, ap = _binary_scores(labels[:, k], probs[:, k])
        aucs.append(auc)
        aps.append(ap)
        eces.append(expected_calibration_error(labels[:, k], probs[:, k]))
        briers.append(brier_score(labels[:, k], probs[:, k]))
        nbs.append(net_benefit_at_threshold(labels[:, k], probs[:, k], 0.10))
        youden = youden_operating_point(labels[:, k], probs[:, k])
        youden_js.append(youden["youden_j"])
        youden_sens.append(youden["sensitivity"])
        youden_specs.append(youden["specificity"])
        youden_thrs.append(youden["threshold"])
        s95 = sensitivity_at_specificity(labels[:, k], probs[:, k], target_specificity=0.95)
        s95_sens.append(s95["sensitivity"])
        s95_specs.append(s95["specificity"])
        s95_thrs.append(s95["threshold"])
        s95_mets.append(s95["met_target"])
        out[f"class_{k}_auroc"] = auc
        if class_names and k < len(class_names):
            out[f"class_{class_names[k]}_auroc"] = auc
    out["auroc"] = float(np.nanmean(aucs)) if aucs else float("nan")
    out["ap"] = float(np.nanmean(aps)) if aps else float("nan")
    out["ece"] = float(np.nanmean(eces)) if eces else float("nan")
    out["brier"] = float(np.nanmean(briers)) if briers else float("nan")
    out["nb@0.10"] = float(np.nanmean(nbs)) if nbs else float("nan")
    out["youden_j"] = float(np.nanmean(youden_js)) if youden_js else float("nan")
    out["youden_sensitivity"] = float(np.nanmean(youden_sens)) if youden_sens else float("nan")
    out["youden_specificity"] = float(np.nanmean(youden_specs)) if youden_specs else float("nan")
    out["youden_threshold"] = float(np.nanmean(youden_thrs)) if youden_thrs else float("nan")
    out["sens@95%spec"] = float(np.nanmean(s95_sens)) if s95_sens else float("nan")
    out["sens@95%spec_threshold"] = float(np.nanmean(s95_thrs)) if s95_thrs else float("nan")
    out["sens@95%spec_specificity"] = float(np.nanmean(s95_specs)) if s95_specs else float("nan")
    out["sens@95%spec_met_target"] = float(np.nanmean(s95_mets)) if s95_mets else float("nan")
    # Macro treat-all / treat-none over classes for DCA context.
    ta, tn = [], []
    for k in range(labels.shape[1]):
        ta.append(treat_all_net_benefit(labels[:, k], 0.10))
        tn.append(treat_none_net_benefit(labels[:, k], 0.10))
    out["treat_all_nb@0.10"] = float(np.nanmean(ta)) if ta else float("nan")
    out["treat_none_nb@0.10"] = float(np.nanmean(tn)) if tn else float("nan")
    return out


def _print_scores(prefix: str, scores: dict[str, float]) -> None:
    extra = ""
    if "treat_all_nb@0.10" in scores:
        extra = (
            f"  treat-all={scores['treat_all_nb@0.10']:.4f}"
            f"  treat-none={scores['treat_none_nb@0.10']:.4f}"
        )
    op = ""
    if "youden_j" in scores and np.isfinite(scores["youden_j"]):
        op = (
            f"  YoudenJ={scores['youden_j']:.4f}@t={scores['youden_threshold']:.4f}"
            f"  sens@95%spec={scores['sens@95%spec']:.4f}"
            f"(spec={scores['sens@95%spec_specificity']:.4f})"
        )
    print(
        f"{prefix}  AUROC={scores['auroc']:.4f}  AP={scores['ap']:.4f}  "
        f"ECE={scores['ece']:.4f}  Brier={scores['brier']:.4f}  NB@0.10={scores['nb@0.10']:.4f}"
        f"{extra}{op}"
    )


def print_mapped_heads(heads: list[MappedHead], train_ds: str, test_ds: str) -> None:
    print(
        f"Endpoint-aware cross-cohort {train_ds} -> {test_ds} "
        "(alignment flags required; related ≠ identical gold standards)"
    )
    for h in heads:
        claim = "clinical_ok" if h.clinical_claim_allowed else "exploratory_only"
        print(
            f"  {h.task} [{h.alignment}/{h.kind}/{claim}]: "
            f"train[{h.train_index}]={h.train_name} -> "
            f"test[{h.test_index}]={h.test_name}"
        )


def print_subgroups(dataset, labels: np.ndarray, probs: np.ndarray) -> None:
    meta = dataset_meta_matrix(dataset)
    names = [n.lower() for n in dataset_meta_names(dataset)]
    if meta is None or not names:
        return
    n = len(labels)

    def _report(mask: np.ndarray, title: str) -> None:
        if int(mask.sum()) < 8:
            return
        sc = score_predictions(labels[mask], probs[mask] if probs.ndim == 1 else probs[mask])
        print(f"  subgroup {title} n={int(mask.sum())} AUROC={sc['auroc']:.4f}")

    if "gender" in names or "sex" in names:
        col = names.index("gender") if "gender" in names else names.index("sex")
        for val in np.unique(meta[:, col]):
            _report(meta[:, col] == val, f"sex={int(val)}")
    if "baselineage" in names or "age" in names:
        col = names.index("baselineage") if "baselineage" in names else names.index("age")
        age = meta[:, col]
        try:
            q1, q2 = np.quantile(age, [1 / 3, 2 / 3])
        except Exception:
            return
        _report(age <= q1, "age_tertile=0")
        _report((age > q1) & (age <= q2), "age_tertile=1")
        _report(age > q2, "age_tertile=2")
    _ = n


def _json_safe(obj):
    """Convert numpy / NaN / Inf so json.dump never raises on metrics."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", required=True, help="Path to .pt state dict or ckpt folder")
    parser.add_argument("--disease", default=None)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--dataset", default=None, choices=["odir", "brset", "rfmid", "ukb", "demo"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--test-data-dir",
        default=None,
        help="Score this cache instead of --data-dir (cross-dataset). Uses all samples.",
    )
    parser.add_argument(
        "--test-dataset",
        default=None,
        choices=["odir", "brset", "rfmid", "ukb", "demo"],
        help="Label ontology of --test-data-dir (required with --test-data-dir)",
    )
    parser.add_argument(
        "--heads",
        default=None,
        help="Comma-separated canonical endpoints (default: hypertension_ocular,diabetes_ocular)",
    )
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--learnable-q", action="store_true", dest="learnable_q")
    parser.add_argument("--calibrate", action="store_true", help="Fit temperature on calibration fold; apply on --split")
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="Refuse calibration leakage and non-direct clinical cross-eval heads",
    )
    parser.add_argument(
        "--clinical-tables",
        action="store_true",
        help="Require alignment=direct for cross-cohort clinical tables",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test", "all", "calibration", "cal"],
        help="Which partition to score (uses split.npz next to checkpoint when present)",
    )
    parser.add_argument("--regenerate-data", action="store_true", help="Overwrite demo npz files")
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument(
        "--out-json",
        default=None,
        help="Write metrics JSON (AUROC/AP/ECE/Brier/Youden/sens@95%spec/DCA curves + calibrated if --calibrate)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    if args.demo and dataset_name is None:
        dataset_name = "demo"
    if dataset_name is None:
        dataset_name = "ukb"

    default_cfg = os.path.join(
        ROOT, "configs", "demo.yaml" if dataset_name == "demo" or args.demo else "default.yaml"
    )
    cfg = load_config(args.config or default_cfg)
    if args.data_dir:
        cfg["data_dir"] = os.path.abspath(args.data_dir)
    split_cfg = cfg.get("split", {})
    val_fraction = (
        args.val_fraction
        if args.val_fraction is not None
        else split_cfg.get("val_fraction", 0.25 if args.demo or dataset_name == "demo" else 0.0)
    )
    split_seed = args.split_seed if args.split_seed is not None else split_cfg.get("seed", 42)
    tcfg = cfg["training"]
    eval_cfg = cfg.get("eval", {})
    do_calibrate = bool(args.calibrate or tcfg.get("calibrate", False) or eval_cfg.get("calibrate", False))

    if args.demo or dataset_name == "demo":
        if args.regenerate_data or not ukb_compressed_ready(cfg["data_dir"]):
            import importlib.util

            gen_path = os.path.join(ROOT, "scripts", "generate_demo_data.py")
            spec = importlib.util.spec_from_file_location("generate_demo_data", gen_path)
            gen_mod = importlib.util.module_from_spec(spec)
            assert spec.loader
            spec.loader.exec_module(gen_mod)
            demo_cfg = cfg.get("demo", {})
            gen_mod.generate(
                cfg["data_dir"],
                n_samples=demo_cfg.get("n_samples", 256),
                seed=demo_cfg.get("seed", 42),
            )
        else:
            print(f"Using existing demo data in {cfg['data_dir']}")

    run_dir = resolve_run_dir(args.ckpt)
    meta = load_run_meta(run_dir)
    diseases = meta.get("diseases")
    if not diseases:
        if args.disease and args.disease != "all":
            diseases = [args.disease]
        elif args.multitask or meta.get("multitask"):
            diseases = None  # fill after dataset load
        else:
            if not args.disease:
                raise ValueError("Pass --disease or a run folder with run_meta.json")
            diseases = [args.disease]
    horizon = int(meta.get("horizon", args.horizon))
    learnable_q = bool(meta.get("learnable_q", args.learnable_q or tcfg.get("learnable_q", False)))
    ensemble = normalize_ensemble(meta.get("ensemble", tcfg.get("ensemble", "released_code")))
    enable_q = bool(meta.get("enable_q", tcfg.get("enable_q", True)))
    quality_router = meta.get(
        "quality_router",
        tcfg.get("quality_router", "monotone" if learnable_q else "fixed"),
    )
    fast_mode = bool(meta.get("fast_mode", tcfg["fast_mode"]))
    paper_mode = bool(args.paper_mode or eval_cfg.get("paper_mode", False))
    clinical_tables = bool(args.clinical_tables or eval_cfg.get("clinical_tables", False) or paper_mode)

    cross = args.test_data_dir is not None
    if cross and not args.test_dataset:
        raise ValueError("--test-dataset is required with --test-data-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ready = ukb_compressed_ready(cfg["data_dir"])
    base_ds = None
    if train_ready:
        base_ds = UKBDatasetFast(
            cfg["data_dir"],
            meta=cfg["clinic_variables"],
            y=horizon,
            disease=[0],
            use_pretrain=cfg["pretrain_features"],
            incident_exclude_prior=dataset_name not in PUBLIC_DATASETS,
        )
        if diseases is None:
            diseases = list(base_ds.disease_names)
        base_ds.set_target(horizon, diseases, incident_exclude_prior=dataset_name not in PUBLIC_DATASETS)
    elif diseases is None:
        raise ValueError("Need run_meta.json diseases or a source cache under --data-dir")

    num_classes = int(meta.get("num_classes", len(diseases)))
    model = get_reti_pioneer(
        fast_mode,
        num_classes=num_classes,
        learnable_q=learnable_q,
        enable_q=enable_q,
        ensemble=ensemble,
        quality_router=quality_router,
    ).to(device)
    ckpt_path = resolve_ckpt_path(args.ckpt)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    mapped_heads: list[MappedHead] = []
    if cross:
        test_dir = os.path.abspath(args.test_data_dir)
        if not ukb_compressed_ready(test_dir):
            raise FileNotFoundError(f"No UKB-style cache in {test_dir}")
        test_name = args.test_dataset
        train_name = str(meta.get("dataset", dataset_name))
        test_ds = UKBDatasetFast(
            test_dir,
            meta=cfg["clinic_variables"],
            y=horizon,
            disease=[0],
            use_pretrain=cfg["pretrain_features"],
            incident_exclude_prior=test_name not in PUBLIC_DATASETS,
        )
        test_ds.set_target(horizon, list(test_ds.disease_names), incident_exclude_prior=False)
        tasks = [x.strip() for x in (args.heads or ",".join(DEFAULT_CROSS_HEADS)).split(",") if x.strip()]
        mapped_heads = resolve_cross_heads(
            list(diseases),
            list(test_ds.disease_names),
            train_name,
            test_name,
            tasks,
            require_clinical=clinical_tables,
            paper_mode=paper_mode,
        )
        if not mapped_heads:
            raise ValueError(
                f"No overlapping heads for {train_name}->{test_name} "
                f"train={diseases} test={test_ds.disease_names} tasks={tasks}"
            )
        if not clinical_tables:
            assert_clinical_alignment(mapped_heads, clinical_tables=False)
        print_mapped_heads(mapped_heads, train_name, test_name)
        eval_ds, split_name = test_ds, "all"
        split_bundle = {
            "train_idx": [],
            "val_idx": [],
            "test_idx": None,
            "calibration_idx": None,
            "patient_ids": None,
            "labels": None,
        }
    else:
        if base_ds is None:
            raise FileNotFoundError(f"No source cache in {cfg['data_dir']}")
        eval_ds, split_name, split_bundle = resolve_eval_split(
            run_dir, base_ds, args.split, val_fraction, split_seed
        )

    # Calibration fold resolution (may nest-split val and shrink eval_ds)
    cal_fit_ids: list[int] | None = None
    t_src: str | None = None
    if do_calibrate and not cross and base_ds is not None:
        cal_fit_ids, t_src = resolve_calibration_fit_ids(
            split_bundle, args.split, paper_mode=paper_mode, split_seed=split_seed
        )
        nested_eval = split_bundle.get("_nested_eval_idx")
        if nested_eval is not None:
            from torch.utils.data import Subset

            eval_ds = Subset(base_ds, nested_eval)
            split_name = "val_nested_eval"
            assert_calibration_disjoint(cal_fit_ids, nested_eval)

    probs, labels, logits = collect_probs(model, eval_ds, device, return_logits=True)
    if mapped_heads:
        m_probs, m_y = slice_mapped(probs, labels, mapped_heads)
        m_logits, _ = slice_mapped(logits, labels, mapped_heads)
        probs, labels, logits = m_probs, m_y, m_logits

    score_names: list[str] | None
    if mapped_heads:
        score_names = [h.task for h in mapped_heads]
    elif num_classes > 1:
        score_names = list(diseases)
    else:
        score_names = [diseases[0]] if diseases else None

    scores = score_predictions(labels, probs, class_names=score_names)
    disease_tag = "multitask" if num_classes > 1 else diseases[0]
    if mapped_heads:
        disease_tag = ",".join(h.task for h in mapped_heads)
    n_eval = len(labels) if labels.ndim == 1 else int(labels.shape[0])
    print(
        f"Disease={disease_tag} horizon={horizon}y split={split_name} "
        f"n={n_eval} "
        f"pos={int(np.asarray(labels).sum())}"
    )
    _print_scores("raw", scores)
    per_head: dict[str, dict[str, float]] = {}
    if mapped_heads and labels.ndim == 2:
        for k, head in enumerate(mapped_heads):
            auc, ap = _binary_scores(labels[:, k], probs[:, k])
            print(f"  {head.task}  AUROC={auc:.4f}  AP={ap:.4f}")
            per_head[head.task] = {"auroc": auc, "ap": ap}
    elif labels.ndim == 2 and score_names:
        for k, name in enumerate(score_names):
            if k >= labels.shape[1]:
                break
            auc, ap = _binary_scores(labels[:, k], probs[:, k])
            print(f"  {name}  AUROC={auc:.4f}  AP={ap:.4f}")
            per_head[name] = {"auroc": auc, "ap": ap}
    print_subgroups(eval_ds, labels, probs)

    cal_scores: dict[str, float] | None = None
    temperature: float | None = None
    intercept: float | None = None
    slope: float | None = None
    dca_curves = per_class_decision_curves(labels, probs, class_names=score_names)
    print(
        "Note: NB@0.10 is illustrative only; prefer per-disease DCA curves "
        "(see out-json dca_curves)."
    )
    if do_calibrate:
        fit_logits, fit_y = logits, labels
        if cross and os.path.isfile(os.path.join(run_dir, "split.npz")) and base_ds is not None:
            val_ds, _, src_bundle = resolve_eval_split(
                run_dir, base_ds, "val", val_fraction, split_seed
            )
            # Prefer source calibration_idx when present
            cal_ids = src_bundle.get("calibration_idx") or src_bundle.get("val_idx")
            if cal_ids:
                from torch.utils.data import Subset

                cal_ds = Subset(base_ds, list(cal_ids))
                _, fit_y, fit_logits = collect_probs(model, cal_ds, device, return_logits=True)
                t_src = "source-calibration" if src_bundle.get("calibration_idx") else "source-val"
            if mapped_heads:
                from reti_pioneer.label_map import take_columns

                idx = [h.train_index for h in mapped_heads]
                fit_logits = take_columns(fit_logits, idx)
                fit_y = take_columns(fit_y, idx)
        elif not cross and cal_fit_ids is not None and base_ds is not None:
            from torch.utils.data import Subset

            cal_ds = Subset(base_ds, cal_fit_ids)
            _, fit_y, fit_logits = collect_probs(model, cal_ds, device, return_logits=True)
            eval_ids = list(range(len(eval_ds)))
            # Compare underlying dataset indices when both are Subsets of the same base.
            if hasattr(eval_ds, "indices"):
                eval_ids = list(eval_ds.indices)
            assert_calibration_disjoint(cal_fit_ids, eval_ids)
        elif not cross:
            # Last resort: previous behavior only when folds unavailable
            if paper_mode:
                raise ValueError("paper_mode requires a disjoint calibration fold")
            t_src = t_src or "eval-split-fallback"
            print(f"WARNING: fitting temperature on evaluation fold ({t_src})")

        temperature = float(fit_temperature(fit_logits, fit_y))
        intercept, slope = fit_calibration_intercept_slope(fit_logits, fit_y)
        cal_probs = apply_temperature(logits, temperature)
        cal_scores = score_predictions(labels, cal_probs, class_names=score_names)
        print(
            f"temperature T={temperature:.4f} fitted on {t_src}; "
            f"intercept={intercept:.4f} slope={slope:.4f}"
        )
        _print_scores("calibrated", cal_scores)
        dca_curves = per_class_decision_curves(labels, cal_probs, class_names=score_names)

    if args.out_json:
        payload = {
            "ckpt": ckpt_path,
            "run_dir": run_dir,
            "dataset": dataset_name,
            "disease": disease_tag,
            "horizon": horizon,
            "split": split_name,
            "n": n_eval,
            "cross_dataset": bool(cross),
            "test_dataset": args.test_dataset if cross else None,
            "calibrate": do_calibrate,
            "temperature": temperature,
            "temperature_fit": t_src,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "raw": scores,
            "calibrated": cal_scores,
            "per_head": per_head or None,
            "dca_curves": dca_curves,
            "nb@0.10_note": (
                "Illustrative single-threshold net benefit only; "
                "prefer per-disease dca_curves for primary narrative."
            ),
            "mapped_heads": (
                [
                    {
                        "task": h.task,
                        "alignment": h.alignment,
                        "kind": h.kind,
                        "clinical_claim_allowed": h.clinical_claim_allowed,
                        "train_name": h.train_name,
                        "test_name": h.test_name,
                    }
                    for h in mapped_heads
                ]
                if mapped_heads
                else None
            ),
            "disclaimer": (
                "Metrics reflect the loaded cache; synthetic features are not for "
                "manuscript AUROC. clinical_claim_allowed=false unless alignment=direct."
            ),
        }
        out_path = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, indent=2)
        print(f"Wrote metrics JSON to {out_path}")


if __name__ == "__main__":
    main()
