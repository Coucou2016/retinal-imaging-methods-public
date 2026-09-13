#!/usr/bin/env python
"""Train Reti-Pioneer heads on pre-extracted features (paper workflow)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from ignite.utils import to_onehot

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset.UKBDataset import UKBDatasetFast
from model.RetiPioneer import get_reti_pioneer
from reti_pioneer.config import load_config
from reti_pioneer.constants import DISEASE_NAMES, PUBLIC_DATASETS
from reti_pioneer.data_paths import ukb_compressed_ready
from model.RetiPioneer import normalize_ensemble
from reti_pioneer.split import (
    dataset_label_matrix,
    dataset_labels,
    dataset_patient_ids,
    load_calibration_idx,
    load_split,
    load_test_idx,
    make_subsets,
    patient_level_train_cal_val_indices,
    patient_level_train_cal_val_test_indices,
    patient_level_train_val_indices,
    patient_level_train_val_test_indices,
    save_split,
)
from utils.functional import pb_l_r_m_q_y
from utils.run import single_fastds_run


def prepare_batch(batch, device, non_blocking):
    (l, r), m, (ql, qr), y = batch
    eth = m[:, -1]
    e = to_onehot(eth.long(), 7)
    m = torch.cat([m[:, :-1], e], 1)
    if y.dim() == 1:
        y = y.view(-1, 1)
    batch = (l, r), m, (ql, qr), y
    return pb_l_r_m_q_y(batch, device, non_blocking)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_demo_data(data_dir: str, n_samples: int, seed: int, force: bool) -> None:
    if not force and ukb_compressed_ready(data_dir):
        print(f"Demo data already present in {data_dir} (use --regenerate-data to overwrite)")
        return
    import importlib.util

    gen_path = os.path.join(ROOT, "scripts", "generate_demo_data.py")
    spec = importlib.util.spec_from_file_location("generate_demo_data", gen_path)
    gen_mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(gen_mod)
    gen_mod.generate(data_dir, n_samples=n_samples, seed=seed)


def ensure_public_cache(dataset: str, data_dir: str, n_samples: int, seed: int, force: bool) -> None:
    if not force and ukb_compressed_ready(data_dir):
        print(f"Public cache already present in {data_dir}")
        return
    from dataset.brset import BRSET_LABELS, make_synthetic_brset_records
    from dataset.odir import ODIR_LABELS, make_synthetic_odir_records
    from dataset.rfmid import make_synthetic_rfmid_records
    from scripts.prepare_public_npz import write_public_npz

    if dataset == "odir":
        recs, names = make_synthetic_odir_records(n_samples, seed), list(ODIR_LABELS)
    elif dataset == "brset":
        recs, names = make_synthetic_brset_records(n_samples, seed), list(BRSET_LABELS)
    else:
        recs, names = make_synthetic_rfmid_records(n_samples, seed)
    write_public_npz(
        data_dir,
        recs,
        names,
        seed=seed,
        synthetic_features=True,
        dataset_name=dataset,
        force=True,
    )


def pos_weight_from_dataset(dataset) -> np.ndarray:
    y = dataset_label_matrix(dataset)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = np.ones(y.shape[1], dtype=np.float32)
    nz = pos > 0
    w[nz] = (neg[nz] / np.maximum(pos[nz], 1.0)).astype(np.float32)
    return w


def write_run_meta(tbdir: str, payload: dict) -> None:
    os.makedirs(tbdir, exist_ok=True)
    path = os.path.join(tbdir, "run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Reti-Pioneer disease heads")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--disease", default=None, help="Single disease name or 'all'")
    parser.add_argument("--horizon", type=int, default=None, choices=[0, 5, 10])
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    parser.add_argument("--dataset", default=None, choices=["odir", "brset", "rfmid", "ukb", "demo"])
    parser.add_argument("--data-dir", default=None, help="Override config data_dir")
    parser.add_argument("--multitask", action="store_true", help="Joint K-class head (BCE)")
    parser.add_argument("--learnable-q", action="store_true", dest="learnable_q")
    parser.add_argument(
        "--ensemble",
        default=None,
        choices=["released_code", "published_soft_vote", "mean", "temp_mean", "paper"],
    )
    parser.add_argument(
        "--quality-router",
        default=None,
        choices=["fixed", "free_linear", "monotone"],
        dest="quality_router",
        help="Quality routing: fixed | free_linear (ablation) | monotone (default learnable)",
    )
    parser.add_argument(
        "--synthetic-public",
        action="store_true",
        help="Write a tiny synthetic public-style cache if --dataset is odir|brset|rfmid",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Regenerate demo/public npz even if files already exist",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Held-out validation fraction (default: 0.25 for --demo, 0 otherwise)",
    )
    parser.add_argument("--split-seed", type=int, default=None, help="RNG seed for train/val split")
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
    elif dataset_name in PUBLIC_DATASETS and args.config is None and not args.demo:
        cfg["data_dir"] = os.path.join(ROOT, "data", dataset_name)

    split_cfg = cfg.get("split", {})
    # Public cohorts need a held-out fold even when default.yaml omits split:.
    _default_val = 0.25 if dataset_name in (*PUBLIC_DATASETS, "demo") or args.demo else 0.0
    val_fraction = (
        args.val_fraction
        if args.val_fraction is not None
        else float(split_cfg.get("val_fraction", _default_val))
    )
    split_seed = args.split_seed if args.split_seed is not None else int(split_cfg.get("seed", 42))
    test_fraction = float(split_cfg.get("test_fraction", 0.2 if dataset_name in PUBLIC_DATASETS else 0.0))
    rng_seed = int(split_cfg.get("torch_seed", split_seed))
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    tcfg = cfg["training"]
    learnable_q = bool(args.learnable_q or tcfg.get("learnable_q", False))
    multitask = bool(args.multitask or tcfg.get("multitask", False))
    ensemble = normalize_ensemble(args.ensemble or tcfg.get("ensemble", "released_code"))
    enable_q = bool(tcfg.get("enable_q", True))
    use_pos_weight = bool(tcfg.get("pos_weight", False))
    quality_router = args.quality_router or tcfg.get(
        "quality_router", "monotone" if learnable_q else "fixed"
    )
    cal_fraction = float(split_cfg.get("cal_fraction", 0.1 if val_fraction > 0 else 0.0))
    masked_bce = bool(tcfg.get("masked_bce", True))
    quality_gating = bool(tcfg.get("quality_gating", False))
    lambda_q = float(tcfg.get("lambda_q", 0.0) or 0.0)

    if dataset_name == "demo" or (args.demo and dataset_name not in PUBLIC_DATASETS):
        demo_cfg = cfg.get("demo", {})
        ensure_demo_data(
            cfg["data_dir"],
            n_samples=demo_cfg.get("n_samples", 256),
            seed=demo_cfg.get("seed", 42),
            force=args.regenerate_data,
        )
    if args.demo:
        cfg["training"] = {**cfg["training"], "epochs": 3, "warmup_epochs": 2, "epochs_factor": 1}
        tcfg = cfg["training"]

    if dataset_name in PUBLIC_DATASETS:
        if args.synthetic_public or args.demo or not ukb_compressed_ready(cfg["data_dir"]):
            if not ukb_compressed_ready(cfg["data_dir"]) and not args.synthetic_public and not args.demo:
                raise FileNotFoundError(
                    f"No UKB_*.npz in {cfg['data_dir']}. Run "
                    f"python scripts/prepare_public_npz.py --dataset {dataset_name} "
                    f"--src <csv-or-images> --out {cfg['data_dir']} "
                    "or pass --synthetic-public for a CI cache."
                )
            demo_cfg = cfg.get("demo", {})
            ensure_public_cache(
                dataset_name,
                cfg["data_dir"],
                n_samples=demo_cfg.get("n_samples", 64),
                seed=demo_cfg.get("seed", 42),
                force=args.regenerate_data or args.synthetic_public,
            )

    device = resolve_device(tcfg.get("device", "auto"))
    ftime = datetime.now().strftime("%Y%m%d%H%M%S")
    clinic_variables = cfg["clinic_variables"]
    pretrain = cfg["pretrain_features"]
    horizons = [args.horizon] if args.horizon is not None else list(cfg["longitudinal_years"])
    if dataset_name in PUBLIC_DATASETS:
        if any(h != 0 for h in horizons):
            print("Public caches store prevalence only (y5/y10 copy y0); using horizon=0")
        horizons = [0]

    base_dataset = UKBDatasetFast(
        cfg["data_dir"],
        None,
        clinic_variables,
        disease=[0],
        use_pretrain=pretrain,
        incident_exclude_prior=dataset_name not in PUBLIC_DATASETS,
    )

    if dataset_name in PUBLIC_DATASETS:
        available = list(base_dataset.disease_names)
        if multitask:
            diseases = available
        elif args.disease and args.disease != "all" and args.disease in available:
            diseases = [args.disease]
        else:
            diseases = [available[0]]
    elif multitask:
        if args.disease and args.disease != "all":
            diseases = [args.disease]
        else:
            diseases = list(base_dataset.disease_names)
    else:
        diseases = [args.disease] if args.disease and args.disease != "all" else list(cfg["diseases"])

    jobs: list[tuple[list[str], int]]
    if multitask:
        jobs = [(diseases, y) for y in horizons]
    else:
        jobs = [([d], y) for d in diseases for y in horizons]

    for target_diseases, y in jobs:
        for d in target_diseases:
            if d not in base_dataset.disease_names and d not in DISEASE_NAMES:
                raise ValueError(f"Unknown disease {d}; expected one of {base_dataset.disease_names}")
        num_classes = len(target_diseases)
        tag = "multitask" if num_classes > 1 else target_diseases[0]
        print(
            f"Training {tag} K={num_classes} (y={y} years) learnable_q={learnable_q} "
            f"quality_router={quality_router} ensemble={ensemble} on {device}"
        )
        model = get_reti_pioneer(
            tcfg["fast_mode"],
            num_classes=num_classes,
            learnable_q=learnable_q,
            enable_q=enable_q,
            ensemble=ensemble,
            quality_router=quality_router,
            quality_gating=quality_gating,
            quality_aux=lambda_q > 0,
            lambda_q=lambda_q,
        )
        model = model.to(device)
        base_dataset.set_target(y, target_diseases, incident_exclude_prior=dataset_name not in PUBLIC_DATASETS)

        tbdir = os.path.join(cfg["ckpt_dir"], ftime, tag, f"y{y}")
        tds = base_dataset
        vds = None
        train_idx: list[int] = []
        val_idx: list[int] = []
        test_idx: list[int] | None = None
        calibration_idx: list[int] | None = None
        frozen_split = os.path.join(cfg["data_dir"], "split.npz")
        # Prefer cache-level frozen split (official RFMiD / prepare_public_npz).
        use_split = val_fraction > 0 or os.path.isfile(frozen_split)
        if use_split:
            labels = dataset_labels(base_dataset)
            pids = dataset_patient_ids(base_dataset)
            n_patients = len(np.unique(pids))
            if os.path.isfile(frozen_split):
                train_idx, val_idx = load_split(frozen_split)
                test_idx = load_test_idx(frozen_split)
                calibration_idx = load_calibration_idx(frozen_split)
                print(f"  using frozen split from {frozen_split}")
                # If frozen split lacks calibration, carve from val when calibrating.
                if calibration_idx is None and cal_fraction > 0 and len(val_idx) >= 4:
                    from reti_pioneer.split import nested_calibration_from_val

                    calibration_idx, val_idx = nested_calibration_from_val(
                        pids, val_idx, labels, cal_fraction, split_seed + 11
                    )
                    print(
                        f"  nested calibration from frozen val: "
                        f"cal={len(calibration_idx)} val={len(val_idx)}"
                    )
            elif (
                test_fraction > 0
                and dataset_name in PUBLIC_DATASETS
                and n_patients >= 8
                and cal_fraction > 0
            ):
                train_idx, calibration_idx, val_idx, test_idx = (
                    patient_level_train_cal_val_test_indices(
                        pids,
                        labels,
                        val_fraction,
                        cal_fraction,
                        test_fraction,
                        split_seed,
                    )
                )
            elif test_fraction > 0 and dataset_name in PUBLIC_DATASETS and n_patients >= 6:
                train_idx, val_idx, test_idx = patient_level_train_val_test_indices(
                    pids, labels, val_fraction, test_fraction, split_seed
                )
            elif cal_fraction > 0 and n_patients >= 6:
                train_idx, calibration_idx, val_idx = patient_level_train_cal_val_indices(
                    pids, labels, max(val_fraction, 0.05), cal_fraction, split_seed
                )
            else:
                train_idx, val_idx = patient_level_train_val_indices(
                    pids, labels, max(val_fraction, 0.05), split_seed
                )
            tds, vds = make_subsets(base_dataset, train_idx, val_idx)
            os.makedirs(tbdir, exist_ok=True)
            save_split(
                os.path.join(tbdir, "split.npz"),
                train_idx,
                val_idx,
                split_seed,
                val_fraction if val_fraction > 0 else 0.25,
                test_idx=test_idx,
                patient_ids=pids,
                calibration_idx=calibration_idx,
            )
            extra = f" test={len(test_idx)}" if test_idx else ""
            cal_extra = f" cal={len(calibration_idx)}" if calibration_idx else ""
            print(
                f"  train={len(train_idx)} val={len(val_idx)}{cal_extra}{extra} "
                f"(val positives={int(labels[val_idx].sum())} patients={n_patients})"
            )

        pos_weight = pos_weight_from_dataset(tds) if use_pos_weight else None
        write_run_meta(
            tbdir,
            {
                "num_classes": num_classes,
                "diseases": target_diseases,
                "horizon": y,
                "learnable_q": learnable_q,
                "enable_q": enable_q,
                "quality_router": quality_router,
                "ensemble": ensemble,
                "multitask": num_classes > 1,
                "masked_bce": masked_bce,
                "quality_gating": quality_gating,
                "lambda_q": lambda_q,
                "dataset": dataset_name,
                "fast_mode": tcfg["fast_mode"],
            },
        )
        single_fastds_run(
            model,
            lr=tcfg["lr"],
            epochs=tcfg["epochs"],
            epochs_factor=tcfg["epochs_factor"],
            warmup_lr=tcfg["warmup_lr"],
            warmup_epochs=tcfg["warmup_epochs"],
            tds=tds,
            vds=vds,
            xdss=[],
            pb=prepare_batch,
            bs=min(tcfg["batch_size"], len(tds)),
            device=device,
            tbdir=tbdir,
            save=True,
            save_prob=True,
            balance_sampler=tcfg.get("balance_sampler", False),
            weight_decay=tcfg.get("weight_decay", 0.01),
            pos_weight=pos_weight,
            simple_metrics=num_classes > 1,
            masked_bce=masked_bce,
            lambda_q=lambda_q,
        )


if __name__ == "__main__":
    main()
