"""Load and strictly validate YAML/JSON training configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

# Injected / computed keys (not required in YAML).
_RUNTIME_KEYS = frozenset({"project_root"})

_TOP_LEVEL_KEYS = frozenset(
    {
        "data_dir",
        "pretrained_dir",
        "ckpt_dir",
        "clinic_variables",
        "diseases",
        "pretrain_features",
        "training",
        "longitudinal_years",
        "inference",
        "split",
        "demo",
        "eval",
    }
)

_TRAINING_KEYS = frozenset(
    {
        "lr",
        "weight_decay",
        "epochs",
        "epochs_factor",
        "warmup_lr",
        "warmup_epochs",
        "batch_size",
        "fast_mode",
        "balance_sampler",
        "device",
        "learnable_q",
        "multitask",
        "pos_weight",
        "enable_q",
        "ensemble",
        "quality_router",
        "quality_gating",
        "lambda_q",
        "masked_bce",
        "calibrate",
    }
)

_SPLIT_KEYS = frozenset(
    {
        "val_fraction",
        "cal_fraction",
        "test_fraction",
        "seed",
        "torch_seed",
    }
)

_EVAL_KEYS = frozenset(
    {
        "calibrate",
        "decision_threshold",
        "paper_mode",
        "clinical_tables",
        "bootstrap",
    }
)

_DEMO_KEYS = frozenset({"n_samples", "seed"})

_INFERENCE_KEYS = frozenset({"image_size", "ukb_center_crop", "thresholds"})

_VALID_ENSEMBLES = frozenset(
    {"released_code", "published_soft_vote", "mean", "temp_mean", "paper"}
)
_VALID_ROUTERS = frozenset({"fixed", "free_linear", "monotone"})


def _forbid_unknown(section: str, mapping: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) under '{section}': {unknown}. "
            f"Allowed: {sorted(allowed)}"
        )


def validate_config(cfg: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """Fail loud on unknown keys and invalid enum fields.

    Required top-level blocks for model-building configs: ``training``,
    ``clinic_variables``, ``pretrain_features``, ``diseases``.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a mapping; got {type(cfg)!r}")

    if strict:
        _forbid_unknown(
            "top-level",
            {k: v for k, v in cfg.items() if k not in _RUNTIME_KEYS},
            _TOP_LEVEL_KEYS,
        )

    for req in ("training", "clinic_variables", "pretrain_features", "diseases"):
        if req not in cfg:
            raise ValueError(f"Config missing required key {req!r}")

    tcfg = cfg["training"]
    if not isinstance(tcfg, dict):
        raise TypeError("'training' must be a mapping")
    if strict:
        _forbid_unknown("training", tcfg, _TRAINING_KEYS)

    ensemble = str(tcfg.get("ensemble", "released_code"))
    if ensemble not in _VALID_ENSEMBLES:
        raise ValueError(
            f"training.ensemble must be one of {sorted(_VALID_ENSEMBLES)}; got {ensemble!r}"
        )
    router = str(tcfg.get("quality_router", "fixed"))
    if router not in _VALID_ROUTERS:
        raise ValueError(
            f"training.quality_router must be one of {sorted(_VALID_ROUTERS)}; got {router!r}"
        )
    for flag in ("learnable_q", "multitask", "enable_q", "fast_mode"):
        if flag in tcfg and not isinstance(tcfg[flag], (bool, int)):
            raise TypeError(f"training.{flag} must be bool-like")

    if "split" in cfg and cfg["split"] is not None:
        if not isinstance(cfg["split"], dict):
            raise TypeError("'split' must be a mapping")
        if strict:
            _forbid_unknown("split", cfg["split"], _SPLIT_KEYS)

    if "eval" in cfg and cfg["eval"] is not None:
        if not isinstance(cfg["eval"], dict):
            raise TypeError("'eval' must be a mapping")
        if strict:
            _forbid_unknown("eval", cfg["eval"], _EVAL_KEYS)

    if "demo" in cfg and cfg["demo"] is not None:
        if not isinstance(cfg["demo"], dict):
            raise TypeError("'demo' must be a mapping")
        if strict:
            _forbid_unknown("demo", cfg["demo"], _DEMO_KEYS)

    if "inference" in cfg and cfg["inference"] is not None:
        if not isinstance(cfg["inference"], dict):
            raise TypeError("'inference' must be a mapping")
        if strict:
            _forbid_unknown("inference", cfg["inference"], _INFERENCE_KEYS)

    return cfg


def build_model_from_config(cfg: dict[str, Any], *, num_classes: int | None = None):
    """Instantiate ``get_reti_pioneer`` from a validated config (fast_mode Identity path)."""
    from model.RetiPioneer import get_reti_pioneer

    validate_config(cfg)
    tcfg = cfg["training"]
    learnable_q = bool(tcfg.get("learnable_q", False))
    quality_router = tcfg.get("quality_router", "monotone" if learnable_q else "fixed")
    k = num_classes
    if k is None:
        k = len(cfg["diseases"]) if bool(tcfg.get("multitask", False)) else 1
    return get_reti_pioneer(
        fast=bool(tcfg.get("fast_mode", True)),
        num_classes=int(k),
        learnable_q=learnable_q,
        enable_q=bool(tcfg.get("enable_q", True)),
        ensemble=str(tcfg.get("ensemble", "released_code")),
        quality_router=str(quality_router),
        quality_gating=bool(tcfg.get("quality_gating", False)),
        quality_aux=float(tcfg.get("lambda_q", 0.0) or 0.0) > 0,
        lambda_q=float(tcfg.get("lambda_q", 0.0) or 0.0),
    )


def load_config(path: str | Path | None = None, *, strict: bool = True) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {cfg_path}")
    cfg["project_root"] = str(PROJECT_ROOT)
    cfg["data_dir"] = os.path.expandvars(
        os.path.expanduser(cfg.get("data_dir", "data/UKBCompressed"))
    )
    if not os.path.isabs(cfg["data_dir"]):
        cfg["data_dir"] = str(PROJECT_ROOT / cfg["data_dir"])
    cfg["pretrained_dir"] = os.path.expandvars(
        os.path.expanduser(
            os.environ.get(
                "RETI_PIONEER_PRETRAINED_DIR",
                cfg.get("pretrained_dir", "pretrained"),
            )
        )
    )
    if not os.path.isabs(cfg["pretrained_dir"]):
        cfg["pretrained_dir"] = str(PROJECT_ROOT / cfg["pretrained_dir"])
    ckpt_dir = cfg.get("ckpt_dir", "ckpt")
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = str(PROJECT_ROOT / ckpt_dir)
    cfg["ckpt_dir"] = ckpt_dir
    return validate_config(cfg, strict=strict)
