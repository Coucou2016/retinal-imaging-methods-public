"""Load YAML/JSON training configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
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
    return cfg
