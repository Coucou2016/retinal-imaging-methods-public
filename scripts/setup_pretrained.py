#!/usr/bin/env python
"""Check or prepare pretrained assets for Reti-Pioneer inference (optional)."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from reti_pioneer.config import load_config

EYEQ_URL = "https://github.com/HzFu/EyeQ"
RETF_HF_REPO = "YukunZhou/RETFound_mae_natureCFP"
VIM_DEFAULT = "vim_s_midclstok_ft_81p6acc.pth"


def _status(path: str, label: str) -> str:
    if os.path.isfile(path):
        return f"[ok] {label}: {path}"
    return f"[missing] {label}: {path}"


def try_retf_cache(pretrained_dir: str) -> str:
    try:
        import huggingface_hub as hf

        path = hf.try_to_load_from_cache(RETF_HF_REPO, "RETFound_mae_natureCFP.pth")
        if path and os.path.isfile(path):
            return f"[ok] RETFound (HF cache): {path}"
    except Exception as exc:  # pragma: no cover - network optional
        return f"[skip] RETFound HF check failed: {exc}"
    return (
        f"[info] RETFound will download on first use from Hugging Face ({RETF_HF_REPO}). "
        "Set HF_HOME or run feature extraction once online."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify pretrained weight paths")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    pre = cfg["pretrained_dir"]
    os.makedirs(os.path.join(pre, "fundus"), exist_ok=True)

    eyeq = os.environ.get(
        "RETI_PIONEER_EYEQ_WEIGHTS",
        os.path.join(pre, "fundus", "DenseNet121_v3_v1.tar"),
    )
    vim = os.environ.get(
        "RETI_PIONEER_VIM_WEIGHTS",
        os.path.join(pre, VIM_DEFAULT),
    )

    lines = [
        f"Pretrained directory: {pre}",
        _status(eyeq, "EyeQ DenseNet121"),
        f"       Download manually from {EYEQ_URL} and place at the path above,",
        f"       or set RETI_PIONEER_EYEQ_WEIGHTS.",
        _status(vim, "Vision Mamba-S"),
        f"       Upstream Reti-Pioneer README; set RETI_PIONEER_VIM_WEIGHTS if elsewhere.",
        try_retf_cache(pre),
        "[info] Swin V2-B uses torchvision DEFAULT weights (auto on first load).",
        "[info] TorchScript disease heads: place T2D_y0.pt etc. under models/ (upstream release).",
    ]
    print("\n".join(lines))

    missing = sum(1 for line in lines if line.startswith("[missing]"))
    if missing:
        print(
            "\nFast-mode training on UKB_*.npz features does not require the weights above. "
            "They are needed for scripts/extract_features.py and inference.py."
        )
        sys.exit(1)
    print("\nAll local weight files found.")


if __name__ == "__main__":
    main()
