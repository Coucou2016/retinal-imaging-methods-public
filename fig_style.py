#!/usr/bin/env python3
"""Shared figure style: SciencePlots base + Times New Roman, print-ready sizes.

Every manuscript figure imports :func:`apply` so that axis type, tick direction,
minimum line width and font family stay identical across panels.  The style is
deliberately restrained: one accent colour per experimental arm, thin rules, no
chart furniture beyond what the caption explains.
"""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import matplotlib as mpl
import matplotlib.pyplot as plt

# Colour-blind-safe arm palette, used identically in every figure.
ARM_COLORS = {
    "E0": "#3B5B7A",
    "E1": "#2F6F63",
    "E2": "#A9743A",
    "E3": "#9C4A4A",
    "E4": "#6E5A8C",
    "E5": "#2F5FA0",
}
ACCENT = "#9C4A4A"
NEUTRAL = "#7A7A7A"
GRID = "#D8D8D8"

APPLIED = False


def apply(base_size: float = 8.5, cjk: bool = False) -> None:
    """Activate the shared style. Safe to call repeatedly."""
    global APPLIED
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except Exception:  # pragma: no cover - fallback when scienceplots is absent
        mpl.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "xtick.direction": "in",
                "ytick.direction": "in",
            }
        )

    serif = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
    if cjk:
        serif += ["SimSun", "SimHei", "Microsoft YaHei", "Noto Serif CJK SC"]

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": base_size,
            "axes.labelsize": base_size + 0.5,
            "axes.titlesize": base_size + 1.0,
            "xtick.labelsize": base_size - 0.5,
            "ytick.labelsize": base_size - 0.5,
            "legend.fontsize": base_size - 1.0,
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "grid.linewidth": 0.45,
            "grid.linestyle": ":",
            "grid.color": GRID,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.2,
            "patch.linewidth": 0.4,
            "savefig.dpi": 600,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.axisbelow": True,
        }
    )
    APPLIED = True


def grid(axis: str = "y") -> None:
    plt.gca().grid(True, axis=axis)
    plt.gca().set_axisbelow(True)
