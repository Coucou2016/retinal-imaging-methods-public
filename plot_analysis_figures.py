#!/usr/bin/env python3
"""Figures built from the cached per-sample predictions.

Two figures here are new relative to the earlier draft:

``fig10_paired_delta``  forest plot of the paired ΔAUROC comparisons, which
    shows whether each design change separates from its own baseline.
``fig11_quality_strata``  AUROC by soft-quality stratum, per arm, so the reader
    can see whether an apparent gain survives inside one quality band.

Both read only ``results/real_rfmid/analysis/stats/*.json``.
"""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
STATS = ROOT
ASSETS = ROOT

LABELS = {
    "E0": "E0 fixed weights, single task",
    "E1": "E1 monotone weights, single task",
    "E2": "E2 fixed weights, masked multitask",
    "E3": "E3 monotone weights, masked multitask",
    "E4": "E4 free-linear weights, masked multitask",
    "E5": "E5 E3 plus backbone routing",
}
SHORT = {
    "E0": "E0 fixed / single",
    "E1": "E1 monotone / single",
    "E2": "E2 fixed / multitask",
    "E3": "E3 monotone / multitask",
    "E4": "E4 free-linear / multitask",
    "E5": "E5 backbone routing",
}
COLORS = {
    "E0": "#4C6E91",
    "E1": "#2E7D6F",
    "E2": "#B07A3C",
    "E3": "#A34E4E",
    "E4": "#7A5C99",
    "E5": "#3F6BA8",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )


def paired_forest(out: Path) -> Path:
    data = json.loads((ROOT / "paired.json").read_text(encoding="utf-8"))
    order = [
        ("E1_minus_E0", "DR, single task:\nmonotone vs fixed"),
        ("E3_minus_E2", "Macro, multitask:\nmonotone vs fixed"),
        ("E4_minus_E3", "Macro, multitask:\nfree-linear vs monotone"),
        ("E5_minus_E3", "Macro, multitask:\nbackbone routing added"),
    ]
    order = [(k, lab) for k, lab in order if k in data]

    fig, ax = plt.subplots(figsize=(5.2, 0.85 * len(order) + 1.0))
    for i, (key, label) in enumerate(order):
        entry = data[key]
        sub = entry.get("DR") if "DR" in entry and key in ("E1_minus_E0",) else entry.get("macro")
        d, lo, hi = sub["delta"], sub["ci_low"], sub["ci_high"]
        y = len(order) - 1 - i
        color = "#2E7D6F" if (lo > 0 or hi < 0) else "#8C8C8C"
        ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="butt", zorder=2)
        ax.plot(d, y, "o", color=color, ms=6, zorder=3)
        ax.annotate(
            f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]",
            xy=(hi, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color="#333333",
        )

    ax.axvline(0, color="#B03A2E", lw=0.9, ls="--", zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([lab for _, lab in order][::-1], fontsize=8)
    ax.set_xlabel("Paired difference in AUROC (95% bootstrap interval)")
    ax.set_xlim(-0.03, 0.055)
    ax.grid(axis="x", ls=":", lw=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def quality_strata(out: Path) -> Path:
    data = json.loads((ROOT / "quality_strata.json").read_text(encoding="utf-8"))
    arms = [a for a in ("E0", "E1", "E2", "E3", "E4", "E5") if a in data]
    bands = [("usable", "Usable"), ("bad", "Bad")]

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    width = 0.8 / max(1, len(arms))
    x = np.arange(len(bands))
    for j, arm in enumerate(arms):
        vals = [data[arm].get(key, {}).get("macro_auroc", np.nan) for key, _ in bands]
        pos = x - 0.4 + width * (j + 0.5)
        bars = ax.bar(
            pos, vals, width=width * 0.9, color=COLORS[arm], label=SHORT[arm],
            edgecolor="white", lw=0.5,
        )
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(
                    f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5.6, color="#333333",
                )

    ns = [data[arms[0]].get(key, {}).get("n", 0) for key, _ in bands]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n = {n})" for (_, lab), n in zip(bands, ns)])
    ax.set_ylabel("Macro AUROC within stratum")
    ax.set_ylim(0.80, 1.0)
    ax.grid(axis="y", ls=":", lw=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right", ncol=2)
    ax.set_xlabel("Soft-quality stratum (argmax of the fused quality vector)", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def main() -> None:
    style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    p1 = paired_forest(ASSETS / "fig10_paired_delta")
    print("wrote", p1)
    if (ROOT / "quality_strata.json").is_file():
        p2 = quality_strata(ASSETS / "fig11_quality_strata")
        print("wrote", p2)


if __name__ == "__main__":
    sys.exit(main())
