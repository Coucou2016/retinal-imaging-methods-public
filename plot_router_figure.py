#!/usr/bin/env python3
"""Figure for the router intervention.

Two panels. Panel (a) shows how far each arm's learned router sits from the fixed
Reti-Pioneer weights, in absolute terms, per quality level. Panel (b) shows what
happens to test macro AUROC when the router is overwritten with those fixed
weights at inference time, so a reader can see that the router contributes
essentially nothing on its own.

Usage
-----
    python scripts/plot_router_figure.py
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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
STATS = ROOT
ASSETS = ROOT

ARMS = ["E1", "E3", "E4", "E5"]
COLORS = {"E1": "#2E7D6F", "E3": "#A34E4E", "E4": "#7A5C99", "E5": "#3F6BA8"}
LEVELS = ["bad", "usable", "good"]
CN_LEVEL = ["Bad", "Usable", "Good"]


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })


def save(fig, stem: str) -> None:
    fig.savefig(ASSETS / f"{stem}.png")
    fig.savefig(ASSETS / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    style()
    coef = json.loads((ROOT / "router_coefficients.json").read_text(encoding="utf-8"))
    interv = json.loads((ROOT / "router_intervention.json").read_text(encoding="utf-8"))
    arms = [a for a in ARMS if a in coef and a in interv]

    fig = plt.figure(figsize=(7.0, 2.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(LEVELS))
    w = 0.8 / max(1, len(arms))
    for j, arm in enumerate(arms):
        dev = np.abs(np.array(coef[arm]["deviation_from_baseline"]))
        pos = x - 0.4 + w * (j + 0.5)
        ax.bar(pos, dev, w * 0.88, color=COLORS[arm], label=arm,
               edgecolor="#333333", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 5e-2)
    ax.set_xticks(x)
    ax.set_xticklabels(CN_LEVEL)
    ax.set_ylabel("|learned weight − fixed weight|")
    ax.grid(axis="y", ls=":", lw=0.45, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              columnspacing=0.8, handlelength=1.0, handletextpad=0.35)
    ax.set_title("(a) Distance from the released-code weights", fontsize=8, pad=14)

    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(arms))[::-1]
    shift = [interv[a]["max_abs_prob_change"] for a in arms]
    ax2.barh(y, shift, height=0.52, color=[COLORS[a] for a in arms],
             edgecolor="#333333", linewidth=0.3)
    for yi, arm in zip(y, arms):
        e = interv[arm]
        ax2.annotate(
            f"max|Δp| {e['max_abs_prob_change']:.3f}\nΔAUROC {e['delta']:+.1e}",
            xy=(e["max_abs_prob_change"], yi), xytext=(6, 0),
            textcoords="offset points", va="center", fontsize=6.2, color="#333333",
            linespacing=1.3,
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{a}\n(router forced)" for a in arms], fontsize=7.4)
    ax2.set_xlim(0, 0.128)
    ax2.set_ylim(-0.62, len(arms) - 0.38)
    ax2.set_xlabel("Largest per-image probability change")
    ax2.grid(axis="x", ls=":", lw=0.45, color="#DDDDDD")
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("(b) Effect of overwriting the router", fontsize=8, pad=8)

    save(fig, "fig18_router_intervention")
    print("wrote fig18_router_intervention")
    for a in arms:
        print(f"  {a}: max|Δw| {coef[a]['max_abs_deviation']:.5f}  ΔAUROC {interv[a]['delta']:+.2e}")


if __name__ == "__main__":
    main()
