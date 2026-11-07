#!/usr/bin/env python3
"""Figures for the recalibrated analysis.

The first draft of the manuscript quoted a 95%-specificity sensitivity that had
been chosen on the test split. Once the threshold is frozen on validation, the
ranking of arms changes, and temperature scaling turns out not to fix the
calibration error at all. Those two facts need their own figures.

Produces
--------
fig13_headline_calibrated   macro AUROC and mean-head ECE, before and after temperature
fig14_threshold_transfer    sensitivity at 95% specificity: threshold from validation
                            (deployable) versus matched on test (upper bound)
fig15_score_shift           DR score distributions on validation and test, per arm
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
ANALYSIS = ROOT
STATS = ROOT
ASSETS = ROOT

ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
COLORS = {
    "E0": "#4C6E91",
    "E1": "#2E7D6F",
    "E2": "#B07A3C",
    "E3": "#A34E4E",
    "E4": "#7A5C99",
    "E5": "#3F6BA8",
}
TICK = {
    "E0": "E0\nfixed\nsingle",
    "E1": "E1\nmonotone\nsingle",
    "E2": "E2\nfixed\nmultitask",
    "E3": "E3\nmonotone\nmultitask",
    "E4": "E4\nfree\nmultitask",
    "E5": "E5\nrouting\nmultitask",
}
EPS = 1e-6


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "SimHei", "DejaVu Serif"],
        "font.size": 9,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })


def save(fig, stem: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / f"{stem}.png")
    fig.savefig(ASSETS / f"{stem}.pdf")
    plt.close(fig)


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fig_calibrated() -> None:
    """Macro AUROC beside mean-head ECE, raw vs temperature scaled."""
    cal = load("calibration_fit.json")
    head = load("headline_calibrated.json")
    arms = [a for a in ARMS if a in head]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    x = np.arange(len(arms))
    w = 0.36

    a0 = axes[0]
    a0.bar(x - w / 2, [head[a]["macro_auroc_raw"] for a in arms], w,
           color="#4C6E91", label="Raw sigmoid score")
    a0.bar(x + w / 2, [head[a]["macro_auroc_cal"] for a in arms], w,
           color="#9DB4C9", label="After temperature scaling")
    a0.set_ylim(0.80, 1.0)
    a0.set_ylabel("Macro AUROC")
    a0.set_title("Discrimination is unchanged", fontsize=8.5)
    a0.legend(fontsize=6.8, loc="lower right")

    a1 = axes[1]
    raw = [head[a]["mean_head_ece_raw"] for a in arms]
    post = [head[a]["mean_head_ece_cal"] for a in arms]
    a1.bar(x - w / 2, raw, w, color="#B07A3C", label="Raw sigmoid score")
    a1.bar(x + w / 2, post, w, color="#E0C39A", label="After temperature scaling")
    for xi, r, p in zip(x, raw, post):
        a1.annotate(f"{r:.3f}", xy=(xi - w / 2, r), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5.8, rotation=90)
    a1.set_ylabel("Mean per-endpoint ECE")
    a1.set_title("Calibration error does not fall", fontsize=8.5)
    a1.legend(fontsize=6.8, loc="upper left")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([TICK[a] for a in arms], fontsize=6.2)
        ax.grid(axis="y", ls=":", lw=0.5, color="#CCCCCC")
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig13_headline_calibrated")
    print("  fig13:", {a: (cal[a]["temperature"], head[a]["mean_head_ece_raw"], head[a]["mean_head_ece_cal"]) for a in arms})


def fig_threshold_transfer() -> None:
    """Frozen-threshold sensitivity next to the matched upper bound."""
    op = load("operating_point_analysis.json")["per_arm"]
    arms = [a for a in ARMS if a in op]

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    x = np.arange(len(arms))
    w = 0.36
    frozen = [op[a]["test_sensitivity_frozen"] for a in arms]
    matched = [op[a]["test_sensitivity_matched"] for a in arms]
    ferr = np.array([
        [op[a]["test_sensitivity_frozen"] - op[a]["test_sensitivity_frozen_ci"][0],
         op[a]["test_sensitivity_frozen_ci"][1] - op[a]["test_sensitivity_frozen"]]
        for a in arms
    ]).T

    ax.bar(x - w / 2, frozen, w, yerr=ferr, capsize=2.2, color="#3F6BA8",
           label="Threshold from validation (deployable)",
           error_kw={"lw": 0.7, "ecolor": "#333333"})
    ax.bar(x + w / 2, matched, w, color="#C9D8E6",
           label="Threshold matched on test (upper bound)")
    for xi, f, m in zip(x, frozen, matched):
        ax.annotate(f"{f:.3f}", xy=(xi - w / 2, f), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.0)
        ax.annotate(f"{m:.3f}", xy=(xi + w / 2, m), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.0, color="#31506E")

    ax.set_xticks(x)
    ax.set_xticklabels([TICK[a] for a in arms], fontsize=6.6)
    ax.set_ylabel("DR sensitivity at 95% specificity")
    ax.set_ylim(0.35, 0.92)
    ax.grid(axis="y", ls=":", lw=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig14_threshold_transfer")
    print("  fig14:", list(zip(arms, frozen, matched)))


def fig_score_shift() -> None:
    """DR score density on validation and on test, per arm."""
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))
    arms = [a for a in ARMS if a in manifest]

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.1), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 41)
    handles = None
    for ax, arm in zip(axes.ravel(), arms):
        e = manifest[arm]
        run_dir = Path(e["run_dir"])
        epoch = Path(e["ckpt"]).stem
        v = np.load(run_dir / "ckpt" / f"py_{epoch}_val.npz", allow_pickle=True)
        vp, vy = v["p"], v["y"]
        if vp.ndim == 1:
            vp, vy = vp.reshape(-1, 1), vy.reshape(-1, 1)
        te = np.load(ANALYSIS / f"arm_{arm}.npz", allow_pickle=True)
        names = [str(x) for x in te["names"]]
        k = names.index("DR")

        vneg = vp[vy[:, k] == 0, k]
        tneg = te["probs"][te["labels"][:, k] == 0, k]
        ax.hist(vneg, bins=bins, density=True, histtype="stepfilled", lw=0.9,
                alpha=0.45, color="#4C6E91", label="Validation negatives")
        ax.hist(tneg, bins=bins, density=True, histtype="step", lw=1.2, ls="--",
                color="#B03A2E", label="Test negatives")
        ax.set_title(f"{arm}  (val n={len(vneg)}, test n={len(tneg)})", fontsize=7.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.grid(ls=":", lw=0.4, color="#DDDDDD")
        if handles is None:
            handles = ax.get_legend_handles_labels()

    for ax in axes[-1]:
        ax.set_xlabel("Predicted DR probability on negative images", fontsize=7.4)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density", fontsize=7.4)
    if handles:
        fig.legend(*handles, fontsize=7.0, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.tight_layout()
    save(fig, "fig15_score_shift")
    print("  fig15: done")


def main() -> None:
    style()
    fig_calibrated()
    fig_threshold_transfer()
    fig_score_shift()
    print("figures written to", ASSETS)


if __name__ == "__main__":
    sys.exit(main())
