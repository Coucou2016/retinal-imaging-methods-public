#!/usr/bin/env python3
"""Two figures that the held-out recalibration result requires.

fig16  the recalibration ladder. Panel (a) is macro per-endpoint ECE on the test
       fold for four post-hoc maps fitted on validation only. Panel (b) is the
       fitted intercept of the per-endpoint Platt map against endpoint
       prevalence, which is the diagnostic that identifies the error as a logit
       offset rather than over-confidence.

fig17  per-endpoint paired differences for the three design comparisons, so the
       macro average can be read against the endpoints that produce it.

Usage
-----
    python scripts/plot_extra_figures.py
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

ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
COLORS = {
    "E0": "#4C6E91", "E1": "#2E7D6F", "E2": "#B07A3C",
    "E3": "#A34E4E", "E4": "#7A5C99", "E5": "#3F6BA8",
}
EP_ORDER = ["Disease_Risk", "DR", "MH", "ODC", "TSLN", "DN", "ARMD", "MYA"]
EP_LABEL = {
    "Disease_Risk": "Any disease", "DR": "Diabetic\nretinopathy", "MH": "Media\nhaze",
    "ODC": "Optic disc\ncupping", "TSLN": "Tessellation", "DN": "Drusen",
    "ARMD": "Macular\ndegeneration", "MYA": "Myopia",
}


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
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / f"{stem}.png")
    fig.savefig(ASSETS / f"{stem}.pdf")
    plt.close(fig)


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def fig_ladder() -> None:
    hold = load("recalibration_holdout.json")
    arms = [a for a in ARMS if a in hold["per_arm"]]

    fig = plt.figure(figsize=(7.1, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    rungs = ["none", "temperature", "platt", "isotonic"]
    names = ["Raw", "Temperature", "Platt", "Isotonic"]
    shades = ["#9DB4C9", "#C9B79D", "#4C6E91", "#7A5C99"]
    x = np.arange(len(arms))
    w = 0.19
    for i, (r, n, c) in enumerate(zip(rungs, names, shades)):
        vals = [hold["per_arm"][a]["summary"][f"macro_ece_{r}"] for a in arms]
        ax.bar(x + (i - 1.5) * w, vals, w, color=c, label=n,
               edgecolor="#333333", linewidth=0.3)
    ax.axhline(0.05, ls=":", lw=0.7, color="#888888")
    ax.text(5.42, 0.053, "0.05", fontsize=6.4, color="#666666", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=8)
    ax.set_ylabel("Macro per-endpoint ECE on test")
    ax.set_ylim(0, 0.175)
    ax.grid(axis="y", ls=":", lw=0.45, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              columnspacing=0.9, handlelength=1.1, handletextpad=0.4)
    ax.set_title("(a) Correction fitted on validation, scored on test", fontsize=8, pad=14)

    ax2 = fig.add_subplot(gs[0, 1])
    marks = {"E2": "^", "E3": "o", "E4": "D", "E5": "s"}
    for arm in ("E2", "E3", "E4", "E5"):
        rows = hold["per_arm"][arm]["per_head"]
        eps = [e for e in EP_ORDER if e in rows]
        xv = [float(np.log(rows[e]["prevalence"] / (1 - rows[e]["prevalence"]))) for e in eps]
        yv = [rows[e]["platt_intercept"] for e in eps]
        corr = float(np.corrcoef(xv, yv)[0, 1])
        ax2.scatter(xv, yv, s=22, marker=marks[arm], facecolor="none",
                    edgecolor=COLORS[arm], linewidth=0.9, label=f"{arm}  r = {corr:.2f}")
        b, a = np.polyfit(xv, yv, 1)
        xs = np.linspace(min(xv) - 0.3, max(xv) + 0.3, 20)
        ax2.plot(xs, a + b * xs, lw=0.8, color=COLORS[arm], alpha=0.65)
    ax2.plot([-3.4, 1.8], [-3.4, 1.8], lw=0.7, ls=":", color="#999999", label="identity")
    ax2.axhline(0.0, lw=0.6, color="#CCCCCC")
    ax2.set_xlabel("logit of endpoint prevalence")
    ax2.set_ylabel("Fitted Platt intercept")
    ax2.set_title("(b) Intercept tracks prevalence", fontsize=8, pad=8)
    ax2.legend(fontsize=6.2, loc="upper left", handletextpad=0.4, borderaxespad=0.2)
    ax2.grid(ls=":", lw=0.45, color="#DDDDDD")
    ax2.set_axisbelow(True)

    save(fig, "fig16_recalibration_ladder")


def fig_endpoint_delta() -> None:
    ext = load("extended.json")
    deltas = ext["per_endpoint"]["delta"]
    pairs = [("E3_minus_E2", "E3 − E2\nmonotone routing"),
             ("E4_minus_E3", "E4 − E3\nconstraint removed"),
             ("E5_minus_E3", "E5 − E3\nbackbone routing")]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ypos, ticks, labels = 0.0, [], []
    for gi, (key, title) in enumerate(pairs):
        rows = deltas.get(key, {})
        if not rows:
            continue
        top = ypos
        for ep in EP_ORDER:
            if ep not in rows:
                continue
            v = rows[ep]
            col = "#2E7D6F" if v["ci_low"] > 0 else ("#A34E4E" if v["ci_high"] < 0 else "#888888")
            ax.plot([v["ci_low"], v["ci_high"]], [ypos, ypos], lw=1.1, color=col,
                    solid_capstyle="butt")
            ax.plot([v["delta"]], [ypos], "o", ms=3.4, color=col, zorder=3)
            ticks.append(ypos)
            labels.append(EP_LABEL[ep].replace("\n", " "))
            ypos -= 1.0
        ax.text(-0.118, top + 1.35, title, fontsize=7.2, color="#222222",
                ha="left", va="bottom", linespacing=1.35)
        ypos -= 2.4
        if gi < len(pairs) - 1:
            ax.axhline(ypos + 1.2, lw=0.5, ls=":", color="#CCCCCC")

    ax.axvline(0.0, lw=0.8, color="#333333", ls="--")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.6)
    ax.set_ylim(ypos + 0.6, 2.1)
    ax.set_xlabel("Difference in AUROC (95% bootstrap interval)")
    ax.set_xlim(-0.125, 0.135)
    ax.grid(axis="x", ls=":", lw=0.45, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig17_endpoint_delta")


def main() -> None:
    style()
    fig_ladder()
    fig_endpoint_delta()
    print("wrote fig16_recalibration_ladder, fig17_endpoint_delta")


if __name__ == "__main__":
    main()
