#!/usr/bin/env python3
"""Rebuild every manuscript figure from the selected-epoch analysis.

This is the single producer for ``docs/paper_assets/fig*.png``.  Each panel reads
only ``results/real_rfmid/analysis_sel`` (predictions, statistics) and
``data/rfmid`` (quality probabilities, split indices, image paths).  Nothing is
computed from labels at plot time, so a figure cannot silently disagree with a
table: the numbers are the same JSON that ``check_manuscript_numbers.py``
validates.

Earlier versions of three figures read ``analysis/stats`` (checkpoints selected by
filename sort, i.e. epoch 9) while the tables read ``analysis_sel/stats`` (epochs
selected on validation).  That mismatch made Figure 5 and Figure 15 contradict
Tables 4 and 15.  They are rebuilt here from the same folder as the tables.

Usage
-----
    python scripts/rebuild_paper_figures.py [--only fig5 fig15 ...]
"""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fig_style  # noqa: E402
from fig_style import ACCENT, ARM_COLORS, GRID, NEUTRAL  # noqa: E402

ANALYSIS = ROOT
STATS = ROOT
DATA = ROOT / "data" / "rfmid"
ASSETS = ROOT

ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
EP_ORDER = ["Disease_Risk", "DR", "MH", "ODC", "TSLN", "DN", "ARMD", "MYA"]
EP_LABEL = {
    "Disease_Risk": "Any disease",
    "DR": "Diabetic\nretinopathy",
    "MH": "Media\nhaze",
    "ODC": "Optic disc\ncupping",
    "TSLN": "Tessellation",
    "DN": "Drusen",
    "ARMD": "Macular\ndegeneration",
    "MYA": "Myopia",
}
EP_SHORT = {
    "Disease_Risk": "Any",
    "DR": "DR",
    "MH": "MH",
    "ODC": "ODC",
    "TSLN": "TSLN",
    "DN": "DN",
    "ARMD": "ARMD",
    "MYA": "MYA",
}
TICK = {
    "E0": "E0\nfixed\nsingle",
    "E1": "E1\nmonotone\nsingle",
    "E2": "E2\nfixed\nmulti",
    "E3": "E3\nmonotone\nmulti",
    "E4": "E4\nfree\nmulti",
    "E5": "E5\nbackbone\nrouting",
}
FULL_W = 6.1  # inches, fits a 15.6 cm text column


def L(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def save(fig, stem: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    png = ASSETS / f"{stem}.png"
    fig.savefig(png)
    fig.savefig(ASSETS / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {png.name}")
    return png


def load_arm(arm: str) -> dict:
    d = np.load(ANALYSIS / f"arm_{arm}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


# --------------------------------------------------------------------- fig1
def fig1_architecture() -> Path:
    """Method flow. Names the surrogates that were actually run."""
    fig, ax = plt.subplots(figsize=(FULL_W, 3.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#E8EDF3", ec="#33506E", fs=6.3, bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=fc, edgecolor=ec, linewidth=0.75, zorder=2,
            )
        )
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.35,
            fontweight="bold" if bold else "normal",
        )

    def arrow(p, q, color="#444444", style="-|>", lw=0.85, rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                p, q, arrowstyle=style, mutation_scale=6.5,
                linewidth=lw, color=color, zorder=1,
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    # column 1: pixel inputs
    box(0.10, 3.95, 1.75, 1.30, "RFMiD fundus pixel\n3200 images\n1920 / 640 / 640", fc="#F2F4F7")
    box(0.10, 2.15, 1.75, 1.30, "Quality proxies\nsharpness + exposure\n$\\rightarrow q\\in\\Delta^{2}$", fc="#F2F4F7")

    # column 2: frozen backbones (actual substitutes)
    box(2.30, 4.35, 1.85, 0.72, "Swin V2-B\nfrozen, 1024-d")
    box(2.30, 3.45, 1.85, 0.72, "ViT-B/16$\\rightarrow$1024\nfrozen, 1024-d")
    box(2.30, 2.55, 1.85, 0.72, "ViT-S/16\nfrozen, 384-d")

    # column 3: fusion
    box(4.55, 3.35, 2.05, 1.20,
        "Monotone bounded\nquality routing\n$w_1\\!<\\!w_2\\!<\\!w_3\\!=\\!1$",
        fc="#FBF0DA", ec="#8A5A1E")
    box(4.55, 1.95, 2.05, 1.05,
        "Quality-conditioned\nbackbone routing\n$\\alpha(q)=\\mathrm{softmax}$",
        fc="#F7E3DC", ec="#9C4A4A")

    # column 4: heads and evaluation
    box(6.95, 3.55, 1.45, 1.00, "8 shared heads\nmasked BCE", fc="#E4EEE4", ec="#3F6B4A")
    box(6.95, 2.25, 1.45, 1.00, "Post-hoc map\n+ operating point", fc="#E4EEE4", ec="#3F6B4A")
    box(8.72, 2.90, 1.20, 1.35,
        "AUROC\nECE\nBrier\nnet benefit", fc="#F2F4F7", fs=6.0)

    for y in (4.71, 3.81, 2.91):
        arrow((1.85, 4.60), (2.30, y), rad=0.0)
    for y in (4.71, 3.81, 2.91):
        arrow((4.15, y), (4.55, 3.95), color="#33506E")
    arrow((1.85, 2.80), (4.55, 3.55), color="#8A5A1E")
    arrow((4.55, 2.48), (4.35, 2.48), color="#9C4A4A", style="-")
    arrow((6.60, 3.95), (6.95, 4.05), color="#33506E")
    arrow((6.60, 2.48), (6.95, 2.75), color="#9C4A4A")
    arrow((8.40, 4.05), (8.72, 3.75), color="#3F6B4A")
    arrow((8.40, 2.75), (8.72, 3.30), color="#3F6B4A")

    ax.text(4.55, 1.50,
            "E5 adds the backbone router; E0–E4 differ only in the shaded fusion and head blocks",
            fontsize=6.0, color="#333333", ha="left", va="center")
    ax.text(0.10, 5.32, "a  pixel to score", fontsize=6.8, fontweight="bold")
    ax.text(4.55, 5.32, "b  fusion and heads", fontsize=6.8, fontweight="bold")
    ax.text(6.95, 5.32, "c  evaluation", fontsize=6.8, fontweight="bold")
    fig.subplots_adjust(0, 0, 1, 1)
    return save(fig, "fig1_architecture")


# --------------------------------------------------------------------- fig2
def fig2_fundus() -> Path:
    """Test-split examples, three per soft-quality stratum, real pixels."""
    from PIL import Image

    sys.path.insert(0, str(ROOT))
    from public_common import load_pairs_manifest

    ql = np.load(DATA / "UKB_mqd.npz")["ql"]
    hard = ql.argmax(axis=1)
    split = np.load(DATA / "split.npz", allow_pickle=True)
    test_idx = np.asarray(split["test_idx"])
    pairs = load_pairs_manifest(DATA / "pairs.csv")

    strata = [("good", 0), ("usable", 1), ("bad", 2)]
    rng = np.random.default_rng(2026)
    fig, axes = plt.subplots(3, 3, figsize=(4.5, 4.75))
    for row, (name, code) in enumerate(strata):
        pool = [i for i in test_idx if hard[i] == code]
        pick = rng.choice(pool, size=min(3, len(pool)), replace=False)
        for col in range(3):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if col >= len(pick):
                ax.axis("off")
                continue
            i = int(pick[col])
            ax.imshow(Image.open(pairs[i][0]).convert("RGB"))
            ax.set_xlabel(
                "q = (%.2f, %.2f, %.2f)" % tuple(float(v) for v in ql[i]),
                fontsize=5.2, labelpad=1.5,
            )
        axes[row, 0].set_ylabel(
            f"{name}\nn = {len(pool)}", fontsize=7.2, labelpad=2,
        )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.985, bottom=0.055, hspace=0.12, wspace=0.04)
    return save(fig, "fig5_fundus_quality_embed")


# --------------------------------------------------------------------- fig3
def fig3_quality_counts() -> Path:
    """Quality strata composition of all three splits from the same cache."""
    ql = np.load(DATA / "UKB_mqd.npz")["ql"]
    hard = ql.argmax(axis=1)
    split = np.load(DATA / "split.npz", allow_pickle=True)
    idx = {"Training": np.asarray(split["train_idx"]),
           "Validation": np.asarray(split["val_idx"]),
           "Test": np.asarray(split["test_idx"])}
    names = ["good", "usable", "bad"]
    shades = ["#4A7C59", "#C9A227", ACCENT]

    fig, ax = plt.subplots(figsize=(3.9, 2.45))
    x = np.arange(len(idx))
    bottom = np.zeros(len(idx))
    for code, (name, col) in enumerate(zip(names, shades)):
        vals = np.array([int((hard[v] == code).sum()) for v in idx.values()], dtype=float)
        frac = vals / np.array([len(v) for v in idx.values()])
        bars = ax.bar(x, frac, 0.62, bottom=bottom, color=col, label=name,
                      edgecolor="white", linewidth=0.5)
        for b, f, v in zip(bars, frac, vals):
            if f > 0.055:
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + f / 2,
                        f"{f*100:.1f}%\n({int(v)})", ha="center", va="center",
                        fontsize=5.6, color="white", linespacing=1.2)
            else:
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + f / 2,
                        f"({int(v)})", ha="center", va="center",
                        fontsize=5.4, color="#333333")
        bottom += frac

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\nn = {len(v)}" for k, v in idx.items()], fontsize=6.4)
    ax.set_ylabel("Share of images")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              handlelength=1.0, columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    return save(fig, "fig6b_quality_pie")


# --------------------------------------------------------------------- fig4
def fig4_ablation() -> Path:
    head = L("headline.json")
    arms = [a for a in ARMS if a in head]
    x = np.arange(len(arms))
    w = 0.36
    fig, ax = plt.subplots(figsize=(FULL_W, 2.55))

    macro = [head[a]["macro_auroc"] for a in arms]
    dr = [head[a]["dr_auroc"] for a in arms]
    err = np.array([
        [head[a]["macro_auroc"] - head[a]["bootstrap"]["ci_low"],
         head[a]["bootstrap"]["ci_high"] - head[a]["macro_auroc"]]
        for a in arms
    ]).T

    ax.bar(x - w / 2, macro, w, yerr=err, capsize=2.0,
           color=[ARM_COLORS[a] for a in arms], edgecolor="white", linewidth=0.5,
           error_kw={"lw": 0.7, "ecolor": "#333333", "capthick": 0.7},
           label="Macro AUROC (eight endpoints)")
    ax.bar(x + w / 2, dr, w, color=[ARM_COLORS[a] for a in arms], alpha=0.42,
           edgecolor="white", linewidth=0.5, label="DR AUROC")
    for xi, m, d in zip(x, macro, dr):
        ax.text(xi - w / 2, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=5.6)
        ax.text(xi + w / 2, d + 0.012, f"{d:.3f}", ha="center", va="bottom", fontsize=5.6)

    ax.set_xticks(x)
    ax.set_xticklabels([TICK[a] for a in arms], fontsize=6.2)
    ax.set_ylabel("AUROC on the held-out test split")
    ax.set_ylim(0.84, 0.98)
    ax.set_yticks([0.85, 0.90, 0.95])
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", ncol=1, handlelength=1.2)
    fig.tight_layout(pad=0.3)
    return save(fig, "fig2_ablation_bars")


# --------------------------------------------------------------------- fig5
def fig5_paired() -> Path:
    """Forest plot plus the bootstrap distribution each interval came from.

    The interval endpoints alone hide how much of the resampling mass sits across
    zero. The right panel draws that mass directly, so a boundary result looks
    like a boundary result instead of a bare pair of numbers.
    """
    paired = L("paired.json")
    boot_path = ANALYSIS / "extra_bootstrap.npz"
    boot = np.load(boot_path, allow_pickle=True) if boot_path.exists() else None

    rows = [
        ("E1_minus_E0", "DR", "DR, single task\nmonotone vs fixed"),
        ("E3_minus_E2", "macro", "Macro, multitask\nmonotone vs fixed"),
        ("E4_minus_E3", "macro", "Macro, multitask\nfree-linear vs monotone"),
        ("E5_minus_E3", "macro", "Macro, multitask\nbackbone routing added"),
    ]
    rows = [r for r in rows if r[0] in paired and r[1] in paired[r[0]]]

    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(FULL_W, 0.78 * len(rows) + 1.45),
        gridspec_kw={"width_ratios": [1.32, 1.0], "wspace": 0.62},
    )

    for i, (key, which, label) in enumerate(rows):
        e = paired[key][which]
        y = len(rows) - 1 - i
        col = "#2F6F63" if (e["ci_low"] > 0 or e["ci_high"] < 0) else NEUTRAL
        ax.plot([e["ci_low"], e["ci_high"]], [y, y], color=col, linewidth=1.5,
                solid_capstyle="butt", zorder=2)
        ax.plot([e["delta"]], [y], "o", color=col, markersize=4.2, zorder=3)
        ax.text(e["ci_high"] + 0.0016, y,
                f"{e['delta']:+.3f}\n[{e['ci_low']:+.3f}, {e['ci_high']:+.3f}]",
                va="center", ha="left", fontsize=5.6, color="#222222",
                linespacing=1.15)

    ax.axvline(0, color=ACCENT, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows][::-1], fontsize=6.0)
    ax.set_xlabel("Paired difference in AUROC")
    ax.set_xlim(-0.038, 0.086)
    ax.set_ylim(-0.62, len(rows) - 0.38)
    ax.grid(True, axis="x", color=GRID, linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title("a  point estimate and 95% interval", fontsize=7.0, loc="left", pad=5)

    if boot is not None:
        grid = np.linspace(-0.045, 0.045, 361)
        for i, (key, which, _) in enumerate(rows):
            if key not in boot.files:
                continue
            d = np.asarray(boot[key], dtype=float)
            dens, _ = np.histogram(d, bins=np.linspace(-0.045, 0.045, 61), density=True)
            centres = 0.5 * (np.linspace(-0.045, 0.045, 61)[1:] + np.linspace(-0.045, 0.045, 61)[:-1])
            dens = np.interp(grid, centres, dens)
            dens = dens / dens.max() * 0.36 if dens.max() > 0 else dens
            y = len(rows) - 1 - i
            col = "#2F6F63" if (np.percentile(d, 2.5) > 0) else NEUTRAL
            axr.fill_between(grid, y, y + dens, color=col, alpha=0.30, linewidth=0)
            axr.plot(grid, y + dens, color=col, linewidth=0.7)
            share = float((d < 0).mean())
            axr.text(-0.0435, y - 0.07, f"{share*100:.0f}% of replicates below zero",
                     fontsize=5.2, color="#333333", va="top", ha="left")
        axr.axvline(0, color=ACCENT, linewidth=0.8, linestyle="--", zorder=1)
        axr.set_yticks(range(len(rows)))
        axr.set_yticklabels([])
        axr.set_ylim(-0.62, len(rows) - 0.38)
        axr.set_xlim(-0.045, 0.045)
        axr.set_xlabel("Bootstrap replicate difference")
        axr.grid(True, axis="x", color=GRID, linewidth=0.45, linestyle=":")
        axr.set_axisbelow(True)
        axr.tick_params(axis="y", length=0)
        axr.set_title("b  resampling mass, 1000 replicates", fontsize=7.0, loc="left", pad=5)

    fig.subplots_adjust(left=0.235, right=0.985, top=0.90, bottom=0.135)
    return save(fig, "fig10_paired_delta")


# --------------------------------------------------------------------- fig6
def fig6_endpoint_delta() -> Path:
    ext = L("extended.json")["per_endpoint"]["delta"]
    pairs = [
        ("E3_minus_E2", "E3 − E2\nrouting, same heads"),
        ("E4_minus_E3", "E4 − E3\nconstraint removed"),
        ("E5_minus_E3", "E5 − E3\nbackbone routing added"),
    ]
    pairs = [(k, t) for k, t in pairs if k in ext]

    fig, ax = plt.subplots(figsize=(FULL_W, 3.5))
    ypos = 0.0
    ticks, labels = [], []
    for gi, (key, title) in enumerate(pairs):
        rows = ext[key]
        top = ypos
        for ep in EP_ORDER:
            if ep not in rows:
                continue
            v = rows[ep]
            col = "#2F6F63" if v["ci_low"] > 0 else (ACCENT if v["ci_high"] < 0 else NEUTRAL)
            ax.plot([v["ci_low"], v["ci_high"]], [ypos, ypos], color=col,
                    linewidth=1.1, solid_capstyle="butt")
            ax.plot([v["delta"]], [ypos], "o", color=col, markersize=3.0, zorder=3)
            ticks.append(ypos)
            labels.append(EP_LABEL[ep].replace("\n", " "))
            ypos -= 1.0
        ax.text(-0.128, top + 0.85, title, fontsize=6.2, color="#222222",
                ha="left", va="bottom", linespacing=1.3)
        ypos -= 1.5
        if gi < len(pairs) - 1:
            ax.axhline(ypos + 0.75, linewidth=0.5, linestyle=":", color="#B0B0B0")

    ax.axvline(0, color="#333333", linewidth=0.75, linestyle="--")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_ylim(ypos + 0.4, 1.6)
    ax.set_xlim(-0.135, 0.135)
    ax.set_xlabel("Difference in AUROC (95% bootstrap interval)")
    ax.grid(True, axis="x", color=GRID, linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=1.5)
    fig.tight_layout(pad=0.3)
    return save(fig, "fig17_endpoint_delta")


# --------------------------------------------------------------------- fig7
def fig7_roc(arm: str = "E5") -> Path:
    d = load_arm(arm)
    names = [str(x) for x in d["names"]]
    fig, axes = plt.subplots(2, 4, figsize=(FULL_W, 3.35))
    for ax, ep in zip(axes.ravel(), EP_ORDER):
        k = names.index(ep)
        y, p = d["labels"][:, k], d["probs"][:, k]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=ARM_COLORS[arm], linewidth=1.15)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.6, color="#AAAAAA")
        ax.set_title(f"{EP_LABEL[ep]}\nAUROC {auc(fpr, tpr):.3f}", fontsize=6.0, pad=2.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=5.2, length=2)
        ax.set_xlabel("1 − specificity", fontsize=5.6, labelpad=1)
        ax.set_ylabel("Sensitivity", fontsize=5.6, labelpad=1)
    fig.tight_layout(pad=0.25, w_pad=0.6, h_pad=0.5)
    return save(fig, "fig8_roc_curves")


# --------------------------------------------------------------------- fig8
def fig8_reliability(arm: str = "E5") -> Path:
    d = load_arm(arm)
    names = [str(x) for x in d["names"]]
    eps = ["Disease_Risk", "DR", "MH"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.1))
    for ax, ep in zip(axes, eps):
        k = names.index(ep)
        y, p = d["labels"][:, k], d["probs"][:, k]
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        mp, mt, cnt = [], [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 8:
                mp.append(p[m].mean())
                mt.append(y[m].mean())
                cnt.append(int(m.sum()))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.65, color="#999999")
        ax.plot(mp, mt, "o-", color=ARM_COLORS[arm], markersize=2.6, linewidth=1.0)
        ece = float(np.sum(np.abs(np.array(mt) - np.array(mp)) * np.array(cnt)) / len(p))
        ax.set_title(f"{EP_LABEL[ep].replace(chr(10), ' ')}   ECE {ece:.3f}", fontsize=6.2, pad=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Mean predicted probability", fontsize=5.8)
        ax.tick_params(labelsize=5.4, length=2)
        ax.grid(True, color=GRID, linewidth=0.4, linestyle=":")
    axes[0].set_ylabel("Observed frequency", fontsize=5.8)
    fig.tight_layout(pad=0.3)
    return save(fig, "fig3_calibration")


# --------------------------------------------------------------------- fig9
def fig9_ece_by_endpoint() -> Path:
    per_head = L("per_head.json")
    co = L("calibration_offset.json")["per_arm"]
    rows = co["E5"]["per_head"]
    eps = sorted(EP_ORDER, key=lambda e: -float(rows[e]["prevalence_test"]))
    arms = ["E3", "E5"]
    x = np.arange(len(eps))
    w = 0.36
    fig, ax = plt.subplots(figsize=(FULL_W, 2.35))
    for j, arm in enumerate(arms):
        vals = [per_head[arm][e]["ece"] for e in eps]
        ax.bar(x - w / 2 + j * w, vals, w * 0.92, color=ARM_COLORS[arm], label=arm,
               edgecolor="white", linewidth=0.4)
    for xi, e in zip(x, eps):
        ax.text(xi, -0.008, f"{float(rows[e]['prevalence_test']):.3f}",
                ha="center", va="top", fontsize=5.0, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([EP_LABEL[e] for e in eps], fontsize=5.4)
    ax.set_ylabel("Expected calibration error")
    ax.set_xlabel("Endpoint, ordered by test prevalence (values below the axis)", fontsize=6.0)
    ax.set_ylim(0, 0.30)
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right", handlelength=1.1)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout(pad=0.3)
    return save(fig, "fig12_ece_by_endpoint")


# -------------------------------------------------------------------- fig10
def fig10_ladder() -> Path:
    hold = L("recalibration_holdout.json")["per_arm"]
    co = L("calibration_offset.json")["per_arm"]
    arms = [a for a in ARMS if a in hold]

    fig = plt.figure(figsize=(FULL_W, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    follow = L("recalibration_followup.json")["arms"]
    rungs = [("none", "None", "#B9C4CF"),
             ("temperature", "Temperature", "#C9B488"),
             ("platt", "Platt", "#3B5B7A"),
             ("isotonic", "Isotonic", "#6E5A8C"),
             ("shared", "Shared Platt", "#A3552F")]
    x = np.arange(len(arms))
    w = 0.165
    for i, (key, name, col) in enumerate(rungs):
        if key == "shared":
            vals = [follow[a]["shared_map"]["macro_ece_shared"] for a in arms]
        else:
            vals = [hold[a]["summary"][f"macro_ece_{key}"] for a in arms]
        ax.bar(x + (i - 2.0) * w, vals, w, color=col, label=name,
               edgecolor="white", linewidth=0.35)
    ax.axhline(0.05, linestyle=":", linewidth=0.6, color="#888888")
    ax.text(len(arms) - 0.45, 0.053, "0.05", fontsize=5.2, color="#666666", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=6.6)
    ax.set_ylabel("Macro per-endpoint ECE on test")
    ax.set_ylim(0, 0.185)
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              columnspacing=0.7, handlelength=0.9, handletextpad=0.3)
    ax.set_title("(a) mapped on validation, scored on test", fontsize=6.6, pad=13)

    ax2 = fig.add_subplot(gs[0, 1])
    marks = {"E2": "^", "E3": "o", "E4": "D", "E5": "s"}
    for arm in ("E2", "E3", "E4", "E5"):
        rows = co[arm]["per_head"]
        eps = [e for e in EP_ORDER if e in rows]
        xv = [float(rows[e]["logit_prevalence_val"]) for e in eps]
        yv = [float(rows[e]["platt_intercept"]) for e in eps]
        r = float(np.corrcoef(xv, yv)[0, 1])
        ax2.scatter(xv, yv, s=16, marker=marks[arm], facecolor="none",
                    edgecolor=ARM_COLORS[arm], linewidth=0.85,
                    label=f"{arm}  r = {r:.2f}")
        b, a = np.polyfit(xv, yv, 1)
        xs = np.linspace(min(xv) - 0.25, max(xv) + 0.25, 20)
        ax2.plot(xs, a + b * xs, linewidth=0.7, color=ARM_COLORS[arm], alpha=0.6)
    ax2.plot([-3.4, 1.8], [-3.4, 1.8], linewidth=0.6, linestyle=":", color="#999999",
             label="identity")
    ax2.axhline(0.0, linewidth=0.5, color="#CCCCCC")
    ax2.set_xlabel("logit of endpoint prevalence")
    ax2.set_ylabel("Fitted Platt intercept")
    ax2.set_title("(b) intercept tracks prevalence", fontsize=6.6, pad=6)
    ax2.legend(frameon=False, fontsize=5.4, loc="upper left", handletextpad=0.3,
               borderaxespad=0.2, labelspacing=0.28)
    ax2.grid(True, color=GRID, linewidth=0.4, linestyle=":")
    ax2.tick_params(labelsize=5.6, length=2)
    fig.tight_layout(pad=0.3)
    return save(fig, "fig16_recalibration_ladder")


# -------------------------------------------------------------------- fig11
def fig11_headline_calibrated() -> Path:
    head = L("headline_calibrated.json")
    arms = [a for a in ARMS if a in head]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.35))
    x = np.arange(len(arms))
    w = 0.36

    a0 = axes[0]
    a0.bar(x - w / 2, [head[a]["macro_auroc_raw"] for a in arms], w,
           color="#3B5B7A", label="Raw sigmoid score", edgecolor="white", linewidth=0.4)
    a0.bar(x + w / 2, [head[a]["macro_auroc_cal"] for a in arms], w,
           color="#A9BCCD", label="After temperature", edgecolor="white", linewidth=0.4)
    a0.set_ylim(0.84, 0.96)
    a0.set_ylabel("Macro AUROC")
    a0.set_title("(a) ranking is unchanged", fontsize=6.8, pad=4)

    a1 = axes[1]
    raw = [head[a]["mean_head_ece_raw"] for a in arms]
    post = [head[a]["mean_head_ece_cal"] for a in arms]
    a1.bar(x - w / 2, raw, w, color="#A9743A", label="Raw sigmoid score",
           edgecolor="white", linewidth=0.4)
    a1.bar(x + w / 2, post, w, color="#DEC3A0", label="After temperature",
           edgecolor="white", linewidth=0.4)
    for xi, r, p in zip(x, raw, post):
        a1.text(xi + w / 2, max(r, p) + 0.006, f"{p:.3f}", ha="center", va="bottom",
                fontsize=5.0, rotation=90)
    a1.set_ylabel("Mean per-endpoint ECE")
    a1.set_ylim(0, 0.20)
    a1.set_title("(b) calibration error rises", fontsize=6.8, pad=4)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(arms, fontsize=6.4)
        ax.grid(True, axis="y", color=GRID, linewidth=0.45, linestyle=":")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=5.4, loc="lower right", handlelength=1.1)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    return save(fig, "fig13_headline_calibrated")


# -------------------------------------------------------------------- fig12
def fig12_threshold_transfer() -> Path:
    """Frozen versus matched operating point, four seeds, plus the mechanism.

    Panel (a) shows every seed as a dot on top of the arm mean, because the whole
    point of this figure is that the single-seed gap in the first draft does not
    survive replication.  Panel (b) plots the excess specificity against the
    realised sensitivity loss, the diagnostic that explains the gap.
    """
    seed_dir = ROOT
    tt = json.loads((seed_dir / "threshold_transfer_replication.json").read_text(encoding="utf-8"))
    arms = [a for a in ARMS if a in tt["per_arm"]]
    seeds = tt["seeds"]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.45),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    x = np.arange(len(arms))
    w = 0.34
    frozen = [tt["per_arm"][a]["aggregate"]["test_sensitivity_frozen"]["mean"] for a in arms]
    matched = [tt["per_arm"][a]["aggregate"]["test_sensitivity_matched"]["mean"] for a in arms]
    ferr = [tt["per_arm"][a]["aggregate"]["test_sensitivity_frozen"]["sd"] for a in arms]
    merr = [tt["per_arm"][a]["aggregate"]["test_sensitivity_matched"]["sd"] for a in arms]
    ax.bar(x - w / 2, frozen, w, yerr=ferr, capsize=1.8, color="#3F6BA8",
           label="Frozen on validation", edgecolor="white", linewidth=0.4,
           error_kw={"lw": 0.6, "ecolor": "#333333", "capthick": 0.6})
    ax.bar(x + w / 2, matched, w, yerr=merr, capsize=1.8, color="#B7C8DA",
           label="Matched on test (upper bound)", edgecolor="white", linewidth=0.4,
           error_kw={"lw": 0.6, "ecolor": "#333333", "capthick": 0.6})
    rng = np.random.default_rng(0)
    for xi, a in zip(x, arms):
        rows = tt["per_arm"][a]["rows"]
        jf = rng.uniform(-0.09, 0.09, len(rows))
        jm = rng.uniform(-0.09, 0.09, len(rows))
        ax.scatter(xi - w / 2 + jf, [r["test_sensitivity_frozen"] for r in rows], s=3.4,
                   color="#1E3A5F", linewidths=0, zorder=4)
        ax.scatter(xi + w / 2 + jm, [r["test_sensitivity_matched"] for r in rows], s=3.4,
                   color="#5E7B9B", linewidths=0, zorder=4)
        ax.text(xi - w / 2, frozen[xi] + ferr[xi] + 0.012, f"{frozen[xi]:.2f}", ha="center",
                va="bottom", fontsize=4.9)
        ax.text(xi + w / 2, matched[xi] + merr[xi] + 0.012, f"{matched[xi]:.2f}", ha="center",
                va="bottom", fontsize=4.9, color="#31506E")
    ax.set_xticks(x)
    ax.set_xticklabels([TICK[a] for a in arms], fontsize=5.6)
    ax.set_ylabel("DR sensitivity at 95% specificity")
    ax.set_ylim(0.30, 0.95)
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.2, loc="lower left", handlelength=1.1)
    ax.set_title(f"(a) {len(seeds)} seeds per arm, dots are single runs", fontsize=6.4, pad=4)

    ax2 = axes[1]
    for a in arms:
        ex = [r["excess_specificity"] for r in tt["per_arm"][a]["rows"]]
        ls = [r["transfer_loss"] for r in tt["per_arm"][a]["rows"]]
        ax2.scatter(ex, ls, s=9, color=ARM_COLORS[a], label=a,
                    edgecolors="white", linewidths=0.3, zorder=3)
    ex_all = np.array([r["excess_specificity"] for a in arms for r in tt["per_arm"][a]["rows"]])
    ls_all = np.array([r["transfer_loss"] for a in arms for r in tt["per_arm"][a]["rows"]])
    slope = float(np.polyfit(ex_all, ls_all, 1)[0])
    xs = np.linspace(0, ex_all.max() * 1.05, 20)
    ax2.plot(xs, slope * xs, color=NEUTRAL, linewidth=0.7, linestyle="--", zorder=2)
    ax2.axhline(0, color=NEUTRAL, linewidth=0.5, linestyle=":", zorder=1)
    ax2.set_xlabel("Excess specificity above 0.95")
    ax2.set_ylabel("Sensitivity lost by freezing")
    ax2.grid(True, color=GRID, linewidth=0.45, linestyle=":")
    ax2.set_axisbelow(True)
    ax2.tick_params(labelsize=5.2, length=2)
    ax2.legend(frameon=False, fontsize=4.9, ncol=2, loc="lower right",
               handlelength=0.7, columnspacing=0.5, handletextpad=0.25,
               labelspacing=0.22, borderpad=0.1)
    ax2.text(0.98, 0.98, f"r = {np.corrcoef(ex_all, ls_all)[0, 1]:.2f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=5.4)
    ax2.set_title("(b) the loss is the ROC slope times that excess", fontsize=6.4, pad=4)

    fig.tight_layout(pad=0.3, w_pad=1.0)
    return save(fig, "fig14_threshold_transfer")


# -------------------------------------------------------------------- fig13
def fig13_score_shift() -> Path:
    """Validation versus test negative-score density, with the four-seed drift.

    The histogram is drawn for the primary run so the shape stays readable, and
    the panel title carries the mean negative-score displacement over four seeds,
    which is the quantity the text quotes.  Drawing four overlaid histograms per
    panel would obscure both curves without adding evidence.
    """
    manifest = json.loads((ANALYSIS / "manifest.json").read_text(encoding="utf-8"))
    seed_dir = ROOT
    tt = json.loads((seed_dir / "threshold_transfer_replication.json").read_text(encoding="utf-8"))
    arms = [a for a in ARMS if a in manifest]
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 3.15), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 41)
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
        ax.hist(vneg, bins=bins, density=True, histtype="stepfilled", linewidth=0,
                alpha=0.40, color="#3B5B7A", label="Validation negatives")
        ax.hist(tneg, bins=bins, density=True, histtype="step", linewidth=0.9,
                linestyle="--", color=ACCENT, label="Test negatives")
        shift = tt["per_arm"][arm]["aggregate"]["negative_mean_shift"]
        ax.set_title(f"{arm}   Δμ = {shift['mean']:+.3f} ± {shift['sd']:.3f}",
                     fontsize=5.8, pad=2.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.grid(True, color=GRID, linewidth=0.4, linestyle=":")
        ax.tick_params(labelsize=5.2, length=2)
    for ax in axes[-1]:
        ax.set_xlabel("Predicted DR probability, negative images", fontsize=5.6, labelpad=1)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density", fontsize=5.6, labelpad=1)
    handles = axes[0, 0].get_legend_handles_labels()
    fig.legend(*handles, fontsize=5.6, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=False, handlelength=1.4)
    fig.tight_layout(pad=0.25, rect=(0, 0, 1, 0.97), w_pad=0.6, h_pad=0.55)
    return save(fig, "fig15_score_shift")


# -------------------------------------------------------------------- fig14
def fig14_dca(arm: str = "E5") -> Path:
    dca = L("decision_curves.json")[arm]
    eps = ["DR", "Disease_Risk", "MH", "DN"]
    eps = [e for e in eps if e in dca]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 2.9))
    for ax, ep in zip(axes.ravel(), eps):
        c = dca[ep]
        thr = np.asarray(c["thresholds"])
        ax.plot(thr, c["net_benefit"], color=ARM_COLORS[arm], linewidth=1.2, label="Model")
        ax.plot(thr, c["treat_all"], color="#A9743A", linewidth=0.9, linestyle="--",
                label="Treat all")
        ax.axhline(0, color=NEUTRAL, linewidth=0.7, linestyle=":", label="Treat none")
        ax.set_title(EP_LABEL[ep].replace("\n", " "), fontsize=6.4, pad=3)
        ax.set_xlim(0.05, 0.5)
        ax.set_ylim(-0.20, max(0.06, float(np.max(c["net_benefit"])) * 1.30))
        ax.set_xlabel("Threshold probability", fontsize=5.8)
        ax.set_ylabel("Net benefit", fontsize=5.8)
        ax.grid(True, color=GRID, linewidth=0.4, linestyle=":")
        ax.tick_params(labelsize=5.2, length=2)
    for ax in axes.ravel()[len(eps):]:
        ax.axis("off")
    axes.ravel()[0].legend(frameon=False, fontsize=5.2, loc="upper right",
                           handlelength=1.2, labelspacing=0.25)
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.6)
    return save(fig, "fig7_decision_curves")


# -------------------------------------------------------------------- fig15
def fig15_quality_strata() -> Path:
    """Macro AUROC inside each stratum, next to the prevalence gap that explains it."""
    st = L("quality_strata.json")
    arms = [a for a in ARMS if a in st and "bad" in st[a]]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.55),
                             gridspec_kw={"width_ratios": [1.65, 1.0]})

    ax = axes[0]
    x = np.arange(len(arms))
    w = 0.36
    usable = [st[a]["usable"]["macro_auroc"] for a in arms]
    bad = [st[a]["bad"]["macro_auroc"] for a in arms]
    ax.bar(x - w / 2, usable, w, color="#B9C4CF", label="Usable (n = 546)",
           edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, bad, w, color=ACCENT, label="Bad (n = 85)",
           edgecolor="white", linewidth=0.4)
    for xi, u, b in zip(x, usable, bad):
        ax.text(xi - w / 2, u + 0.004, f"{u:.3f}", ha="center", va="bottom", fontsize=5.0)
        ax.text(xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", va="bottom", fontsize=5.0)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=6.4)
    ax.set_ylim(0.86, 0.98)
    ax.set_ylabel("Macro AUROC within stratum")
    ax.set_title("(a) the bad stratum scores higher in every arm", fontsize=6.4, pad=4)
    ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="lower left", handlelength=1.1)
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)

    ax2 = axes[1]
    prev_u = st[arms[0]]["usable"]["dr_prevalence"]
    prev_b = st[arms[0]]["bad"]["dr_prevalence"]
    bars = ax2.bar([0, 1], [prev_u, prev_b], 0.5,
                   color=["#B9C4CF", ACCENT], edgecolor="white", linewidth=0.4)
    for b, v in zip(bars, [prev_u, prev_b]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=5.8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Usable", "Bad"], fontsize=6.4)
    ax2.set_ylabel("DR prevalence in stratum")
    ax2.set_ylim(0, 0.26)
    ax2.set_title("(b) the strata differ in case mix", fontsize=6.4, pad=4)
    ax2.grid(True, axis="y", color=GRID, linewidth=0.45, linestyle=":")
    ax2.set_axisbelow(True)

    fig.tight_layout(pad=0.3, w_pad=1.2)
    return save(fig, "fig11_quality_strata")


# -------------------------------------------------------------------- fig16
def fig16_router() -> Path:
    coef = L("router_coefficients.json")
    interv = L("router_intervention.json")
    arms = [a for a in ("E1", "E3", "E4", "E5") if a in coef and a in interv]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    levels = ["bad", "usable", "good"]
    x = np.arange(len(levels))
    w = 0.8 / len(arms)
    for j, arm in enumerate(arms):
        dev = np.abs(np.array(coef[arm]["deviation_from_baseline"], dtype=float))
        ax.bar(x - 0.4 + w * (j + 0.5), np.maximum(dev, 2e-5), w * 0.86,
               color=ARM_COLORS[arm], label=arm, edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_ylim(2e-5, 3e-2)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in levels], fontsize=6.4)
    ax.set_ylabel("|learned weight − fixed weight|")
    ax.grid(True, axis="y", color=GRID, linewidth=0.4, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.6, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.15), columnspacing=0.8, handlelength=0.9,
              handletextpad=0.3)
    ax.set_title("(a) distance from released weights", fontsize=6.4, pad=13)

    ax2 = axes[1]
    y = np.arange(len(arms))[::-1]
    shift = [interv[a]["max_abs_prob_change"] for a in arms]
    ax2.barh(y, shift, height=0.5, color=[ARM_COLORS[a] for a in arms],
             edgecolor="white", linewidth=0.3)
    for yi, arm in zip(y, arms):
        e = interv[arm]
        ax2.text(e["max_abs_prob_change"] + 0.0018, yi,
                 f"max |Δp| {e['max_abs_prob_change']:.3f}\nΔAUROC {e['delta']:+.1e}",
                 va="center", ha="left", fontsize=5.2, color="#333333",
                 linespacing=1.35)
    ax2.set_yticks(y)
    ax2.set_yticklabels(arms, fontsize=6.4)
    ax2.set_xlim(0, 0.108)
    ax2.set_ylim(-0.6, len(arms) - 0.4)
    ax2.set_xlabel("Largest per-image probability change")
    ax2.grid(True, axis="x", color=GRID, linewidth=0.4, linestyle=":")
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("(b) forcing the router back", fontsize=6.4, pad=6)

    fig.tight_layout(pad=0.3, w_pad=1.2)
    return save(fig, "fig18_router_intervention")


# -------------------------------------------------------------------- fig17
def fig17_seed_replication() -> Path:
    """Seed-level spread next to the single-run bootstrap interval.

    Left panel: every training seed as a dot, the arm mean as a bar, so the
    reader sees how far apart two runs of the same design can land.
    Right panel: the four design comparisons, with the single-run bootstrap
    interval drawn as a horizontal bar and the four seed-level deltas on top.
    """
    seed_dir = ROOT
    repl = json.loads((seed_dir / "seed_replication.json").read_text(encoding="utf-8"))
    per_arm = repl["per_arm"]
    arms = [a for a in ARMS if a in per_arm]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.55),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    x = np.arange(len(arms))
    means = [per_arm[a]["mean"] for a in arms]
    ax.bar(x, means, 0.56, color=["#B9C4CF" if a in ("E0", "E1", "E2") else "#8FA6BC"
                                  for a in arms],
           edgecolor="white", linewidth=0.4, zorder=1)
    rng = np.random.default_rng(0)
    for xi, arm in zip(x, arms):
        vals = list(per_arm[arm]["by_seed"].values())
        vals = [v["macro_auroc"] for v in vals]
        offs = rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(xi + offs, vals, s=9, color=ARM_COLORS[arm], zorder=3,
                   edgecolors="white", linewidths=0.35)
        ax.text(xi, 0.9515, f"sd {per_arm[arm]['sd']:.3f}", ha="center",
                va="top", fontsize=4.9, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=6.4)
    ax.set_ylim(0.885, 0.956)
    ax.set_ylabel("Macro AUROC, seed level")
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_title("(a) four runs per design", fontsize=6.4, pad=4)

    ax2 = axes[1]
    comps = [c for c in repl["comparisons"] if c["n_seeds"] > 1]
    labels = {
        "单任务 单调路由 减 固定权重": "E1 − E0",
        "多任务 单调路由 减 固定权重": "E3 − E2",
        "多任务 自由线性 减 单调路由": "E4 − E3",
        "骨干路由 减 单调路由": "E5 − E3",
    }
    y = np.arange(len(comps))[::-1]
    for yi, c in zip(y, comps):
        deltas = np.array(list(c["delta_by_seed"].values()))
        ax2.plot([deltas.min(), deltas.max()], [yi, yi], color="#B0B8C0",
                 linewidth=3.2, solid_capstyle="round", zorder=1)
        ax2.scatter(deltas, np.full(deltas.shape, yi), s=12, zorder=3,
                    color=ACCENT if c["mean_delta"] < 0 else ARM_COLORS["E5"],
                    edgecolors="white", linewidths=0.4)
        ax2.scatter([c["mean_delta"]], [yi], marker="D", s=20, color="#222222", zorder=4)
    ax2.axvline(0, color=NEUTRAL, linewidth=0.7, linestyle="--", zorder=0)
    ax2.set_yticks(y)
    ax2.set_yticklabels(
        [f"{labels.get(c['label'], c['label'])}\n{c['mean_delta']:+.4f}" for c in comps],
        fontsize=5.9, linespacing=1.5,
    )
    ax2.set_xlim(-0.021, 0.027)
    ax2.set_ylim(-0.55, len(comps) - 0.45)
    ax2.set_xlabel("Δ macro AUROC, one dot per seed")
    ax2.grid(True, axis="x", color=GRID, linewidth=0.45, linestyle=":")
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("(b) only E5 − E3 keeps its sign", fontsize=6.4, pad=4)

    fig.tight_layout(pad=0.3, w_pad=1.1)
    return save(fig, "fig19_seed_replication")


# -------------------------------------------------------------------- fig18
def fig18_calibration_replication() -> Path:
    """Temperature versus prevalence shift, per seed, shared-head arms only.

    The left panel plots every seed as a line from the identity error to the
    error after each rule. A rule that helps shows its markers below the entry
    point, which temperature never does.
    """
    seed_dir = ROOT
    cal = json.loads((seed_dir / "calibration_replication.json").read_text(encoding="utf-8"))
    arms = [a for a in ("E2", "E3", "E4", "E5") if a in cal["per_arm"]]
    rules = [("ece_identity", "Raw"),
             ("ece_temperature_global", "Temperature"),
             ("ece_temperature_per_head", "Per-endpoint T"),
             ("ece_prevalence_shift", "Prevalence shift"),
             ("ece_fitted_shift", "Fitted shift")]
    colors = ["#444444", ACCENT, "#C58A50", ARM_COLORS["E5"], "#2F6F63"]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.85),
                             gridspec_kw={"width_ratios": [1.0, 1.35]})

    ax = axes[0]
    x = np.arange(len(rules))
    for arm in arms:
        rows = cal["per_arm"][arm]["per_seed"]
        for row in rows:
            vals = [row[k] for k, _ in rules]
            ax.plot(x, vals, color=ARM_COLORS[arm], linewidth=0.5, alpha=0.45,
                    marker="o", markersize=1.5, zorder=2)
        mean = [cal["per_arm"][arm]["aggregate"][f"{k}_mean"] for k, _ in rules]
        ax.plot(x, mean, color=ARM_COLORS[arm], linewidth=1.6, zorder=4,
                marker="s", markersize=2.8, label=arm)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in rules], fontsize=5.2, rotation=24, ha="right")
    ax.set_ylabel("Mean per-endpoint ECE, test")
    ax.set_ylim(0.0, 0.20)
    ax.set_xlim(-0.25, len(rules) - 0.75)
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.2, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), handlelength=0.9, columnspacing=0.6,
              handletextpad=0.25)
    ax.set_title("(a) thin lines are single seeds", fontsize=6.4, pad=13)

    ax2 = axes[1]
    y = np.arange(len(arms))[::-1]
    w = 0.17
    for j, (key, lab) in enumerate(rules[:4]):
        vals = [cal["per_arm"][a]["aggregate"][f"{key}_mean"] for a in arms]
        ax2.barh(y + (1.5 - j) * w, vals, w * 0.9, color=colors[j],
                 edgecolor="white", linewidth=0.3, label=lab)
    ax2.set_yticks(y)
    ax2.set_yticklabels(arms, fontsize=6.4)
    ax2.set_xlabel("Mean per-endpoint ECE")
    ax2.set_xlim(0, 0.185)
    ax2.grid(True, axis="x", color=GRID, linewidth=0.45, linestyle=":")
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", length=0)
    ax2.legend(frameon=False, fontsize=5.0, ncol=2, loc="lower right",
               handlelength=0.9, columnspacing=0.7, handletextpad=0.35,
               labelspacing=0.3)
    ax2.set_title("(b) shift closes the gap, temperature does not",
                  fontsize=6.4, pad=4)

    fig.tight_layout(pad=0.3, w_pad=1.0)
    return save(fig, "fig20_calibration_replication")


# --------------------------------------------------------------------- main
BUILDERS = {
    "fig1": fig1_architecture,
    "fig2": fig2_fundus,
    "fig3": fig3_quality_counts,
    "fig4": fig4_ablation,
    "fig5": fig5_paired,
    "fig6": fig6_endpoint_delta,
    "fig7": fig7_roc,
    "fig8": fig8_reliability,
    "fig9": fig9_ece_by_endpoint,
    "fig10": fig10_ladder,
    "fig11": fig11_headline_calibrated,
    "fig12": fig12_threshold_transfer,
    "fig13": fig13_score_shift,
    "fig14": fig14_dca,
    "fig15": fig15_quality_strata,
    "fig16": fig16_router,
    "fig17": fig17_seed_replication,
    "fig18": fig18_calibration_replication,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of keys, e.g. fig5 fig15")
    args = ap.parse_args()
    keys = args.only or list(BUILDERS)
    fig_style.apply(8.5)
    for k in keys:
        if k not in BUILDERS:
            raise SystemExit(f"unknown figure key {k}")
        print(f"{k}:")
        BUILDERS[k]()
    print(f"\nall figures written to {ASSETS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
