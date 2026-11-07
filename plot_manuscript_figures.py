#!/usr/bin/env python3
"""Main figures for the manuscript, rebuilt from the selected-epoch statistics.

This replaces the older plot path that read ``metrics_E*_rfmid_test.json``
(written at epoch 9 by filename sorting). Everything here reads
``results/real_rfmid/analysis_sel/stats``.

Figures produced
----------------
fig2  ablation bars: macro AUROC and DR AUROC per arm, with bootstrap error bars
fig3  reliability diagram (DR, Disease_Risk, MH) for the best arm
fig7  decision curves for four endpoints, three decision strategies each
fig8  ROC curves for all eight endpoints of the best arm
fig9  DR score distribution by label
fig12 per-endpoint calibration error, comparing a single-task and a multitask arm
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
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent
ANALYSIS = Path(os.environ.get("ARM_ANALYSIS_DIR", ROOT))
STATS = Path(os.environ.get("ARM_STATS_DIR", ROOT))
ASSETS = ROOT

ARMS = ["E0", "E1", "E2", "E3", "E4", "E5"]
SHORT = {
    "E0": "E0\nfixed\nsingle",
    "E1": "E1\nmonotone\nsingle",
    "E2": "E2\nfixed\nmultitask",
    "E3": "E3\nmonotone\nmultitask",
    "E4": "E4\nfree\nmultitask",
    "E5": "E5\nrouting\nmultitask",
}
COLORS = {
    "E0": "#4C6E91",
    "E1": "#2E7D6F",
    "E2": "#B07A3C",
    "E3": "#A34E4E",
    "E4": "#7A5C99",
    "E5": "#3F6BA8",
}
ENDPOINT_CN = {
    "Disease_Risk": "Any disease",
    "DR": "Diabetic retinopathy",
    "MH": "Media haze",
    "ODC": "Optic disc cupping",
    "TSLN": "Tessellation",
    "DN": "Drusen",
    "ARMD": "Macular degeneration",
    "MYA": "Myopia",
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
            "legend.frameon": False,
        }
    )


def load_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def load_arm(name: str) -> dict:
    d = np.load(ANALYSIS / f"arm_{name}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def save(fig, stem: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / f"{stem}.png")
    fig.savefig(ASSETS / f"{stem}.pdf")
    plt.close(fig)


def fig_ablation(headline: dict) -> None:
    arms = [a for a in ARMS if a in headline]
    x = np.arange(len(arms))
    macro = [headline[a]["macro_auroc"] for a in arms]
    dr = [headline[a]["dr_auroc"] for a in arms]
    macro_err = [
        [
            headline[a]["macro_auroc"] - headline[a]["bootstrap"]["ci_low"],
            headline[a]["bootstrap"]["ci_high"] - headline[a]["macro_auroc"],
        ]
        for a in arms
    ]
    macro_err = np.array(macro_err).T

    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    w = 0.36
    ax.bar(x - w / 2, macro, w, yerr=macro_err, capsize=2.5, color="#4C6E91",
           label="Macro AUROC (8 endpoints)", error_kw={"lw": 0.8, "ecolor": "#333333"})
    ax.bar(x + w / 2, dr, w, color="#9DB4C9", label="DR AUROC")
    for xi, m, d in zip(x, macro, dr):
        ax.annotate(f"{m:.3f}", xy=(xi - w / 2, m), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.4)
        ax.annotate(f"{d:.3f}", xy=(xi + w / 2, d), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.4, color="#31506E")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[a] for a in arms], fontsize=7)
    ax.set_ylabel("AUROC on the held-out test split")
    ax.set_ylim(0.80, 1.0)
    ax.grid(axis="y", ls=":", lw=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig2_ablation_bars")


def fig_reliability(arm: str, endpoints=("DR", "Disease_Risk", "MH")) -> None:
    d = load_arm(arm)
    names = [str(x) for x in d["names"]]
    fig, axes = plt.subplots(1, len(endpoints), figsize=(2.5 * len(endpoints), 2.5))
    for ax, ep in zip(np.atleast_1d(axes), endpoints):
        if ep not in names:
            ax.axis("off")
            continue
        k = names.index(ep)
        y, p = d["labels"][:, k], d["probs"][:, k]
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        mp, mt = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 8:
                mp.append(p[m].mean())
                mt.append(y[m].mean())
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#999999")
        ax.plot(mp, mt, "o-", color="#2E5E8C", ms=3.5, lw=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability", fontsize=8)
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel("Observed frequency", fontsize=8)
        ax.set_title(ENDPOINT_CN.get(ep, ep), fontsize=8)
        ax.set_aspect("equal")
        ax.grid(ls=":", lw=0.4, color="#DDDDDD")
    fig.tight_layout()
    save(fig, "fig3_calibration")


def fig_dca(arm: str, endpoints=("DR", "Disease_Risk", "MH", "DN")) -> None:
    dca = load_json("decision_curves.json")[arm]
    endpoints = [e for e in endpoints if e in dca]
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.6))
    for ax, ep in zip(axes.ravel(), endpoints):
        c = dca[ep]
        thr = np.asarray(c["thresholds"])
        ax.plot(thr, c["net_benefit"], color="#2E5E8C", lw=1.4, label="Model")
        ax.plot(thr, c["treat_all"], color="#B07A3C", lw=1.0, ls="--", label="Treat all")
        ax.axhline(0, color="#7A7A7A", lw=0.9, ls=":", label="Treat none")
        ax.set_title(ENDPOINT_CN.get(ep, ep), fontsize=8.5)
        ax.set_xlim(0.05, 0.5)
        ax.set_ylim(-0.25, max(0.05, float(np.max(c["net_benefit"])) * 1.35))
        ax.set_xlabel("Threshold probability", fontsize=8)
        ax.set_ylabel("Net benefit", fontsize=8)
        ax.grid(ls=":", lw=0.4, color="#DDDDDD")
    for ax in axes.ravel()[len(endpoints):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=6.8, loc="upper right")
    fig.tight_layout()
    save(fig, "fig7_decision_curves")


def fig_roc(arm: str) -> None:
    d = load_arm(arm)
    names = [str(x) for x in d["names"]]
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.0))
    for ax, ep in zip(axes.ravel(), names):
        k = names.index(ep)
        y, p = d["labels"][:, k], d["probs"][:, k]
        fpr, tpr, _ = roc_curve(y, p)
        from sklearn.metrics import auc

        ax.plot(fpr, tpr, color="#2E5E8C", lw=1.3)
        ax.plot([0, 1], [0, 1], ls="--", lw=0.7, color="#AAAAAA")
        ax.set_title(f"{ENDPOINT_CN.get(ep, ep)}\nAUROC {auc(fpr, tpr):.3f}", fontsize=7.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=6.5)
    for ax in axes.ravel():
        ax.set_xlabel("1 - specificity", fontsize=6.8, labelpad=1)
        ax.set_ylabel("Sensitivity", fontsize=6.8, labelpad=1)
    fig.tight_layout()
    save(fig, "fig8_roc_curves")


def fig_score_hist(arm: str, ep: str = "DR") -> None:
    d = load_arm(arm)
    names = [str(x) for x in d["names"]]
    k = names.index(ep)
    y, p = d["labels"][:, k], d["probs"][:, k]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bins = np.linspace(0, 1, 26)
    ax.hist(p[y == 0], bins=bins, alpha=0.65, color="#7C9EB8", label="No DR", density=True)
    ax.hist(p[y == 1], bins=bins, alpha=0.65, color="#A34E4E", label="DR", density=True)
    ax.set_xlabel("Predicted probability of DR", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(ls=":", lw=0.4, color="#DDDDDD")
    fig.tight_layout()
    save(fig, "fig9_score_hist")


def fig_calibration_bars() -> None:
    per_head = load_json("per_head.json")
    arms = [a for a in ("E3", "E5") if a in per_head]
    eps = ["DR", "MH", "ODC", "TSLN", "DN", "ARMD", "MYA"]
    x = np.arange(len(eps))
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    w = 0.8 / max(1, len(arms))
    for j, arm in enumerate(arms):
        vals = [per_head[arm].get(e, {}).get("ece", np.nan) for e in eps]
        ax.bar(x - 0.4 + w * (j + 0.5), vals, w * 0.9, color=COLORS[arm],
               label={"E3": "E3 monotone weights, masked multitask",
                      "E5": "E5 E3 plus quality-conditioned backbone routing"}[arm])
    ax.set_xticks(x)
    ax.set_xticklabels([ENDPOINT_CN.get(e, e).replace(" ", "\n") for e in eps], fontsize=6.6)
    ax.set_ylabel("Expected calibration error")
    ax.grid(axis="y", ls=":", lw=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig12_ece_by_endpoint")


def main() -> None:
    style()
    headline = load_json("headline.json")
    fig_ablation(headline)
    fig_reliability("E5")
    fig_dca("E5")
    fig_roc("E5")
    fig_score_hist("E5")
    fig_calibration_bars()
    print("figures written to", ASSETS)


if __name__ == "__main__":
    sys.exit(main())
