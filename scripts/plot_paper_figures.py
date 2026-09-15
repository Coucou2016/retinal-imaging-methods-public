#!/usr/bin/env python3
"""Regenerate SciencePlots figures from ablation_summary.csv / eval JSON.

All AUROC/ECE values from synthetic feature caches are labeled SYNTHETIC.
Do not use these plots as clinical performance claims.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _setup_style(*, allow_cjk: bool = False) -> None:
    """SciencePlots + Times New Roman; optional CJK fallback for Chinese labels.

    Axis / panel labels stay English by default (Times New Roman). If a future
    figure needs Chinese strings, set allow_cjk=True so SimHei/SimSun can mix
    without tofu glyphs while Latin still prefers Times New Roman.
    """
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex", "grid"])
    except Exception:
        mpl.rcParams.update(
            {
                "axes.grid": True,
                "grid.alpha": 0.28,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
    # Prefer Times New Roman; CJK-safe stack only when Chinese labels appear.
    serif = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "DejaVu Serif",
    ]
    if allow_cjk:
        # Put CJK fonts after Times so Latin glyphs still resolve to TNR when possible.
        serif.extend(["SimSun", "SimHei", "Microsoft YaHei", "Noto Serif CJK SC"])
        mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = serif
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9
    mpl.rcParams["legend.fontsize"] = 8.5
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.major.width"] = 0.6
    mpl.rcParams["ytick.major.width"] = 0.6
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["figure.facecolor"] = "white"


def _read_ablation(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            rows.append(row)
    return rows


def _f(row: dict, key: str) -> float:
    return float(row[key])


def plot_ablation_bars(rows: list[dict], out: Path) -> Path:
    arms = ["baseline", "learnq", "multitask", "full"]
    labels = {
        "baseline": "Fixed-q\n(single-task)",
        "learnq": "Learnable-q\n(single-task)",
        "multitask": "Multi-task\n(fixed-q)",
        "full": "Full\n(learnq+MT)",
    }
    val = {r["ablation"]: r for r in rows if r["eval"] == "odir_val"}
    cross = {r["ablation"]: r for r in rows if r["eval"] == "odir_to_brset"}

    x = np.arange(len(arms))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), constrained_layout=True)

    for ax, metric, title, ylim in [
        (axes[0], "auroc_D", "ODIR-D head AUROC (SYNTHETIC)", (0.0, 1.0)),
        (axes[1], "ece", "ODIR val ECE before / after T-scaling (SYNTHETIC)", (0.0, 0.8)),
    ]:
        if metric == "auroc_D":
            v = [_f(val[a], "auroc_D") for a in arms]
            c = [_f(cross[a], "auroc_D") for a in arms]
            ax.bar(x - width / 2, v, width, label="ODIR val", color="#1f4e79", edgecolor="0.2", lw=0.4)
            ax.bar(x + width / 2, c, width, label="ODIR→BRSET", color="#8faadc", edgecolor="0.2", lw=0.4)
            ax.set_ylabel("AUROC")
        else:
            raw = [_f(val[a], "ece") for a in arms]
            cal = [_f(val[a], "cal_ece") for a in arms]
            ax.bar(x - width / 2, raw, width, label="Raw ECE", color="#c55a11", edgecolor="0.2", lw=0.4)
            ax.bar(x + width / 2, cal, width, label="Calibrated ECE", color="#70ad47", edgecolor="0.2", lw=0.4)
            ax.set_ylabel("ECE")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[a] for a in arms])
        ax.set_ylim(*ylim)
        ax.set_title(title, pad=8)
        ax.legend(frameon=False, loc="upper right")
        ax.axhline(0.5 if metric == "auroc_D" else 0.0, color="0.45", ls="--", lw=0.75)

    fig.suptitle(
        "Ablation matrix on synthetic ODIR/BRSET feature caches — pipeline sanity only",
        fontsize=11,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_calibration_effect(rows: list[dict], out: Path) -> Path:
    arms = ["baseline", "learnq", "multitask", "full"]
    val = {r["ablation"]: r for r in rows if r["eval"] == "odir_val"}
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    raw = [_f(val[a], "ece") for a in arms]
    cal = [_f(val[a], "cal_ece") for a in arms]
    x = np.arange(len(arms))
    ax.plot(x, raw, "o-", label="Raw ECE", color="#c55a11")
    ax.plot(x, cal, "s-", label="After temperature scaling", color="#2e7d32")
    for i, a in enumerate(arms):
        ax.annotate(
            f"Δ={raw[i] - cal[i]:.2f}",
            (x[i], cal[i]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("Expected calibration error (ECE)")
    ax.set_xlabel("Ablation arm")
    ax.set_title("Temperature scaling reduces ECE (SYNTHETIC)")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(raw) * 1.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_cross_domain(rows: list[dict], out: Path) -> Path:
    arms = ["baseline", "learnq", "multitask", "full"]
    val = {r["ablation"]: r for r in rows if r["eval"] == "odir_val"}
    cross = {r["ablation"]: r for r in rows if r["eval"] == "odir_to_brset"}
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x = np.arange(len(arms))
    v = [_f(val[a], "auroc_D") for a in arms]
    c = [_f(cross[a], "auroc_D") for a in arms]
    ax.plot(x, v, "o-", label="ODIR val (D head)", color="#1f4e79")
    ax.plot(x, c, "s--", label="ODIR→BRSET (mapped D)", color="#c00000")
    ax.fill_between(x, v, c, alpha=0.15, color="#c00000", label="Domain gap")
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("AUROC (D / diabetes-related)")
    ax.set_xlabel("Ablation arm")
    ax.set_title("Cross-dataset drop (SYNTHETIC; not clinical)")
    ax.axhline(0.5, color="0.5", ls=":", lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0.2, 0.7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_architecture_schematic(out: Path) -> Path:
    """English schematic of Reti-Pioneer clone vs proposed extensions."""
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#d6e3f0", ec="#1f4e79"):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.15, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, zorder=3)

    box(0.25, 3.45, 1.9, 1.05, "CFP L/R +\nmetadata +\nquality probs", fc="#eef3f8")
    box(2.5, 3.75, 1.65, 0.7, "Frozen\nRETFound")
    box(2.5, 2.75, 1.65, 0.7, "Frozen\nSwin V2-B")
    box(2.5, 1.75, 1.65, 0.7, "Frozen\nVim-S")
    box(4.55, 2.15, 2.05, 1.7, "Monotone bounded\nquality routing\n+ bilinear fusion", fc="#fff2cc", ec="#b45f06")
    box(7.0, 2.45, 2.55, 1.25, "Partial-label MTL\n(masked BCE)\n+ cal/DCA eval", fc="#e2efda", ec="#548235")

    # Input → each backbone
    for y in (4.1, 3.1, 2.1):
        ax.annotate(
            "",
            xy=(2.5, y),
            xytext=(2.15, 3.95),
            arrowprops=dict(arrowstyle="->", color="#444", lw=0.9),
        )
    ax.annotate("", xy=(4.55, 3.0), xytext=(4.15, 3.1), arrowprops=dict(arrowstyle="->", color="#444", lw=0.9))
    ax.annotate("", xy=(7.0, 3.1), xytext=(6.6, 3.1), arrowprops=dict(arrowstyle="->", color="#444", lw=0.9))

    ax.text(
        5.0,
        0.55,
        "This work: monotone quality routing; masked partial-label multi-task head;\n"
        "endpoint-aware public cohorts; calibration/DCA as evaluation framework\n"
        "(Yellow/green = extensions; blue backbones = shared Reti-Pioneer skeleton)",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    ax.set_title("Figure 1. Method overview (architecture schematic)", fontsize=11, pad=6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def png_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def write_uri_sidecar(paths: list[Path], out_json: Path) -> None:
    payload = {p.name: png_to_data_uri(p) for p in paths if p.suffix == ".png"}
    out_json.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "results" / "ablation_summary.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "paper_assets",
    )
    args = ap.parse_args()
    _setup_style()
    rows = _read_ablation(args.csv)
    if not rows:
        raise SystemExit(f"No OK rows in {args.csv}")

    outs = [
        plot_architecture_schematic(args.out_dir / "fig1_architecture.png"),
        plot_ablation_bars(rows, args.out_dir / "fig2_ablation_bars.png"),
        plot_calibration_effect(rows, args.out_dir / "fig3_calibration.png"),
        plot_cross_domain(rows, args.out_dir / "fig4_cross_domain.png"),
    ]
    # Mirror into results/figures
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for p in outs:
        target = fig_dir / p.name
        target.write_bytes(p.read_bytes())
        pdf = p.with_suffix(".pdf")
        if pdf.exists():
            (fig_dir / pdf.name).write_bytes(pdf.read_bytes())

    write_uri_sidecar(outs, args.out_dir / "embedded_png_uris.json")
    print("Wrote:")
    for p in outs:
        print(" ", p)
    print(" ", args.out_dir / "embedded_png_uris.json")


if __name__ == "__main__":
    main()
