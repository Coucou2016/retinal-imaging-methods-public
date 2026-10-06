#!/usr/bin/env python
"""Intuitive SciencePlots figures from real RFMiD runs (curves first, then tables)."""

from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import argparse
import base64
import csv
import json
import sys
import types
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_lzma_shim() -> None:
    try:
        import lzma as z  # noqa: F401

        if hasattr(z, "open"):
            return
    except Exception:
        pass
    lz = types.ModuleType("lzma")
    lz.FORMAT_XZ = 1
    lz.FORMAT_ALONE = 2
    lz.FORMAT_RAW = 3
    lz.CHECK_NONE = 0
    lz.CHECK_CRC32 = 1
    lz.CHECK_CRC64 = 2
    lz.CHECK_SHA256 = 3

    class LZMAError(Exception):
        pass

    lz.LZMAError = LZMAError
    lz.open = lambda *a, **k: (_ for _ in ()).throw(LZMAError("no"))  # type: ignore
    sys.modules["lzma"] = lz


_ensure_lzma_shim()


def _setup_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex", "grid"])
    except Exception:
        mpl.rcParams.update({"axes.grid": True, "grid.alpha": 0.28})
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.facecolor"] = "white"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {path.with_suffix('.png')}")
    return path.with_suffix(".png")


def collect_predictions(ckpt_run: Path, data_dir: Path, split: str = "test") -> dict:
    from UKBDataset import UKBDatasetFast
    from RetiPioneer import get_reti_pioneer, normalize_ensemble
    from evaluate import (
        collect_probs,
        load_run_meta,
        resolve_ckpt_path,
        resolve_eval_split,
        resolve_run_dir,
    )

    run_dir = resolve_run_dir(str(ckpt_run))
    meta = load_run_meta(run_dir)
    diseases = list(meta.get("diseases") or [])
    device = torch.device("cpu")
    model = get_reti_pioneer(
        True,
        num_classes=int(meta.get("num_classes", len(diseases) or 1)),
        learnable_q=bool(meta.get("learnable_q", False)),
        enable_q=bool(meta.get("enable_q", True)),
        ensemble=normalize_ensemble(meta.get("ensemble", "released_code")),
        quality_router=meta.get("quality_router", "fixed"),
        quality_gating=bool(meta.get("quality_gating", False)),
        quality_aux=False,
        lambda_q=float(meta.get("lambda_q", 0) or 0),
    )
    ckpt_path = resolve_ckpt_path(str(ckpt_run))
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)

    base_ds = UKBDatasetFast(
        str(data_dir),
        meta=["baselineage", "gender", "weight", "ethnicity"],
        y=0,
        disease=[0],
        use_pretrain=["RETF", "SwinB", "VimS"],
        incident_exclude_prior=False,
    )
    if not diseases:
        diseases = list(base_ds.disease_names)
    base_ds.set_target(0, diseases, incident_exclude_prior=False)
    eval_ds, split_name, _bundle = resolve_eval_split(run_dir, base_ds, split, 0.25, 42)
    probs, labels = collect_probs(model, eval_ds, device, batch_size=64)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
        labels = labels.reshape(-1, 1)
    # Quality vectors for the same fold
    ql = []
    for i in range(len(eval_ds)):
        _lr, _m, (q_l, _q_r), _y = eval_ds[i]
        ql.append(np.asarray(q_l, dtype=np.float32).reshape(-1))
    return {
        "probs": probs,
        "labels": labels,
        "ql": np.stack(ql),
        "diseases": diseases,
        "split": split_name,
    }


def plot_roc(pred: dict, out: Path) -> Path:
    _setup_style()
    diseases = pred["diseases"]
    y = pred["labels"]
    p = pred["probs"]
    focus = [d for d in ["Disease_Risk", "DR", "MH", "ARMD", "ODC", "MYA"] if d in diseases]
    if not focus:
        focus = diseases[:4]
    n = len(focus)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, name in zip(axes, focus):
        k = diseases.index(name)
        yt, yp = y[:, k], p[:, k]
        if len(np.unique(yt)) < 2:
            ax.set_title(f"{name} (undefined)")
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        ax.plot(fpr, tpr, color="#1f4e79", lw=1.8, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(name)
        ax.legend(loc="lower right", frameon=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes[len(focus) :]:
        ax.axis("off")
    fig.suptitle("RFMiD test ROC curves (real-pixel features)", y=1.02)
    fig.tight_layout()
    return _save(fig, out)


def plot_reliability(pred: dict, out: Path) -> Path:
    _setup_style()
    diseases = pred["diseases"]
    focus = [d for d in ["DR", "Disease_Risk", "MH"] if d in diseases][:3]
    fig, axes = plt.subplots(1, max(len(focus), 1), figsize=(3.4 * max(len(focus), 1), 3.2))
    if len(focus) <= 1:
        axes = [axes]
    for ax, name in zip(axes, focus):
        k = diseases.index(name)
        yt, yp = pred["labels"][:, k], pred["probs"][:, k]
        try:
            frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=8, strategy="uniform")
        except ValueError:
            ax.set_title(f"{name} (too few bins)")
            continue
        ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=0.8, label="Ideal")
        ax.plot(mean_pred, frac_pos, "o-", color="#c65911", lw=1.5, label=name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Reliability · {name}")
        ax.legend(frameon=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Calibration curves (RFMiD test)", y=1.03)
    fig.tight_layout()
    return _save(fig, out)


def plot_dca(metrics_json: Path, out: Path) -> Path:
    _setup_style()
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    curves = payload.get("dca_curves") or {}
    names = [n for n in ["DR", "Disease_Risk", "MH", "ARMD"] if n in curves] or list(curves)[:4]
    if not names:
        raise RuntimeError("no dca_curves in metrics json")
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.2))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        c = curves[name]
        thr = np.asarray(c["thresholds"])
        ax.plot(thr, c["net_benefit"], color="#1f4e79", lw=1.8, label="Model")
        ax.plot(thr, c["treat_all"], ls="--", color="#8faadc", lw=1.2, label="Treat all")
        ax.plot(thr, c["treat_none"], ls=":", color="0.4", lw=1.0, label="Treat none")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title(f"DCA · {name}")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Decision curves (RFMiD test)", y=1.03)
    fig.tight_layout()
    return _save(fig, out)


def plot_ablation_bars(summary_csv: Path, out: Path) -> Path:
    _setup_style()
    rows = []
    with summary_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok" and r.get("eval") == "rfmid_test":
                rows.append(r)
    if not rows:
        raise SystemExit(f"No ok rfmid_test rows in {summary_csv}")
    arms = [r["ablation"] for r in rows]
    vals = []
    yerr_lo, yerr_hi = [], []
    for r in rows:
        v = r.get("auroc_DR") or r.get("auroc") or "nan"
        try:
            vals.append(float(v))
        except ValueError:
            vals.append(float("nan"))
        try:
            lo = float(r.get("bootstrap_ci_low") or "nan")
            hi = float(r.get("bootstrap_ci_high") or "nan")
            est = float(r.get("bootstrap_auroc") or v)
            yerr_lo.append(max(0.0, est - lo) if np.isfinite(lo) else 0.0)
            yerr_hi.append(max(0.0, hi - est) if np.isfinite(hi) else 0.0)
        except ValueError:
            yerr_lo.append(0.0)
            yerr_hi.append(0.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(arms))
    ax.bar(
        x,
        vals,
        color="#1f4e79",
        edgecolor="0.2",
        lw=0.4,
        yerr=np.vstack([yerr_lo, yerr_hi]) if any(yerr_lo) or any(yerr_hi) else None,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("AUROC (DR head or macro)")
    ax.set_ylim(0, 1)
    ax.set_title("Ablation on real-pixel RFMiD test")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, min(0.98, v + 0.03), f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


def plot_quality_routing(out: Path) -> Path:
    _setup_style()
    from QualityAware import QualityAware
    from quality_gate import QualityBackboneRouter
    from intervention import canonical_quality_vectors

    qa = QualityAware(8, 4, enable_q=True, learnable_q=True, quality_router="monotone")
    vecs = canonical_quality_vectors()
    router_scores = {}
    with torch.no_grad():
        for name, v in vecs.items():
            router_scores[name] = float(
                qa.quality_scalar(torch.as_tensor(v, dtype=torch.float32).view(1, 3)).item()
            )
    router_bb = QualityBackboneRouter(n_backbones=3, enabled=True)
    router_bb.mlp = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(3, 3))
    with torch.no_grad():
        router_bb.mlp[-1].weight.copy_(
            torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
        )
        router_bb.mlp[-1].bias.zero_()
        bb = {
            name: router_bb(torch.as_tensor(v, dtype=torch.float32).view(1, 3))
            .numpy()
            .ravel()
            for name, v in vecs.items()
        }

    labels = ["good", "usable", "bad"]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1))
    axes[0].bar(labels, [router_scores[k] for k in labels], color="#2e7d32")
    axes[0].set_title("Monotone router scalar")
    axes[0].set_ylim(0, 1.05)
    for ax, key in zip(axes[1:], ["good", "bad"]):
        ax.pie(
            bb[key],
            labels=["RETF*", "Swin", "Vim*"],
            autopct="%1.0f%%",
            startangle=90,
            colors=["#1f4e79", "#8faadc", "#c65911"],
        )
        ax.set_title(f"E5 backbone mix · q={key}")
    fig.suptitle("Quality intervention: fusion & backbone routing", y=1.05)
    fig.tight_layout()
    return _save(fig, out)


def plot_fundus_examples(data_dir: Path, out: Path, n_per: int = 3) -> Path:
    _setup_style()
    from public_common import load_pairs_manifest

    mqd = np.load(data_dir / "UKB_mqd.npz", allow_pickle=True)
    ql = mqd["ql"]
    hard = ql.argmax(axis=1)
    pairs = load_pairs_manifest(data_dir / "pairs.csv")
    fig, axes = plt.subplots(3, n_per, figsize=(2.4 * n_per, 7.2))
    strata = ["good", "usable", "bad"]
    rng = np.random.default_rng(42)
    for row, name in enumerate(strata):
        idxs = np.where(hard == row)[0]
        if len(idxs) == 0:
            for col in range(n_per):
                axes[row, col].axis("off")
            continue
        pick = rng.choice(idxs, size=min(n_per, len(idxs)), replace=False)
        for col in range(n_per):
            ax = axes[row, col]
            if col >= len(pick):
                ax.axis("off")
                continue
            i = int(pick[col])
            img = Image.open(pairs[i][0]).convert("RGB")
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            conf = ql[i]
            ax.set_xlabel(f"q=[{conf[0]:.2f},{conf[1]:.2f},{conf[2]:.2f}]", fontsize=7)
            if col == 0:
                ax.set_ylabel(name, fontsize=11)
    fig.suptitle("RFMiD quality strata (pixel proxy → soft q)", y=0.995)
    fig.tight_layout()
    return _save(fig, out)


def plot_score_hist(pred: dict, out: Path) -> Path:
    _setup_style()
    diseases = pred["diseases"]
    name = "DR" if "DR" in diseases else diseases[0]
    k = diseases.index(name)
    yt, yp = pred["labels"][:, k], pred["probs"][:, k]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.hist(yp[yt < 0.5], bins=30, alpha=0.65, label="Negative", color="#8faadc", density=True)
    ax.hist(yp[yt >= 0.5], bins=30, alpha=0.65, label="Positive", color="#1f4e79", density=True)
    ax.set_xlabel(f"Predicted probability ({name})")
    ax.set_ylabel("Density")
    ax.set_title(f"Score distributions · {name} (RFMiD test)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, out)


def plot_quality_strata_counts(data_dir: Path, out: Path) -> Path:
    _setup_style()
    ql = np.load(data_dir / "UKB_mqd.npz")["ql"]
    hard = ql.argmax(axis=1)
    labels = ["good", "usable", "bad"]
    counts = [int((hard == i).sum()) for i in range(3)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].bar(labels, counts, color=["#2e7d32", "#f9a825", "#c62828"])
    axes[0].set_ylabel("N images")
    axes[0].set_title("RFMiD quality strata counts")
    axes[1].pie(counts, labels=labels, autopct="%1.0f%%", colors=["#2e7d32", "#f9a825", "#c62828"])
    axes[1].set_title("Share by argmax(q)")
    fig.tight_layout()
    return _save(fig, out)


def write_uri_sidecar(pngs: list[Path], out_json: Path) -> None:
    payload = {}
    for p in pngs:
        if p.is_file():
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            payload[p.name] = f"data:image/png;base64,{b64}"
    existing = {}
    if out_json.is_file():
        existing = json.loads(out_json.read_text(encoding="utf-8"))
    existing.update(payload)
    out_json.write_text(json.dumps(existing), encoding="utf-8")
    print(f"Wrote URI sidecar (+{len(payload)} keys)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "real_rfmid"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "rfmid"))
    parser.add_argument("--assets-dir", default=str(ROOT))
    parser.add_argument("--ckpt-run", default=None)
    args = parser.parse_args()

    results = Path(args.results_dir)
    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    pngs: list[Path] = []

    # Figures that only need quality/mqd (can run mid-extract after quality done)
    try:
        pngs.append(plot_fundus_examples(data_dir, assets / "fig5_fundus_quality"))
        pngs.append(plot_quality_strata_counts(data_dir, assets / "fig6b_quality_pie"))
    except Exception as exc:
        print(f"Fundus/quality plots deferred: {exc}")

    pngs.append(plot_quality_routing(assets / "fig6_quality_routing"))

    summary = results / "ablation_summary.csv"
    if summary.is_file():
        try:
            pngs.append(plot_ablation_bars(summary, assets / "fig2_ablation_bars"))
        except SystemExit as exc:
            print(exc)

    for name in ("E5", "E4", "E3", "E2", "E1", "E0"):
        cand = results / f"metrics_{name}_rfmid_test.json"
        if cand.is_file():
            try:
                pngs.append(plot_dca(cand, assets / "fig7_decision_curves"))
            except Exception as exc:
                print(f"DCA plot skipped: {exc}")
            break

    ckpt_run = args.ckpt_run
    manifest = results / "run_manifest.json"
    if ckpt_run is None and manifest.is_file():
        run_map = json.loads(manifest.read_text(encoding="utf-8")).get("run_map", {})
        ckpt_run = run_map.get("E5") or run_map.get("E4") or run_map.get("E3") or run_map.get("E2")

    if ckpt_run:
        try:
            pred = collect_predictions(Path(ckpt_run), data_dir, split="test")
            np.savez_compressed(
                results / "test_predictions.npz",
                probs=pred["probs"],
                labels=pred["labels"],
                ql=pred["ql"],
                diseases=np.array(pred["diseases"]),
            )
            pngs.append(plot_roc(pred, assets / "fig8_roc_curves"))
            pngs.append(plot_reliability(pred, assets / "fig3_calibration"))
            pngs.append(plot_score_hist(pred, assets / "fig9_score_hist"))
        except Exception as exc:
            print(f"Prediction plots skipped: {exc}")
            import traceback

            traceback.print_exc()

    write_uri_sidecar([p for p in pngs if p is not None], assets / "embedded_png_uris.json")
    fig_dir = results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for p in pngs:
        if p is not None and p.is_file():
            (fig_dir / p.name).write_bytes(p.read_bytes())
            pdf = p.with_suffix(".pdf")
            if pdf.is_file():
                (fig_dir / pdf.name).write_bytes(pdf.read_bytes())


if __name__ == "__main__":
    main()
