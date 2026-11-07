#!/usr/bin/env python3
"""Merge every statistics artefact into one numbers file the manuscript build reads.

Inputs (all produced by scripts, none typed by hand)
    stats/headline.json                 AUROC, AP, ECE, Brier per arm (raw scores)
    stats/paired.json                   paired bootstrap ΔAUROC
    stats/per_head.json                 per-endpoint metrics per arm
    stats/quality_strata.json           metrics by soft quality band
    stats/router.json                   router type and monotone weights
    stats/calibration_fit.json          temperature fitted on the 64-image cal fold
    stats/headline_calibrated.json      metrics before/after that temperature
    stats/temperature_schemes.json      none / single / per-endpoint calibration
    stats/operating_points_cal.json     Youden and 95%-specificity with val thresholds
    stats/operating_point_analysis.json frozen and matched sensitivity with bootstrap
    stats/threshold_transfer.json       threshold drift diagnostics
    stats/decision_curves.json          net benefit curves
    epoch_selection.json                reporting epoch per arm

Output: ``results/real_rfmid/analysis_sel/manuscript_numbers.json``

Usage:
    python scripts/export_manuscript_numbers.py
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ANALYSIS = Path(os.environ.get("ARM_ANALYSIS_DIR", ROOT))
STATS = Path(os.environ.get("ARM_STATS_DIR", ROOT))


def r(x, n=4):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def load(name: str):
    p = ROOT / name
    if not p.is_file():
        raise FileNotFoundError(f"missing statistic file {p}; run the pipeline first")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    headline = load("headline.json")
    paired = load("paired.json")
    per_head = load("per_head.json")
    strata = load("quality_strata.json")
    router = load("router.json")
    calib = load("calibration_fit.json")
    cal_head = load("headline_calibrated.json")
    schemes = load("temperature_schemes.json")
    oracle = load("calibration_oracle.json")
    ops = load("operating_points_cal.json")
    op_analysis = load("operating_point_analysis.json")
    transfer = load("threshold_transfer.json")
    dca = load("decision_curves.json")
    selection = json.loads((ANALYSIS / "epoch_selection.json").read_text(encoding="utf-8"))

    # optional: newer analyses, included when the scripts have been run
    def maybe(name):
        p = (ROOT / name).resolve() if name.startswith("..") else ROOT / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    extended = maybe("extended.json")
    holdout = maybe("recalibration_holdout.json")
    followup = maybe("recalibration_followup.json")
    # seed replicates: optional, merged when the replicate pipeline has run
    seed_repl = maybe("../../analysis_seed/seed_replication.json")
    seed_cal = maybe("../../analysis_seed/calibration_replication.json")
    seed_ep = maybe("../../analysis_seed/seed_endpoint_analysis.json")
    seed_tt = maybe("../../analysis_seed/threshold_transfer_replication.json")

    out: dict = {
        "arms": {},
        "paired": paired,
        "per_head": per_head,
        "strata": strata,
        "router": router,
        "calibration_fit": calib,
        "calibration_schemes": schemes,
        "calibration_oracle": oracle,
        "operating_points": ops,
        "operating_point_analysis": op_analysis,
        "threshold_transfer": transfer,
        "decision_curves": dca,
        "epoch_selection": {
            k: {
                "epoch": v["selected_epoch"],
                "val_macro_auroc": r(v["selected_val_macro_auroc"]),
                "n_epochs": v["n_epochs_scored"],
            }
            for k, v in selection.items()
        },
        "extended": extended,
        "recalibration_holdout": holdout,
        "recalibration_followup": followup,
        "seed_replication": seed_repl,
        "calibration_replication": seed_cal,
        "seed_endpoint_analysis": seed_ep,
        "threshold_transfer_replication": seed_tt,
    }

    for arm, row in headline.items():
        cal = calib.get(arm, {})
        post = cal_head.get(arm, {})
        scheme = schemes.get(arm, {})
        out["arms"][arm] = {
            "n": row["n"],
            "k": row["k"],
            "macro_auroc": r(row["macro_auroc"]),
            "macro_auroc_ci": [r(row["bootstrap"]["ci_low"]), r(row["bootstrap"]["ci_high"])],
            "macro_auroc_delong": [r(row["delong"]["ci_low"]), r(row["delong"]["ci_high"])],
            "macro_ap": r(row["macro_ap"]),
            "dr_auroc": r(row["dr_auroc"]),
            "dr_ap": r(row["dr_ap"]),
            "pooled_ece": r(row["pooled_ece"]),
            "mean_head_ece": r(row["mean_head_ece"]),
            "pooled_brier": r(row["pooled_brier"]),
            "temperature_cal64": r(cal.get("temperature")),
            "temperature_val640": r(scheme.get("temperature_single")),
            "ece_on_cal_raw": r(cal.get("ece_on_cal_raw")),
            "ece_on_cal_post": r(cal.get("ece_on_cal_post")),
            "pooled_ece_after_T": r(post.get("pooled_ece_cal")),
            "mean_head_ece_after_T": r(post.get("mean_head_ece_cal")),
            "brier_after_T": r(post.get("brier_cal")),
            "mean_head_ece_per_head_T": r(scheme.get("per_head", {}).get("mean_head_ece")),
            "worst_head_ece": r(scheme.get("none", {}).get("worst_head_ece")),
            "worst_head_ece_after_per_head_T": r(scheme.get("per_head", {}).get("worst_head_ece")),
            "mean_head_ece_oracle": r(oracle.get(arm, {}).get("mean_head_ece_oracle")),
            "mean_head_ece_oracle_gain": r(oracle.get(arm, {}).get("mean_head_ece_gain")),
            "holdout_ece_none": r(
                (holdout or {}).get("per_arm", {}).get(arm, {}).get("summary", {}).get("macro_ece_none")
            ),
            "holdout_ece_temperature": r(
                (holdout or {})
                .get("per_arm", {})
                .get(arm, {})
                .get("summary", {})
                .get("macro_ece_temperature")
            ),
            "holdout_ece_platt": r(
                (holdout or {})
                .get("per_arm", {})
                .get(arm, {})
                .get("summary", {}).get("macro_ece_platt")
            ),
            "holdout_ece_isotonic": r(
                (holdout or {})
                .get("per_arm", {})
                .get(arm, {})
                .get("summary", {}).get("macro_ece_isotonic")
            ),
            "holdout_brier_platt": r(
                (holdout or {})
                .get("per_arm", {})
                .get(arm, {})
                .get("summary", {}).get("macro_brier_platt")
            ),
            "holdout_shared_platt_macro_ece": r(
                (followup or {}).get("arms", {}).get(arm, {}).get("shared_map", {}).get("macro_ece_shared")
            ),
            "holdout_shared_platt_pooled_ece": r(
                (followup or {}).get("arms", {}).get(arm, {}).get("shared_map", {}).get("pooled_ece_shared")
            ),
            "equal_mass_ece": r(
                (extended or {}).get("adaptive_ece", {}).get(arm, {}).get("mean_head_ece_equal_mass")
            ),
            "seed_macro_mean": r(
                ((seed_repl or {}).get("per_arm", {}).get(arm, {}) or {}).get("mean")
            ),
            "seed_macro_sd": r(
                ((seed_repl or {}).get("per_arm", {}).get(arm, {}) or {}).get("sd")
            ),
            "seed_macro_min": r(
                ((seed_repl or {}).get("per_arm", {}).get(arm, {}) or {}).get("min")
            ),
            "seed_macro_max": r(
                ((seed_repl or {}).get("per_arm", {}).get(arm, {}) or {}).get("max")
            ),
            "seed_n": ((seed_repl or {}).get("per_arm", {}).get(arm, {}) or {}).get("n_seeds"),
            "seed_ece_identity": r(
                ((seed_cal or {}).get("per_arm", {}).get(arm, {}) or {})
                .get("aggregate", {})
                .get("ece_identity_mean")
            ),
            "seed_ece_temperature": r(
                ((seed_cal or {}).get("per_arm", {}).get(arm, {}) or {})
                .get("aggregate", {})
                .get("ece_temperature_global_mean")
            ),
            "seed_ece_prevalence_shift": r(
                ((seed_cal or {}).get("per_arm", {}).get(arm, {}) or {})
                .get("aggregate", {})
                .get("ece_prevalence_shift_mean")
            ),
        }

    # prevalence and quality counts from the cached labels
    d = np.load(ANALYSIS / "arm_E5.npz", allow_pickle=True)
    names = [str(x) for x in d["names"]]
    out["prevalence"] = {ep: r(d["labels"][:, k].mean(), 3) for k, ep in enumerate(names)}
    q = 0.5 * (d["ql"] + d["qr"])
    band = q.argmax(axis=1)
    out["quality_counts"] = {
        "good": int((band == 0).sum()),
        "usable": int((band == 1).sum()),
        "bad": int((band == 2).sum()),
        "mean_good": r(q[:, 0].mean(), 3),
        "mean_usable": r(q[:, 1].mean(), 3),
        "mean_bad": r(q[:, 2].mean(), 3),
    }
    out["endpoint_names"] = names

    dst = ANALYSIS / "manuscript_numbers.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", dst)
    for arm, row in out["arms"].items():
        print(
            f"  {arm}: macro {row['macro_auroc']} {row['macro_auroc_ci']} DR {row['dr_auroc']} "
            f"T64={row['temperature_cal64']} ECE {row['mean_head_ece']}->{row['mean_head_ece_after_T']}"
        )


if __name__ == "__main__":
    main()
