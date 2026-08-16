#!/usr/bin/env python
"""Run the 4 ablation configs on tiny synthetic public caches and summarize metrics.

SYNTHETIC / CI only — do not put these AUROCs in a manuscript.
Works on CPU feature-cache path; fails soft if a train/eval subprocess errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from reti_pioneer.data_paths import ukb_compressed_ready

ABLATIONS: list[dict[str, Any]] = [
    {
        "name": "baseline",
        "config": "ablation_baseline.yaml",
        "multitask": False,
        "learnable_q": False,
        "tag": "D",
    },
    {
        "name": "learnq",
        "config": "ablation_learnq.yaml",
        "multitask": False,
        "learnable_q": True,
        "tag": "D",
    },
    {
        "name": "multitask",
        "config": "ablation_multitask.yaml",
        "multitask": True,
        "learnable_q": False,
        "tag": "multitask",
    },
    {
        "name": "full",
        "config": "ablation_full.yaml",
        "multitask": True,
        "learnable_q": True,
        "tag": "multitask",
    },
]

DISCLAIMER = (
    "SYNTHETIC FEATURE CACHE - not for manuscript AUROC. "
    "Replace with real ODIR/BRSET images + extract_features.py before paper tables."
)


def ensure_synthetic_cache(dataset: str, out_dir: str, n_samples: int, seed: int, force: bool) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if not force and ukb_compressed_ready(out_dir):
        print(f"[cache] {dataset}: reuse {out_dir}")
        return out_dir
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "prepare_public_npz.py"),
        "--dataset",
        dataset,
        "--out",
        out_dir,
        "--synthetic-demo",
        "--n-samples",
        str(n_samples),
        "--force",
    ]
    print(f"[cache] building synthetic {dataset} -> {out_dir} (n={n_samples})")
    subprocess.check_call(cmd, cwd=ROOT)
    return out_dir


def find_latest_run(
    ckpt_root: str,
    tag: str,
    *,
    learnable_q: bool | None = None,
) -> str | None:
    """Pick newest ckpt/<timestamp>/<tag>/y0 under ckpt_root.

    When learnable_q is set, require matching run_meta.json so multitask vs full
    (same tag folder) are not confused.
    """
    if not os.path.isdir(ckpt_root):
        return None
    candidates: list[tuple[float, str]] = []
    for ts in os.listdir(ckpt_root):
        run = os.path.join(ckpt_root, ts, tag, "y0")
        meta = os.path.join(run, "run_meta.json")
        ckpt_dir = os.path.join(run, "ckpt")
        if not (os.path.isfile(meta) and os.path.isdir(ckpt_dir)):
            continue
        pts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if not pts:
            continue
        if learnable_q is not None:
            try:
                with open(meta, encoding="utf-8") as f:
                    meta_j = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if bool(meta_j.get("learnable_q", False)) != bool(learnable_q):
                continue
        candidates.append((os.path.getmtime(run), run))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def run_cmd(cmd: list[str], soft: bool = True) -> tuple[bool, str]:
    print("+", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except OSError as exc:
        msg = f"spawn failed: {exc}"
        print(f"  FAIL soft: {msg}")
        return False, msg
    out = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        tail = out[-4000:] if out else "(no output)"
        msg = f"exit={completed.returncode}\n{tail}"
        if soft:
            print(f"  FAIL soft: exit={completed.returncode}")
            print(tail[-1500:])
            return False, msg
        raise RuntimeError(msg)
    return True, out


def train_ablation(
    abl: dict[str, Any],
    *,
    odir_dir: str,
    ckpt_dir: str,
    quick: bool,
    soft: bool,
) -> tuple[bool, str | None]:
    cfg = os.path.join(ROOT, "configs", abl["config"])
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "train.py"),
        "--config",
        cfg,
        "--dataset",
        "odir",
        "--data-dir",
        odir_dir,
        "--horizon",
        "0",
    ]
    if quick:
        cmd.append("--demo")
    if abl["multitask"]:
        cmd.append("--multitask")
    else:
        cmd.extend(["--disease", "D"])
    if abl["learnable_q"]:
        cmd.append("--learnable-q")

    # Isolate runs under results/ablation_runs/... via ckpt_dir override in config:
    # train.py reads ckpt_dir from yaml; we pass a temp yaml overlay via env not available,
    # so copy path by writing a small override config.
    override = os.path.join(ckpt_dir, f"_cfg_{abl['name']}.yaml")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(cfg, encoding="utf-8") as f:
        body = f.read()
    # Replace ckpt_dir and shrink demo n_samples for quick CI
    lines = []
    for line in body.splitlines():
        if line.startswith("ckpt_dir:"):
            lines.append(f"ckpt_dir: {ckpt_dir.replace(os.sep, '/')}")
        elif line.startswith("data_dir:"):
            lines.append(f"data_dir: {odir_dir.replace(os.sep, '/')}")
        else:
            lines.append(line)
    if quick:
        # Ensure short epochs even if config was edited
        rewritten: list[str] = []
        in_demo = False
        for line in lines:
            if line.startswith("demo:"):
                in_demo = True
                rewritten.append(line)
                continue
            if in_demo and line.startswith("  n_samples:"):
                rewritten.append("  n_samples: 48")
                continue
            if in_demo and line and not line.startswith(" ") and not line.startswith("\t"):
                in_demo = False
            if line.strip().startswith("epochs:") and "warmup" not in line:
                rewritten.append("  epochs: 2" if line.startswith("  ") else "epochs: 2")
                continue
            if "warmup_epochs:" in line:
                rewritten.append("  warmup_epochs: 2" if line.startswith("  ") else "warmup_epochs: 2")
                continue
            rewritten.append(line)
        lines = rewritten
    with open(override, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    cmd[cmd.index("--config") + 1] = override

    ok, log = run_cmd(cmd, soft=soft)
    if not ok:
        return False, None
    run = find_latest_run(ckpt_dir, abl["tag"], learnable_q=bool(abl["learnable_q"]))
    if run is None:
        print(f"  WARN: no checkpoint for tag={abl['tag']} learnable_q={abl['learnable_q']} under {ckpt_dir}")
        return False, None
    return True, run


def eval_run(
    run_dir: str,
    *,
    data_dir: str,
    dataset: str,
    out_json: str,
    calibrate: bool = True,
    test_data_dir: str | None = None,
    test_dataset: str | None = None,
    soft: bool = True,
) -> dict[str, Any] | None:
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "evaluate.py"),
        "--ckpt",
        run_dir,
        "--dataset",
        dataset,
        "--data-dir",
        data_dir,
        "--split",
        "val" if test_data_dir is None else "all",
        "--out-json",
        out_json,
    ]
    if calibrate:
        cmd.append("--calibrate")
    if test_data_dir and test_dataset:
        cmd.extend(
            [
                "--test-data-dir",
                test_data_dir,
                "--test-dataset",
                test_dataset,
            ]
        )
    ok, _ = run_cmd(cmd, soft=soft)
    if not ok or not os.path.isfile(out_json):
        return None
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def _comparable_odir_d_auroc(raw: dict[str, Any]) -> Any:
    """Prefer ODIR D head AUROC so single-task and multitask rows are comparable."""
    if not raw:
        return ""
    for key in ("class_D_auroc", "class_1_auroc"):
        if key in raw and raw[key] != "" and raw[key] is not None:
            return raw[key]
    # Single-task D runs only expose top-level auroc.
    return raw.get("auroc", "")


def _metric_row(name: str, kind: str, payload: dict[str, Any] | None, error: str = "") -> dict[str, Any]:
    raw = (payload or {}).get("raw") or {}
    cal = (payload or {}).get("calibrated") or {}
    return {
        "ablation": name,
        "eval": kind,
        "status": "ok" if payload else "failed",
        "auroc": raw.get("auroc", ""),
        "auroc_D": _comparable_odir_d_auroc(raw),
        "ap": raw.get("ap", ""),
        "ece": raw.get("ece", ""),
        "brier": raw.get("brier", ""),
        "nb@0.10": raw.get("nb@0.10", ""),
        "treat_all_nb@0.10": raw.get("treat_all_nb@0.10", ""),
        "cal_auroc": cal.get("auroc", ""),
        "cal_ece": cal.get("ece", ""),
        "temperature": (payload or {}).get("temperature", ""),
        "n": (payload or {}).get("n", ""),
        "error": error,
        "disclaimer": DISCLAIMER,
        "metric_note": "auroc=macro for multitask; auroc_D=ODIR D head (comparable across arms)",
    }


def write_summary(rows: list[dict[str, Any]], out_csv: str, out_md: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fields = [
        "ablation",
        "eval",
        "status",
        "auroc",
        "auroc_D",
        "ap",
        "ece",
        "brier",
        "nb@0.10",
        "treat_all_nb@0.10",
        "cal_auroc",
        "cal_ece",
        "temperature",
        "n",
        "error",
        "disclaimer",
        "metric_note",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    lines = [
        "# Ablation summary (SYNTHETIC)",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "Primary comparable column: **auroc_D** (ODIR D head). `auroc` is macro-K for multitask arms.",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Ablation | Eval | Status | AUROC | AUROC_D | AP | ECE | Brier | NB@0.10 | Treat-all | Cal AUROC | Cal ECE | T | n |",
        "|----------|------|--------|-------|---------|----|-----|-------|---------|-----------|-----------|---------|---|---|",
    ]

    def fmt(v: Any) -> str:
        if v == "" or v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    for row in rows:
        lines.append(
            "| {ablation} | {eval} | {status} | {auroc} | {auroc_D} | {ap} | {ece} | {brier} | {nb} | {ta} | {cal_auroc} | {cal_ece} | {T} | {n} |".format(
                ablation=row["ablation"],
                eval=row["eval"],
                status=row["status"],
                auroc=fmt(row["auroc"]),
                auroc_D=fmt(row.get("auroc_D", "")),
                ap=fmt(row["ap"]),
                ece=fmt(row["ece"]),
                brier=fmt(row["brier"]),
                nb=fmt(row["nb@0.10"]),
                ta=fmt(row.get("treat_all_nb@0.10", "")),
                cal_auroc=fmt(row["cal_auroc"]),
                cal_ece=fmt(row["cal_ece"]),
                T=fmt(row["temperature"]),
                n=row.get("n") or "—",
            )
        )
    lines.extend(["", f"CSV: `{out_csv}`", ""])
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CI ablation matrix on synthetic ODIR/BRSET caches")
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Short epochs + tiny n via train --demo (default; CI path)",
    )
    parser.add_argument(
        "--no-quick",
        action="store_false",
        dest="quick",
        help="Use config epoch counts (still synthetic unless real caches exist)",
    )
    parser.add_argument("--n-samples", type=int, default=None, help="Synthetic cache size")
    parser.add_argument("--odir-dir", default=os.path.join(ROOT, "data", "odir"))
    parser.add_argument("--brset-dir", default=os.path.join(ROOT, "data", "brset"))
    parser.add_argument("--ckpt-dir", default=os.path.join(ROOT, "results", "ablation_runs"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--no-cross", action="store_true", help="Skip ODIR→BRSET cross eval")
    parser.add_argument("--force-cache", action="store_true", help="Rebuild synthetic caches")
    parser.add_argument("--strict", action="store_true", help="Raise on first train/eval failure")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated ablation names (baseline,learnq,multitask,full)",
    )
    args = parser.parse_args()

    quick = bool(args.quick)
    n_samples = args.n_samples if args.n_samples is not None else (48 if quick else 128)
    soft = not args.strict
    odir_dir = os.path.abspath(args.odir_dir)
    brset_dir = os.path.abspath(args.brset_dir)
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    selected = ABLATIONS
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        selected = [a for a in ABLATIONS if a["name"] in want]
        if not selected:
            raise SystemExit(f"No ablations matched --only {args.only!r}")

    print(DISCLAIMER)
    try:
        ensure_synthetic_cache("odir", odir_dir, n_samples, seed=42, force=args.force_cache)
        ensure_synthetic_cache("brset", brset_dir, n_samples, seed=43, force=args.force_cache)
    except Exception as exc:
        print(f"Cache build failed (soft): {exc}")
        if args.strict:
            raise
        write_summary(
            [_metric_row("all", "cache", None, error=str(exc))],
            os.path.join(out_dir, "ablation_summary.csv"),
            os.path.join(out_dir, "ablation_summary.md"),
        )
        return

    rows: list[dict[str, Any]] = []
    for abl in selected:
        print(f"\n=== Ablation: {abl['name']} ===")
        ok, run = train_ablation(
            abl, odir_dir=odir_dir, ckpt_dir=ckpt_dir, quick=quick, soft=soft
        )
        if not ok or not run:
            rows.append(_metric_row(abl["name"], "odir_val", None, error="train_failed"))
            if not args.no_cross:
                rows.append(_metric_row(abl["name"], "odir_to_brset", None, error="train_failed"))
            continue

        json_path = os.path.join(out_dir, f"metrics_{abl['name']}_odir_val.json")
        payload = eval_run(
            run,
            data_dir=odir_dir,
            dataset="odir",
            out_json=json_path,
            calibrate=True,
            soft=soft,
        )
        rows.append(_metric_row(abl["name"], "odir_val", payload, error="" if payload else "eval_failed"))

        if not args.no_cross:
            xjson = os.path.join(out_dir, f"metrics_{abl['name']}_odir_to_brset.json")
            xpayload = eval_run(
                run,
                data_dir=odir_dir,
                dataset="odir",
                out_json=xjson,
                calibrate=True,
                test_data_dir=brset_dir,
                test_dataset="brset",
                soft=soft,
            )
            rows.append(
                _metric_row(
                    abl["name"],
                    "odir_to_brset",
                    xpayload,
                    error="" if xpayload else "cross_eval_failed",
                )
            )

    write_summary(
        rows,
        os.path.join(out_dir, "ablation_summary.csv"),
        os.path.join(out_dir, "ablation_summary.md"),
    )
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"Done: {n_ok}/{len(rows)} eval rows ok (SYNTHETIC).")


if __name__ == "__main__":
    main()
