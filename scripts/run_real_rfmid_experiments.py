#!/usr/bin/env python
"""Train E0–E5 + evaluate on real-pixel RFMiD cache; write results/real_rfmid.

Requires ``data/rfmid`` without SYNTHETIC_FEATURES.txt (see build_real_rfmid_cache.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ABLATIONS = [
    {
        "name": "E0",
        "config": "ablation_e0.yaml",
        "multitask": False,
        "learnable_q": False,
        "disease": "DR",
        "tag": "DR",
    },
    {
        "name": "E1",
        "config": "ablation_e1.yaml",
        "multitask": False,
        "learnable_q": True,
        "disease": "DR",
        "tag": "DR",
    },
    {
        "name": "E2",
        "config": "ablation_e2.yaml",
        "multitask": True,
        "learnable_q": False,
        "disease": None,
        "tag": "multitask",
    },
    {
        "name": "E3",
        "config": "ablation_e3.yaml",
        "multitask": True,
        "learnable_q": True,
        "disease": None,
        "tag": "multitask",
    },
    {
        "name": "E4",
        "config": "ablation_e4.yaml",
        "multitask": True,
        "learnable_q": True,
        "disease": None,
        "tag": "multitask",
    },
    {
        "name": "E5",
        "config": "ablation_e5.yaml",
        "multitask": True,
        "learnable_q": True,
        "disease": None,
        "tag": "multitask",
    },
]

DISCLAIMER = (
    "REAL-PIXEL RFMiD features (ImageNet foundation surrogates: Swin-V2-B + "
    "ViT-B→1024 + ViT-S/384). Not label-derived SYNTHETIC. Not RETFound/Vim UKB "
    "replication (RETFound HF gated; Vim unavailable)."
)


def run_cmd(cmd: list[str]) -> tuple[bool, str]:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )
    out = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        print(out[-4000:])
        return False, out
    print(out[-2000:])
    return True, out


def write_override_cfg(src: Path, dst: Path, data_dir: Path, ckpt_dir: Path, epochs: int) -> None:
    lines = []
    in_training = False
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("data_dir:"):
            lines.append(f"data_dir: {data_dir.as_posix()}")
            continue
        if line.startswith("ckpt_dir:"):
            lines.append(f"ckpt_dir: {ckpt_dir.as_posix()}")
            continue
        if line.startswith("training:"):
            in_training = True
            lines.append(line)
            continue
        if in_training and line.strip().startswith("epochs:") and "warmup" not in line:
            lines.append(f"  epochs: {epochs}")
            continue
        if in_training and line and not line.startswith(" ") and not line.startswith("\t"):
            in_training = False
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_latest_run(ckpt_root: Path, tag: str, learnable_q: bool) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    if not ckpt_root.is_dir():
        return None
    for ts in ckpt_root.iterdir():
        run = ts / tag / "y0"
        meta = run / "run_meta.json"
        ckpt = run / "ckpt"
        if not (meta.is_file() and ckpt.is_dir()):
            continue
        if not any(ckpt.glob("*.pt")):
            continue
        meta_j = json.loads(meta.read_text(encoding="utf-8"))
        if bool(meta_j.get("learnable_q", False)) != bool(learnable_q):
            continue
        # Prefer quality_gating match for E5 vs E3/E4
        candidates.append((run.stat().st_mtime, run))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_e5_run(ckpt_root: Path) -> Path | None:
    best = None
    best_mtime = -1.0
    if not ckpt_root.is_dir():
        return None
    for ts in ckpt_root.iterdir():
        run = ts / "multitask" / "y0"
        meta = run / "run_meta.json"
        if not meta.is_file():
            continue
        meta_j = json.loads(meta.read_text(encoding="utf-8"))
        if not meta_j.get("quality_gating"):
            continue
        mtime = run.stat().st_mtime
        if mtime > best_mtime:
            best_mtime = mtime
            best = run
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "rfmid"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "real_rfmid"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--arms", nargs="+", default=[a["name"] for a in ABLATIONS])
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    marker = data_dir / "SYNTHETIC_FEATURES.txt"
    if marker.is_file():
        raise SystemExit(
            f"Refusing to train clinical real runs: {marker} still present. "
            "Run scripts/build_real_rfmid_cache.py first."
        )
    prov = data_dir / "FEATURE_PROVENANCE.json"
    if not prov.is_file():
        raise SystemExit(f"Missing {prov}; run build_real_rfmid_cache.py first.")

    rows = []
    run_map: dict[str, str] = {}
    for abl in ABLATIONS:
        if abl["name"] not in args.arms:
            continue
        override = ckpt_dir / f"_cfg_{abl['name']}.yaml"
        write_override_cfg(
            ROOT / "configs" / abl["config"],
            override,
            data_dir,
            ckpt_dir,
            args.epochs,
        )
        if not args.skip_train:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "train.py"),
                "--config",
                str(override),
                "--dataset",
                "rfmid",
                "--data-dir",
                str(data_dir),
                "--horizon",
                "0",
            ]
            if abl["multitask"]:
                cmd.append("--multitask")
            else:
                cmd.extend(["--disease", abl["disease"]])
            if abl["learnable_q"]:
                cmd.append("--learnable-q")
            ok, log = run_cmd(cmd)
            (out_dir / f"train_{abl['name']}.log").write_text(log, encoding="utf-8")
            if not ok:
                rows.append(
                    {
                        "ablation": abl["name"],
                        "eval": "rfmid_test",
                        "status": "train_failed",
                        "disclaimer": DISCLAIMER,
                    }
                )
                continue

        if abl["name"] == "E5":
            run = find_e5_run(ckpt_dir)
        else:
            run = find_latest_run(ckpt_dir, abl["tag"], abl["learnable_q"])
            # Disambiguate E3 vs E4: E4 has calibrate true in meta? Not stored.
            # Prefer newest matching learnable_q + multitask; E3/E4 both match —
            # train sequentially so newest after E4 train is E4; E5 filtered by gating.
        if run is None:
            rows.append(
                {
                    "ablation": abl["name"],
                    "eval": "rfmid_test",
                    "status": "no_ckpt",
                    "disclaimer": DISCLAIMER,
                }
            )
            continue
        run_map[abl["name"]] = str(run)

        for split_name in ("val", "test"):
            out_json = out_dir / f"metrics_{abl['name']}_rfmid_{split_name}.json"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate.py"),
                "--ckpt",
                str(run),
                "--dataset",
                "rfmid",
                "--data-dir",
                str(data_dir),
                "--split",
                split_name,
                "--calibrate",
                "--bootstrap",
                str(args.bootstrap),
                "--out-json",
                str(out_json),
            ]
            ok, log = run_cmd(cmd)
            (out_dir / f"eval_{abl['name']}_{split_name}.log").write_text(log, encoding="utf-8")
            if not ok or not out_json.is_file():
                rows.append(
                    {
                        "ablation": abl["name"],
                        "eval": f"rfmid_{split_name}",
                        "status": "eval_failed",
                        "disclaimer": DISCLAIMER,
                    }
                )
                continue
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            raw = payload.get("raw") or {}
            cal = payload.get("calibrated") or {}
            boot = payload.get("bootstrap_auroc") or {}
            # Prefer DR head when present for comparable column
            auroc_dr = raw.get("class_DR_auroc", raw.get("auroc"))
            row = {
                "ablation": abl["name"],
                "eval": f"rfmid_{split_name}",
                "status": "ok",
                "auroc": raw.get("auroc", ""),
                "auroc_DR": auroc_dr if auroc_dr is not None else "",
                "ap": raw.get("ap", ""),
                "ece": raw.get("ece", ""),
                "brier": raw.get("brier", ""),
                "cal_auroc": cal.get("auroc", ""),
                "cal_ece": cal.get("ece", ""),
                "temperature": payload.get("temperature", ""),
                "n": payload.get("n", ""),
                "bootstrap_auroc": boot.get("estimate", ""),
                "bootstrap_ci_low": boot.get("ci_low", ""),
                "bootstrap_ci_high": boot.get("ci_high", ""),
                "run_dir": str(run),
                "disclaimer": DISCLAIMER,
            }
            rows.append(row)
            # Keep a copy of probs for plotting if present
            for key in ("probs", "labels", "logits"):
                if key in payload:
                    npz_side = out_dir / f"arrays_{abl['name']}_{split_name}.json"
                    # arrays may be huge; evaluate may not dump them — ignore

    csv_path = out_dir / "ablation_summary.csv"
    fields = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "provenance": json.loads(prov.read_text(encoding="utf-8")),
        "run_map": run_map,
        "epochs": args.epochs,
        "bootstrap": args.bootstrap,
        "disclaimer": DISCLAIMER,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Runs: {run_map}")


if __name__ == "__main__":
    main()
