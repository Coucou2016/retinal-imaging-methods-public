#!/usr/bin/env python
"""Repeat every ablation arm under additional random seeds.

The main analysis trains each arm once. Paired bootstrap intervals then cover
test-set sampling noise only, so a reader cannot tell whether a +0.010 macro
delta is a property of the design or of one lucky initialisation. This script
keeps the frozen RFMiD split, calibration carve and test fold untouched, changes
only ``split.torch_seed`` (weight initialisation and batch order), trains the
same arms, picks the reporting epoch by validation macro AUROC with the same
rule, and stores per-sample test predictions next to the main cache.

Outputs
-------
results/real_rfmid/analysis_seed/arm_<ARM>_s<SEED>.npz     predictions
results/real_rfmid/analysis_seed/replicate_manifest.json   run dirs + epochs

Usage
-----
    python scripts/run_seed_replicates.py --seeds 43 44 --jobs 3
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
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "results" / "real_rfmid"
SEED_ROOT = OUT_ROOT / "ckpts_seed"
ANALYSIS = OUT_ROOT

# ``router``/``gating``/``multitask`` are used to identify the run directory that
# a given process just produced, because several arms share the ``multitask`` tag.
ARMS = [
    {"name": "E0", "config": "ablation_e0.yaml", "disease": "DR", "tag": "DR",
     "epochs": 15, "router": "fixed", "gating": False, "learnable": False, "multi": False},
    {"name": "E1", "config": "ablation_e1.yaml", "disease": "DR", "tag": "DR",
     "epochs": 15, "router": "monotone", "gating": False, "learnable": True, "multi": False},
    {"name": "E2", "config": "ablation_e2.yaml", "disease": None, "tag": "multitask",
     "epochs": 15, "router": "fixed", "gating": False, "learnable": False, "multi": True},
    {"name": "E3", "config": "ablation_e3.yaml", "disease": None, "tag": "multitask",
     "epochs": 15, "router": "monotone", "gating": False, "learnable": True, "multi": True},
    {"name": "E4", "config": "ablation_e4_free.yaml", "disease": None, "tag": "multitask",
     "epochs": 15, "router": "free_linear", "gating": False, "learnable": True, "multi": True},
    {"name": "E5", "config": "ablation_e5.yaml", "disease": None, "tag": "multitask",
     "epochs": 15, "router": "monotone", "gating": True, "learnable": True, "multi": True},
]


def write_cfg(src: Path, dst: Path, ckpt_dir: Path, epochs: int, seed: int) -> None:
    """Copy an arm config, pin paths/epochs, force CPU and set ``split.torch_seed``.

    Each task gets its own ``ckpt_dir`` because ``train.py`` names its run dir from
    a wall-clock timestamp; several arms trained in the same second would share a
    timestamp folder and overwrite each other's ``DR`` or ``multitask`` subtree.
    """
    lines: list[str] = []
    in_training = False
    in_split = False
    saw_torch_seed = False
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("data_dir:"):
            lines.append(f"data_dir: {(ROOT / 'data' / 'rfmid').as_posix()}")
            continue
        if line.startswith("ckpt_dir:"):
            lines.append(f"ckpt_dir: {ckpt_dir.as_posix()}")
            continue
        if line.startswith("training:"):
            in_training, in_split = True, False
            lines.append(line)
            continue
        if line.startswith("split:"):
            in_split, in_training = True, False
            lines.append(line)
            continue
        if line and not line.startswith((" ", "\t")):
            in_training = in_split = False
        if in_training and line.strip().startswith("device:"):
            lines.append("  device: cpu")
            continue
        if in_training and line.strip().startswith("epochs:") and "warmup" not in line:
            lines.append(f"  epochs: {epochs}")
            continue
        if in_split and line.strip().startswith("torch_seed:"):
            lines.append(f"  torch_seed: {seed}")
            saw_torch_seed = True
            continue
        lines.append(line)
    if not saw_torch_seed:
        out: list[str] = []
        inserted = False
        for line in lines:
            out.append(line)
            if (
                line.startswith("  seed:")
                and "torch_seed" not in line
                and not inserted
            ):
                out.append(f"  torch_seed: {seed}")
                inserted = True
        if not inserted:
            out.append("split:")
            out.append(f"  torch_seed: {seed}")
        lines = out
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def match_run(start_time: float, arm: dict, ckpt_root: Path) -> Path | None:
    """Run directory produced by this task under its private ``ckpt_root``."""
    best: tuple[float, Path] | None = None
    for run in ckpt_root.glob(f"*/{arm['tag']}/y0"):
        meta_path = run / "run_meta.json"
        if not meta_path.is_file():
            continue
        if run.stat().st_mtime < start_time - 2:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("quality_router") != arm["router"]:
            continue
        if bool(meta.get("quality_gating", False)) != arm["gating"]:
            continue
        if bool(meta.get("learnable_q", False)) != arm["learnable"]:
            continue
        if bool(meta.get("multitask", False)) != arm["multi"]:
            continue
        stamp = run.stat().st_mtime
        if best is None or stamp > best[0]:
            best = (stamp, run)
    return best[1] if best else None


def train_one(arm: dict, seed: int) -> tuple[str, str]:
    cfg_dir = SEED_ROOT / "_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = SEED_ROOT / f"{arm['name']}_s{seed}"
    cfg = cfg_dir / f"{arm['name']}_s{seed}.yaml"
    write_cfg(ROOT / arm["config"], cfg, ckpt_root, arm["epochs"], seed)
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config",
        str(cfg),
        "--dataset",
        "rfmid",
        "--data-dir",
        str(ROOT / "data" / "rfmid"),
        "--horizon",
        "0",
    ]
    if arm["disease"]:
        cmd.extend(["--disease", arm["disease"]])
    else:
        cmd.append("--multitask")
    if arm["learnable"]:
        cmd.append("--learnable-q")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "8")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "8")
    start = time.time()
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env,
    )
    log = (proc.stdout or "") + (proc.stderr or "")
    (OUT_ROOT / f"train_{arm['name']}_s{seed}.log").write_text(log, encoding="utf-8")
    if proc.returncode != 0:
        return arm["name"], f"train_failed: {log[-500:]}"
    run_dir = match_run(start, arm, ckpt_root)
    if run_dir is None:
        return arm["name"], "no_matching_run_dir"
    return arm["name"], f"ok:{run_dir}"


def best_epoch(run_dir: Path) -> int:
    from select_epochs import scan

    rows = scan(run_dir)
    if not rows:
        raise RuntimeError(f"no per-epoch validation predictions in {run_dir}")
    return int(max(rows, key=lambda r: r["val_macro_auroc"])["epoch"])


def collect(arm: str, seed: int, run_dir: Path) -> dict:
    import torch

    from collect_arm_predictions import collect_arm

    ep = best_epoch(run_dir)
    ckpt = run_dir / "ckpt" / f"{ep}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = collect_arm(
        run_dir, ROOT / "data" / "rfmid", "test", device, ckpt_override=str(ckpt)
    )
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ANALYSIS / f"arm_{arm}_s{seed}.npz", **payload)
    return {
        "run_dir": str(run_dir),
        "ckpt": str(ckpt),
        "epoch": ep,
        "n": int(payload["probs"].shape[0]),
        "k": int(payload["probs"].shape[1]),
        "monotone_weights": payload["monotone_weights"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=[a["name"] for a in ARMS])
    args = ap.parse_args()

    SEED_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ANALYSIS / "replicate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    arms = [a for a in ARMS if a["name"] in args.arms]
    tasks = [(arm, seed) for seed in args.seeds for arm in arms]
    statuses: dict[tuple[str, int], str] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(train_one, arm, seed): (arm["name"], seed) for arm, seed in tasks}
        for fut in as_completed(futures):
            name, seed = futures[fut]
            try:
                _, status = fut.result()
            except Exception as exc:  # noqa: BLE001
                status = f"error: {exc}"
            statuses[(name, seed)] = status
            print(f"[{name} s{seed}] {status}", flush=True)

    for (name, seed), status in sorted(statuses.items()):
        if not status.startswith("ok:"):
            continue
        try:
            rec = collect(name, seed, Path(status[3:]))
        except Exception as exc:  # noqa: BLE001
            print(f"[{name} s{seed}] collect failed: {exc}", flush=True)
            continue
        manifest[f"{name}_s{seed}"] = rec
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{name} s{seed}] epoch {rec['epoch']} n={rec['n']} k={rec['k']}", flush=True)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
