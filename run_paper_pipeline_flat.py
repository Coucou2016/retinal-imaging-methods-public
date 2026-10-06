#!/usr/bin/env python
"""Orchestrate the flat-snapshot pipeline (real-pixel RFMiD -> paper).

Run from the repository root.  Every stage is a thin ``subprocess`` call to a
flat module, mirroring the commands recorded in ``AUTHENTICITY_AUDIT.md`` -
nothing is hidden behind an import.

    python run_paper_pipeline_flat.py --stages demo
    python run_paper_pipeline_flat.py --stages cache train figures report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "rfmid"
OUT_DIR = ROOT / "results" / "real_rfmid"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call([sys.executable, *cmd], cwd=ROOT)


def stage_demo() -> None:
    run(["generate_demo_data.py", "--out-dir", str(ROOT / "data" / "UKBCompressed")])
    run(["train.py", "--demo", "--disease", "t2dm", "--horizon", "0"])
    run(["evaluate.py", "--demo", "--disease", "t2dm", "--horizon", "0", "--split", "val"])
    run(["-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-q"])


def stage_cache() -> None:
    run(["build_real_rfmid_cache.py", "--cache-dir", str(DATA_DIR), "--batch-size", "8"])


def stage_train() -> None:
    run([
        "run_real_rfmid_experiments.py",
        "--data-dir", str(DATA_DIR),
        "--out-dir", str(OUT_DIR),
        "--epochs", "12",
        "--bootstrap", "200",
    ])


def stage_eval() -> None:
    print("run_real_rfmid_experiments.py writes metrics_E*_rfmid_{val,test}.json")


def stage_figures() -> None:
    run(["plot_real_results.py"])


def stage_report() -> None:
    run(["build_paper_report.py"])


STAGES = {
    "demo": stage_demo,
    "cache": stage_cache,
    "train": stage_train,
    "eval": stage_eval,
    "figures": stage_figures,
    "report": stage_report,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="+", default=["demo"], choices=sorted(STAGES))
    args = ap.parse_args()
    for name in args.stages:
        print(f"=== stage: {name} ===", flush=True)
        STAGES[name]()


if __name__ == "__main__":
    main()
