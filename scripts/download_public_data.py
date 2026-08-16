#!/usr/bin/env python
"""Best-effort helper: print download steps; optionally pull ODIR/RFMiD if credentials exist.

Never fails the suite if download is unavailable. BRSET (PhysioNet) is credentialed —
this script only prints the URL and checklist, it does not scrape PhysioNet.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ODIR_KAGGLE = "andrewmvd/ocular-disease-recognition-odir5k"
ODIR_GC = "https://odir2019.grand-challenge.org/dataset/"
BRSET_PHYSIONET = "https://www.physionet.org/content/brazilian-ophthalmological/1.0.2/"
RFMID_HF = "ctmedtech/RFMID"


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _kaggle_ready() -> bool:
    if not _have("kaggle"):
        return False
    home = os.path.expanduser("~")
    return os.path.isfile(os.path.join(home, ".kaggle", "kaggle.json")) or bool(
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )


def _hf_ready() -> bool:
    return _have("huggingface-cli") or _have("hf")


def print_manual_steps(dest: str) -> None:
    print(
        f"""
=== Public fundus download checklist ===
Destination root: {dest}

1) ODIR-5K
   - Grand Challenge: {ODIR_GC}
   - Kaggle (if credentialed): kaggle datasets download -d {ODIR_KAGGLE}
   - Expect: full_df.csv (or similar) + left/right fundus images
   - Then: python scripts/prepare_public_npz.py --dataset odir --src <odir-root> --out data/odir

2) BRSET (PhysioNet — credentialed; this script will NOT download)
   - Apply / sign DUA: {BRSET_PHYSIONET}
   - Download the official archive after credentialing
   - Then: python scripts/prepare_public_npz.py --dataset brset --src <brset-root> --out data/brset

3) RFMiD / RFMiD 2.0
   - Hugging Face: {RFMID_HF}
   - Or IEEE DataPort / MDPI Data 2021 links in docs/PUBLIC_DATA.md
   - Then: python scripts/prepare_public_npz.py --dataset rfmid --src <rfmid-root> --out data/rfmid

4) Real backbone features (GPU + RETFound / Swin / optional Vim)
   - python scripts/extract_features.py --manifest pairs.csv --out data/odir/UKB_RETF.npz --backbone retf
   - Until then, caches may contain SYNTHETIC_FEATURES.txt — not for manuscript AUROC.

5) EyeQ / UKB
   - EyeQ DenseNet weights: see docs/DATA.md (HzFu/EyeQ)
   - UK Biobank: application required; not public (docs/DATA.md)

Optional CI path without downloads:
  python scripts/prepare_public_npz.py --dataset odir --synthetic-demo --out data/odir --force
  python scripts/run_ablations.py --quick
""".strip()
    )


def try_kaggle_odir(dest: str) -> bool:
    out = os.path.join(dest, "odir")
    os.makedirs(out, exist_ok=True)
    if not _kaggle_ready():
        print("[odir] Kaggle CLI/credentials not found — skip auto-download.")
        return False
    zip_path = os.path.join(out, "odir5k.zip")
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        ODIR_KAGGLE,
        "-p",
        out,
        "--unzip",
    ]
    print("+", " ".join(cmd))
    try:
        subprocess.run(cmd, check=False, cwd=ROOT)
    except OSError as exc:
        print(f"[odir] kaggle spawn failed: {exc}")
        return False
    # Success if any csv appears
    for root, _, files in os.walk(out):
        if any(f.lower().endswith(".csv") for f in files):
            print(f"[odir] appears present under {out}")
            return True
    if os.path.isfile(zip_path):
        print(f"[odir] zip downloaded to {zip_path} (unzip manually if needed)")
        return True
    print("[odir] download did not produce CSV — check Kaggle access.")
    return False


def try_hf_rfmid(dest: str) -> bool:
    out = os.path.join(dest, "rfmid")
    os.makedirs(out, exist_ok=True)
    if not _hf_ready():
        # Try python huggingface_hub if installed
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError:
            print("[rfmid] huggingface-cli / huggingface_hub not found — skip auto-download.")
            return False
        try:
            snapshot_download(repo_id=RFMID_HF, repo_type="dataset", local_dir=out)
            print(f"[rfmid] snapshot under {out}")
            return True
        except Exception as exc:  # noqa: BLE001 — best-effort
            print(f"[rfmid] huggingface_hub failed (soft): {exc}")
            return False

    cli = "hf" if _have("hf") else "huggingface-cli"
    cmd = [cli, "download", RFMID_HF, "--repo-type", "dataset", "--local-dir", out]
    print("+", " ".join(cmd))
    try:
        completed = subprocess.run(cmd, check=False, cwd=ROOT)
    except OSError as exc:
        print(f"[rfmid] CLI spawn failed: {exc}")
        return False
    if completed.returncode != 0:
        print("[rfmid] CLI returned non-zero — soft skip.")
        return False
    print(f"[rfmid] download attempted under {out}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print public-data download steps; optional Kaggle/HF pull (never hard-fail)."
    )
    parser.add_argument(
        "--dest",
        default=os.path.join(ROOT, "data", "raw"),
        help="Root for optional downloads (default: data/raw)",
    )
    parser.add_argument(
        "--try-download",
        action="store_true",
        help="Attempt ODIR (Kaggle) and RFMiD (HF) if credentials exist",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "odir", "brset", "rfmid"],
        default="all",
        help="Which dataset section to emphasize",
    )
    args = parser.parse_args()
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    print_manual_steps(dest)
    print(f"\n[brset] Credentialed URL only: {BRSET_PHYSIONET}")
    print("        This helper never downloads BRSET automatically.")

    if not args.try_download:
        print("\nPass --try-download to attempt Kaggle ODIR / Hugging Face RFMiD when creds exist.")
        return 0

    if args.dataset in ("all", "odir"):
        try_kaggle_odir(dest)
    if args.dataset in ("all", "rfmid"):
        try_hf_rfmid(dest)
    if args.dataset == "brset":
        print("[brset] No auto-download. Use the PhysioNet URL above after credentialing.")

    print("\nExit 0 (download failures are soft).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
