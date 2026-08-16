#!/usr/bin/env python
"""Build UKB-compatible feature caches from public fundus manifests.

If images (or extracted backbone features) are missing, writes **deterministic
synthetic features** from labels. Those caches are for CI / pipeline demos only
and must never be reported as clinical AUROC.

Layout (same as data/UKBCompressed):
  UKB_RETF.npz / UKB_swin.npz / UKB_vim.npz  — left, right
  UKB_mqd.npz  — m, mn, ql, qr, center, pid
  UKB_y0.npz (and y5/y10 copies for loader compatibility) — y, yn

Public sets have no 5/10-year incidence: y5/y10 are copies of prevalence y0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset.brset import BRSET_LABELS, make_synthetic_brset_records, parse_brset_csv
from dataset.odir import ODIR_LABELS, make_synthetic_odir_records, parse_odir_csv
from dataset.public_common import (
    PatientRecord,
    pair_paths_for_record,
    synthetic_backbone_features,
    ukb_meta_matrix,
    write_pairs_manifest,
)
from dataset.rfmid import make_synthetic_rfmid_records, parse_rfmid_csv, parse_rfmid_splits
from reti_pioneer.data_paths import ukb_compressed_ready
from reti_pioneer.split import patient_level_train_val_indices, save_split

BACKBONE_DIMS = (("UKB_RETF", 1024, 0), ("UKB_swin", 1024, 1), ("UKB_vim", 384, 2))

HONESTY = {
    "odir": (
        "ODIR D (Diabetes) is ocular-evidence diabetes / DR findings, not UKB T2DM ICD. "
        "Gout/osteoporosis/hyperlipidemia/thyroid are omitted."
    ),
    "brset": (
        "BRSET diabetes, DR grade, and hypertensive retinopathy are distinct heads. "
        "Gout/osteoporosis/hyperlipidemia/thyroid are omitted."
    ),
    "rfmid": (
        "RFMiD labels are retinal disease signs, not UKB systemic ICD endpoints. "
        "Gout/osteoporosis/hyperlipidemia/thyroid are omitted."
    ),
}


def records_have_images(records: list[PatientRecord], root: Path | None) -> bool:
    for rec in records[:8]:
        for p in (rec.left_path, rec.right_path):
            if not p:
                continue
            path = Path(p)
            if path.is_file():
                return True
            if root is not None and (root / p).is_file():
                return True
    return False


def write_public_npz(
    out_dir: str | Path,
    records: list[PatientRecord],
    label_names: list[str],
    seed: int = 0,
    synthetic_features: bool = True,
    dataset_name: str = "public",
    force: bool = False,
    labels_only: bool = False,
    force_features: bool = False,
) -> Path:
    """Write UKB_*.npz from PatientRecord list.

    ``labels_only`` refreshes mqd/y/split without touching backbone feature npz.
    ``force_features`` is required to overwrite real (non-synthetic) backbone caches.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "SYNTHETIC_FEATURES.txt"
    has_cache = ukb_compressed_ready(str(out_dir))
    real_features = has_cache and not marker.is_file()

    if not force and not labels_only and has_cache:
        print(f"Cache already present in {out_dir} (use --force to overwrite)")
        return out_dir
    if not records:
        raise ValueError("No records to write")
    if real_features and synthetic_features and not labels_only and not force_features:
        raise RuntimeError(
            f"Refusing to overwrite real backbone features in {out_dir} with synthetic ones. "
            "Pass --labels-only to refresh mqd/y/split, or --force-features to intentionally replace."
        )

    y = np.stack([np.asarray(r.labels, dtype=np.float32).reshape(-1) for r in records])
    if y.shape[1] != len(label_names):
        raise ValueError(f"label width {y.shape[1]} != names {len(label_names)}")

    m, mn = ukb_meta_matrix(records)
    ql = np.stack([np.asarray(r.ql, dtype=np.float32) for r in records])
    qr = np.stack([np.asarray(r.qr, dtype=np.float32) for r in records])
    center = np.zeros(len(records), dtype=np.float32)
    # Keep pid as unicode so mixed ODIR/BRSET ids round-trip.
    pid = np.array([str(r.patient_id) for r in records])
    yn = np.array(label_names)

    np.savez_compressed(
        out_dir / "UKB_mqd.npz",
        m=m,
        mn=mn,
        ql=ql,
        qr=qr,
        center=center,
        pid=pid,
    )
    for horizon in (0, 5, 10):
        # y5/y10 are layout stubs: public cohorts have prevalence only.
        np.savez_compressed(out_dir / f"UKB_y{horizon}.npz", y=y, yn=yn)

    write_backbones = synthetic_features and not labels_only
    if write_backbones:
        marker.write_text(
            "SYNTHETIC backbone features derived from labels, not fundus photographs.\n"
            f"dataset={dataset_name} n={len(records)} seed={seed}\n"
            f"{HONESTY.get(dataset_name, '')}\n"
            "Do not report AUROC from this cache as a clinical result.\n",
            encoding="utf-8",
        )
        for fname, dim, stream in BACKBONE_DIMS:
            left, right = synthetic_backbone_features(y, dim, seed, stream)
            np.savez_compressed(out_dir / f"{fname}.npz", left=left, right=right)
    elif labels_only:
        print(f"labels-only: refreshed mqd/y under {out_dir}; backbone npz unchanged")
    elif not synthetic_features:
        raise RuntimeError(
            "Real-image feature extraction is not run from this script on CPU; "
            "pass --synthetic-features / --labels-only or extract with scripts/extract_features.py"
        )

    splits = [r.split for r in records]
    if any(s in {"train", "val", "test"} for s in splits):
        train_idx = [i for i, s in enumerate(splits) if s == "train" or s is None]
        val_idx = [i for i, s in enumerate(splits) if s == "val"]
        test_idx = [i for i, s in enumerate(splits) if s == "test"]
        if not val_idx:
            train_idx, val_idx = patient_level_train_val_indices(pid, y, 0.2, seed)
            test_idx = []
        save_split(
            str(out_dir / "split.npz"),
            train_idx,
            val_idx,
            seed,
            0.2,
            test_idx=test_idx or None,
            patient_ids=pid,
        )
    else:
        if len(np.unique(pid)) >= 4:
            train_idx, val_idx = patient_level_train_val_indices(pid, y, 0.25, seed)
            save_split(str(out_dir / "split.npz"), train_idx, val_idx, seed, 0.25, patient_ids=pid)

    meta = {
        "dataset": dataset_name,
        "n": len(records),
        "label_names": label_names,
        "synthetic_features": bool(marker.is_file()),
        "labels_only": labels_only,
        "honesty": HONESTY.get(dataset_name, ""),
        "horizons": "prevalence_only (y5/y10 copy y0)",
        "weight": "padded 0 (not collected)",
        "ethnicity": "padded 0 (not collected)",
        "seed": seed,
    }
    (out_dir / "label_map.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {dataset_name} cache ({len(records)} rows, K={len(label_names)}) to {out_dir}")
    return out_dir


def _find_csv(root: Path, names: list[str]) -> Path | None:
    for name in names:
        p = root / name
        if p.is_file():
            return p
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    csvs = list(root.glob("*.csv"))
    return csvs[0] if csvs else None


def load_records(dataset: str, src: Path | None, args) -> tuple[list[PatientRecord], list[str]]:
    # Explicit CSVs / split files always win (do not require --src).
    has_explicit_csv = bool(args.csv or args.train_csv or args.val_csv or args.test_csv)
    if args.synthetic_demo or (src is None and not has_explicit_csv):
        n = args.n_samples
        if dataset == "odir":
            return make_synthetic_odir_records(n, args.seed), list(ODIR_LABELS)
        if dataset == "brset":
            return make_synthetic_brset_records(n, args.seed), list(BRSET_LABELS)
        recs, names = make_synthetic_rfmid_records(n, args.seed)
        return recs, names

    if dataset == "odir":
        csv_path = Path(args.csv) if args.csv else (_find_csv(src, ["full_df.csv", "train.csv", "odir.csv"]) if src else None)
        if csv_path is None:
            raise FileNotFoundError(f"No ODIR CSV under {src}")
        return parse_odir_csv(csv_path), list(ODIR_LABELS)
    if dataset == "brset":
        csv_path = Path(args.csv) if args.csv else (_find_csv(src, ["labels.csv", "brset.csv", "metadata.csv"]) if src else None)
        if csv_path is None:
            raise FileNotFoundError(f"No BRSET CSV under {src}")
        return parse_brset_csv(csv_path), list(BRSET_LABELS)
    # rfmid
    if args.train_csv or args.val_csv or args.test_csv:
        recs, names = parse_rfmid_splits(args.train_csv, args.val_csv, args.test_csv)
        return recs, names
    if src is None:
        raise FileNotFoundError("RFMiD requires --src or --train-csv/--val-csv/--test-csv/--csv")
    train = _find_csv(src, ["RFMiD_Training_Labels.csv", "Training_Labels.csv"])
    val = _find_csv(src, ["RFMiD_Validation_Labels.csv", "Validation_Labels.csv"])
    test = _find_csv(src, ["RFMiD_Testing_Labels.csv", "Testing_Labels.csv"])
    if train or val or test:
        recs, names = parse_rfmid_splits(train, val, test)
        return recs, names
    csv_path = Path(args.csv) if args.csv else _find_csv(src, ["labels.csv"])
    if csv_path is None:
        raise FileNotFoundError(f"No RFMiD CSV under {src}")
    recs, names = parse_rfmid_csv(csv_path, split=None)
    return recs, names


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public-data UKB-style npz caches")
    parser.add_argument("--dataset", required=True, choices=["odir", "brset", "rfmid"])
    parser.add_argument("--src", default=None, help="Directory with CSV / images")
    parser.add_argument("--csv", default=None, help="Explicit labels CSV")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--out", default=None, help="Output directory for UKB_*.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=64, help="Used with --synthetic-demo")
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Ignore CSV/images; write a tiny deterministic public-style cache",
    )
    parser.add_argument(
        "--synthetic-features",
        action="store_true",
        default=True,
        help="Write deterministic features from labels (default; required without images)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Refresh UKB_mqd / UKB_y* / split.npz without rewriting backbone feature npz",
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Allow overwriting non-synthetic backbone features with synthetic ones",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write pairs.csv (left,right) under --out, same row order as UKB_mqd/y*",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Drop records with no readable left/right image before writing cache/manifest",
    )
    parser.add_argument(
        "--manifest-name",
        default="pairs.csv",
        help="Manifest filename under --out (default: pairs.csv)",
    )
    args = parser.parse_args()

    src = Path(args.src) if args.src else None
    out = Path(args.out) if args.out else Path(ROOT) / "data" / args.dataset
    records, names = load_records(args.dataset, src, args)

    if args.require_images or args.write_manifest:
        kept = [r for r in records if pair_paths_for_record(r, src) is not None]
        if args.require_images:
            if not kept:
                raise SystemExit("No records with readable images; check --src paths")
            if len(kept) < len(records):
                print(f"require-images: kept {len(kept)}/{len(records)} records with files")
            records = kept
        elif args.write_manifest and not kept and not args.synthetic_demo:
            print(
                "WARNING: --write-manifest but no image files resolve; "
                "pairs.csv will be empty unless paths exist under --src"
            )

    if src is not None and records_have_images(records, src) and not args.synthetic_demo:
        if not args.labels_only:
            print(
                "Images detected but this helper still writes SYNTHETIC features unless --labels-only. "
                "Recommended: prepare_public_npz.py --labels-only --write-manifest --require-images, "
                "then scripts/extract_features.py --cache-dir <out> --all-backbones"
            )

    write_public_npz(
        out,
        records,
        names,
        seed=args.seed,
        synthetic_features=True,
        dataset_name=args.dataset,
        force=args.force,
        labels_only=args.labels_only,
        force_features=args.force_features,
    )

    if args.write_manifest:
        if args.synthetic_demo and not args.require_images:
            print("skip --write-manifest for --synthetic-demo (no real images); use with --src + --require-images")
        else:
            manifest = out / args.manifest_name
            idx = write_pairs_manifest(records, manifest, root=src, skip_missing=True)
            if len(idx) != len(records):
                raise SystemExit(
                    f"Manifest wrote {len(idx)} rows but cache has {len(records)}. "
                    "Pass --require-images so labels and pairs stay aligned."
                )
            print(f"Wrote aligned manifest {manifest} (n={len(idx)})")
            print(
                "Next: python scripts/extract_features.py "
                f"--cache-dir {out} --all-backbones   # or --stub-identity for CI"
            )


if __name__ == "__main__":
    main()
