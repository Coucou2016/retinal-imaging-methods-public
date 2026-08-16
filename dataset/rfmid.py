"""RFMiD / RFMiD 2.0 official-split multi-label loader.

Label honesty
-------------
RFMiD labels are **retinal disease signs** (DR, ARMD, hypertensive retinopathy
among many others), not UKB ICD systemic endpoints. Hypertensive retinopathy
here is an ocular finding. Gout / osteoporosis / hyperlipidemia / thyroid are
**omitted**. Prefer the official train/val/test CSVs over a random split.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
from torch.utils.data import Dataset

from dataset.public_common import (
    PatientRecord,
    default_quality,
    find_column,
    parse_binary,
)

# Core columns present in RFMiD 1.0 training labels; extras are kept if present.
RFMID_DEFAULT_LABELS = [
    "Disease_Risk",
    "DR",
    "ARMD",
    "MH",
    "DN",
    "MYA",
    "BRVO",
    "TSLN",
    "ERM",
    "LS",
    "MS",
    "CSR",
    "ODC",
    "CRVO",
    "TV",
    "AH",
    "ODP",
    "ODE",
    "ST",
    "AION",
    "PT",
    "RT",
    "RS",
    "CRS",
    "EDN",
    "RPEC",
    "MHL",
    "RP",
    "OTHER",
]

_SKIP_COLS = {"id", "ID", "image_id", "filename", "file"}


def infer_rfmid_label_columns(fieldnames: list[str] | None) -> list[str]:
    if not fieldnames:
        return list(RFMID_DEFAULT_LABELS)
    cols = []
    for name in fieldnames:
        if name.strip() in _SKIP_COLS or name.strip().lower() == "id":
            continue
        cols.append(name.strip())
    return cols or list(RFMID_DEFAULT_LABELS)


def parse_rfmid_csv(
    path: str | Path,
    split: str | None = None,
    label_names: list[str] | None = None,
) -> tuple[list[PatientRecord], list[str]]:
    """Parse one official RFMiD labels CSV. Single image → copied to both eyes."""
    path = Path(path)
    records: list[PatientRecord] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        names = label_names or infer_rfmid_label_columns(reader.fieldnames)
        for row in reader:
            pid = find_column(row, "ID", "id", "image_id", default="")
            if not pid:
                continue
            labels = np.array([parse_binary(row.get(c, 0)) for c in names], dtype=np.float32)
            img = find_column(row, "filename", "file", "path", default=f"{pid}.png")
            records.append(
                PatientRecord(
                    patient_id=str(pid),
                    age=0.0,
                    sex=0.0,
                    labels=labels,
                    left_path=img,
                    right_path=img,  # unpaired CFP: duplicate to both eyes
                    ql=default_quality(),
                    qr=default_quality(),
                    split=split,
                )
            )
    return records, names


def parse_rfmid_splits(
    train_csv: str | Path | None = None,
    val_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
) -> tuple[list[PatientRecord], list[str]]:
    chunks: list[tuple[str, str | Path]] = []
    if train_csv:
        chunks.append(("train", train_csv))
    if val_csv:
        chunks.append(("val", val_csv))
    if test_csv:
        chunks.append(("test", test_csv))
    if not chunks:
        raise ValueError("Need at least one of train_csv, val_csv, test_csv")
    all_records: list[PatientRecord] = []
    names: list[str] | None = None
    for split, csv_path in chunks:
        recs, names = parse_rfmid_csv(csv_path, split=split, label_names=names)
        all_records.extend(recs)
    return all_records, names or list(RFMID_DEFAULT_LABELS)


def make_synthetic_rfmid_records(n: int, seed: int = 0) -> tuple[list[PatientRecord], list[str]]:
    rng = np.random.default_rng(seed)
    names = ["Disease_Risk", "DR", "ARMD", "MYA", "ODC", "OTHER"]
    records = []
    n_train = max(2, int(0.6 * n))
    n_val = max(1, int(0.2 * n))
    for i in range(n):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        labels = (rng.random(len(names)) < 0.25).astype(np.float32)
        records.append(
            PatientRecord(
                patient_id=str(i + 1),
                age=0.0,
                sex=0.0,
                labels=labels,
                left_path=f"{i + 1}.png",
                right_path=f"{i + 1}.png",
                split=split,
            )
        )
    return records, names


class RFMiDDataset(Dataset):
    def __init__(
        self,
        root: str | None = None,
        train_csv: str = "RFMiD_Training_Labels.csv",
        val_csv: str = "RFMiD_Validation_Labels.csv",
        test_csv: str = "RFMiD_Testing_Labels.csv",
        records: list[PatientRecord] | None = None,
        label_names: list[str] | None = None,
        transform: Callable | None = None,
        load_images: bool = False,
    ) -> None:
        self.root = Path(root) if root else Path(".")
        if records is not None:
            self.records = records
            self.label_names = label_names or list(RFMID_DEFAULT_LABELS)
        else:
            def _p(name: str) -> Path | None:
                p = Path(name)
                if p.is_file():
                    return p
                q = self.root / name
                return q if q.is_file() else None

            t, v, te = _p(train_csv), _p(val_csv), _p(test_csv)
            if t is None and v is None and te is None:
                raise FileNotFoundError(f"No RFMiD CSVs under {self.root}")
            self.records, self.label_names = parse_rfmid_splits(t, v, te)
        self.labels = self.label_names
        self.transform = transform
        self.load_images = load_images

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        meta = np.array([rec.age, rec.sex, rec.weight, rec.ethnicity], dtype=np.float32)
        return rec.left_path, rec.right_path, meta, (rec.ql, rec.qr), rec.labels.copy()
