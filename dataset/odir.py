"""ODIR-5K manifest loader (paired left/right fundus + 8 multi-labels).

Label honesty
-------------
ODIR column **D (Diabetes)** is primarily *diabetic retinopathy / diabetic
ocular findings* annotated on color fundus photographs. It is **not** UK
Biobank ICD-style type 2 diabetes (Reti-Pioneer T2DM, paper AUROC 0.833).
Report ODIR-D as *ocular-evidence diabetes*, never as a drop-in replica of
UKB T2DM.

Gout, osteoporosis, hyperlipidemia, and thyroid disease have **no** ODIR
gold standard and must be omitted on this public set (UKB-only extension).
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
    parse_float,
    parse_sex,
)

# Official ODIR-2019 8-class multi-label vector (N/D/G/C/A/H/M/O).
ODIR_LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]
ODIR_LABEL_NAMES = {
    "N": "normal",
    "D": "diabetes_ocular",  # NOT UKB T2DM
    "G": "glaucoma",
    "C": "cataract",
    "A": "amd",
    "H": "hypertension_ocular",
    "M": "myopia",
    "O": "other",
}


def _odir_labels_from_row(row: dict) -> np.ndarray:
    vals = []
    for key in ODIR_LABELS:
        raw = find_column(row, key, ODIR_LABEL_NAMES[key], default="0")
        vals.append(parse_binary(raw))
    return np.array(vals, dtype=np.float32)


def parse_odir_csv(path: str | Path) -> list[PatientRecord]:
    """Parse an ODIR-style CSV. Images are not required."""
    path = Path(path)
    records: list[PatientRecord] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = find_column(row, "ID", "id", "Patient ID", "patient_id", default="")
            if pid is None or pid == "":
                continue
            age = parse_float(find_column(row, "Patient Age", "age", "Age"), 0.0)
            sex = parse_sex(find_column(row, "Patient Sex", "sex", "Sex", "gender"))
            left = find_column(row, "Left-Fundus", "Left Fundus", "left", "left_path")
            right = find_column(row, "Right-Fundus", "Right Fundus", "right", "right_path")
            records.append(
                PatientRecord(
                    patient_id=str(pid),
                    age=age,
                    sex=sex,
                    labels=_odir_labels_from_row(row),
                    left_path=left,
                    right_path=right,
                    ql=default_quality(),
                    qr=default_quality(),
                )
            )
    return records


def make_synthetic_odir_records(n: int, seed: int = 0) -> list[PatientRecord]:
    """Tiny deterministic ODIR-shaped table for CI (no images)."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        labels = (rng.random(len(ODIR_LABELS)) < 0.2).astype(np.float32)
        if labels.sum() == 0:
            labels[0] = 1.0  # N
        records.append(
            PatientRecord(
                patient_id=str(i),
                age=float(rng.normal(60, 10)),
                sex=float(rng.integers(0, 2)),
                labels=labels,
                left_path=f"{i}_left.jpg",
                right_path=f"{i}_right.jpg",
            )
        )
    return records


class ODIRDataset(Dataset):
    """Paired-eye ODIR records. Does not load pixels unless `load_images=True`."""

    labels = ODIR_LABELS

    def __init__(
        self,
        root: str | None = None,
        csv_name: str = "full_df.csv",
        records: list[PatientRecord] | None = None,
        transform: Callable | None = None,
        load_images: bool = False,
    ) -> None:
        if records is not None:
            self.records = records
            self.root = Path(root) if root else Path(".")
        else:
            if root is None:
                raise ValueError("root or records is required")
            self.root = Path(root)
            csv_path = self.root / csv_name if not Path(csv_name).is_file() else Path(csv_name)
            if not csv_path.is_file():
                # common Kaggle layout
                alt = self.root / "data" / csv_name
                csv_path = alt if alt.is_file() else csv_path
            self.records = parse_odir_csv(csv_path)
        self.transform = transform
        self.load_images = load_images

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        meta = np.array([rec.age, rec.sex, rec.weight, rec.ethnicity], dtype=np.float32)
        if not self.load_images:
            return rec.left_path, rec.right_path, meta, (rec.ql, rec.qr), rec.labels.copy()
        from PIL import Image

        from dataset.transforms import fundus_transform

        tf = self.transform or fundus_transform(source="hospital")

        def _open(name: str | None):
            if not name:
                raise FileNotFoundError("missing eye path")
            path = Path(name)
            if not path.is_file():
                path = self.root / name
            img = Image.open(path).convert("RGB")
            return tf(img)

        return _open(rec.left_path), _open(rec.right_path), meta, (rec.ql, rec.qr), rec.labels.copy()
