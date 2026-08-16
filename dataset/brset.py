"""BRSET-style CSV loader (laterality, demographics, diabetes vs HR).

Label honesty
-------------
BRSET columns must be treated as **distinct heads**:

* ``diabetes`` — self-reported / clinical diabetes diagnosis (systemic).
* diabetic retinopathy grade (if present) — ocular DR, not the same as diabetes.
* ``hypertensive_retinopathy`` — specialist retinal sign, not systemic hypertension.

Never pool these without a table footnote. Gout, osteoporosis, hyperlipidemia,
and thyroid disease are **omitted** (no public CFP gold standard).
"""

from __future__ import annotations

import csv
from collections import defaultdict
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
    quality_from_scores,
)

# Default multi-label heads used in this repo (extend if extra CSV columns exist).
BRSET_LABELS = ["diabetes", "hypertensive_retinopathy", "dr_referable"]
BRSET_LABEL_NOTES = {
    "diabetes": "systemic diabetes diagnosis — not DR grade, not UKB T2DM ICD",
    "hypertensive_retinopathy": "ocular hypertensive retinopathy — not systemic HTN",
    "dr_referable": "referable DR (grade >= 2) when a DR-grade column exists; else 0",
}


def _laterality(value: str | None) -> str:
    s = (value or "").strip().lower()
    if s in {"l", "left", "os", "e", "olho esquerdo"}:
        return "left"
    if s in {"r", "right", "od", "d", "olho direito"}:
        return "right"
    return s


def _dr_referable(row: dict) -> float:
    raw = find_column(
        row,
        "dr_referable",
        "DR",
        "diabetic_retinopathy",
        "retinopathy_grade",
        "DR_ICDR",
        "icdr",
        default=None,
    )
    if raw is None:
        return 0.0
    try:
        grade = float(raw)
        return 1.0 if grade >= 2.0 else 0.0
    except ValueError:
        return parse_binary(raw)


def _quality_from_row(row: dict) -> np.ndarray:
    good = find_column(row, "quality_good", "q_good", default=None)
    usable = find_column(row, "quality_usable", "q_usable", default=None)
    bad = find_column(row, "quality_bad", "q_bad", default=None)
    focus = find_column(row, "quality_focus", "focus", "Focus", default=None)
    illum = find_column(row, "quality_illumination", "illumination", default=None)
    art = find_column(row, "quality_artifacts", "artifacts", "artifact", default=None)
    if any(v is not None for v in (good, usable, bad, focus, illum, art)):
        return quality_from_scores(good=good, usable=usable, bad=bad, focus=focus, illumination=illum, artifacts=art)
    return default_quality()


def parse_brset_csv(path: str | Path) -> list[PatientRecord]:
    """Parse a BRSET-style image-level CSV and pair left/right by patient.

    Patients with only one laterality have that eye copied to the missing side
    (documented fallback; not a second independent photo).
    """
    path = Path(path)
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = find_column(row, "patient_id", "patient", "ID", "id", default=None)
            if not pid:
                continue
            groups[str(pid)].append(row)

    records: list[PatientRecord] = []
    for pid, rows in groups.items():
        left_rows = [r for r in rows if _laterality(find_column(r, "laterality", "eye", "side")) == "left"]
        right_rows = [r for r in rows if _laterality(find_column(r, "laterality", "eye", "side")) == "right"]
        unlabeled = [r for r in rows if _laterality(find_column(r, "laterality", "eye", "side")) not in {"left", "right"}]
        if not left_rows and not right_rows:
            left_rows = unlabeled[:1]
            right_rows = unlabeled[1:2] or unlabeled[:1]
        n_pair = max(len(left_rows), len(right_rows), 1)
        for i in range(n_pair):
            left = left_rows[i] if i < len(left_rows) else (left_rows[-1] if left_rows else right_rows[min(i, len(right_rows) - 1)])
            right = right_rows[i] if i < len(right_rows) else (right_rows[-1] if right_rows else left_rows[min(i, len(left_rows) - 1)])
            src = left or right
            labels = np.array(
                [
                    parse_binary(find_column(src, "diabetes", "dm", "Diabetes")),
                    parse_binary(find_column(src, "hypertensive_retinopathy", "hr", "hypertensive retinopathy")),
                    _dr_referable(src),
                ],
                dtype=np.float32,
            )
            age = parse_float(find_column(src, "age", "Age", "patient_age"), 0.0)
            sex = parse_sex(find_column(src, "sex", "gender", "Sex"))
            left_path = find_column(left, "image_id", "image", "filename", "file", "path") if left else None
            right_path = find_column(right, "image_id", "image", "filename", "file", "path") if right else None
            records.append(
                PatientRecord(
                    patient_id=str(pid),
                    age=age,
                    sex=sex,
                    labels=labels,
                    left_path=left_path,
                    right_path=right_path,
                    ql=_quality_from_row(left) if left else default_quality(),
                    qr=_quality_from_row(right) if right else default_quality(),
                )
            )
    return records


def make_synthetic_brset_records(n: int, seed: int = 0) -> list[PatientRecord]:
    """n patients; a few share ids (two visits) so patient-level split can be tested."""
    rng = np.random.default_rng(seed)
    records = []
    n_patients = max(4, n - n // 6)
    extra = n - n_patients
    pids = [str(i) for i in range(n_patients)] + [str(i % max(1, n_patients)) for i in range(extra)]
    pids = pids[:n]
    for i, pid in enumerate(pids):
        labels = np.array(
            [
                float(rng.random() < 0.3),
                float(rng.random() < 0.15),
                float(rng.random() < 0.2),
            ],
            dtype=np.float32,
        )
        records.append(
            PatientRecord(
                patient_id=pid,
                age=float(rng.normal(58, 12)),
                sex=float(rng.integers(0, 2)),
                labels=labels,
                left_path=f"{pid}_{i}_left.jpg",
                right_path=f"{pid}_{i}_right.jpg",
                ql=quality_from_scores(focus=1.0, illumination=1.0, artifacts=float(rng.random() < 0.1)),
                qr=quality_from_scores(focus=1.0, illumination=0.8, artifacts=0.0),
            )
        )
    return records


class BRSETDataset(Dataset):
    labels = BRSET_LABELS

    def __init__(
        self,
        root: str | None = None,
        csv_name: str = "labels.csv",
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
            csv_path = Path(csv_name)
            if not csv_path.is_file():
                csv_path = self.root / csv_name
            self.records = parse_brset_csv(csv_path)
        self.transform = transform
        self.load_images = load_images

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        meta = np.array([rec.age, rec.sex, rec.weight, rec.ethnicity], dtype=np.float32)
        return rec.left_path, rec.right_path, meta, (rec.ql, rec.qr), rec.labels.copy()
