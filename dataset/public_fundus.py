"""Loaders for public fundus datasets.

Full paper labels (six endocrine/metabolic diseases) require UK Biobank and
hospital cohorts. Public sets support ocular / mixed labels for the follow-up
methods paper — see docs/PUBLIC_DATA.md for mapping and honesty notes.

APTOS / MESSIDOR remain DR-grade **proxy** pipelines only (not T2DM).
ODIR / BRSET / RFMiD are re-exported from dedicated modules.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
from torch.utils.data import Dataset

from dataset.odir import ODIR_LABELS, ODIRDataset, parse_odir_csv  # noqa: F401
from dataset.brset import BRSET_LABELS, BRSETDataset, parse_brset_csv  # noqa: F401
from dataset.rfmid import RFMID_DEFAULT_LABELS, RFMiDDataset, parse_rfmid_csv  # noqa: F401


class AptosCsvDataset(Dataset):
    """APTOS 2019 blindness detection — DR grade as proxy label column.

    Referable DR (grade >= 2) is **not** systemic T2DM.
    """

    def __init__(
        self,
        root: str,
        csv_name: str = "train.csv",
        image_subdir: str = "train_images",
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        if transform is None:
            from dataset.transforms import fundus_transform

            transform = fundus_transform(source="hospital")
        self.transform = transform
        rows = []
        with open(self.root / csv_name, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        self.rows = rows
        self.image_subdir = image_subdir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        img_id = row["id_code"] if "id_code" in row else row.get("id", "")
        path = self.root / self.image_subdir / f"{img_id}.png"
        if not path.exists():
            path = self.root / self.image_subdir / f"{img_id}.jpg"
        from PIL import Image

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        grade = int(row.get("diagnosis", row.get("level", 0)))
        # Binary proxy: referable DR (grade >= 2)
        y = np.array([1.0 if grade >= 2 else 0.0], dtype=np.float32)
        meta = np.zeros(4, dtype=np.float32)
        q = np.array([0.8, 0.15, 0.05], dtype=np.float32)
        return img, img, meta, (q, q), y


class MessidorIndexDataset(Dataset):
    """MESSIDOR-style layout: images/ + optional labels.csv (image, grade)."""

    def __init__(self, root: str, transform: Callable | None = None) -> None:
        self.root = Path(root)
        if transform is None:
            from dataset.transforms import fundus_transform

            transform = fundus_transform(source="hospital")
        self.transform = transform
        label_path = self.root / "labels.csv"
        if label_path.exists():
            self.items = []
            with open(label_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.items.append((row["image"], int(row["grade"])))
        else:
            images = list((self.root / "images").glob("*.jpg")) + list(
                (self.root / "images").glob("*.tif")
            )
            self.items = [(p.name, 0) for p in images]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        name, grade = self.items[index]
        from PIL import Image

        path = self.root / "images" / name
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = np.array([1.0 if grade >= 2 else 0.0], dtype=np.float32)
        meta = np.zeros(4, dtype=np.float32)
        q = np.array([0.7, 0.2, 0.1], dtype=np.float32)
        return img, img, meta, (q, q), y
