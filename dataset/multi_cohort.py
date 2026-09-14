"""Multi-cohort partial-label dataset: ODIR + BRSET + RFMiD → one joint vocabulary."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset

from dataset.UKBDataset import UKBDatasetFast
from reti_pioneer.joint_vocab import (
    JOINT_VOCAB,
    build_joint_projection,
    project_labels_to_joint,
)


class MultiCohortDataset(Dataset):
    """Concatenate UKB-style caches remapped onto ``JOINT_VOCAB``.

    Each sample keeps its backbone features / meta / quality; labels are projected
    with ``-1`` for endpoints absent in the source cohort (masked BCE).
    """

    def __init__(
        self,
        cohort_dirs: dict[str, str | Path],
        meta: Sequence[str],
        pretrain: Sequence[str] | None = None,
        horizon: int = 0,
    ) -> None:
        if not cohort_dirs:
            raise ValueError("cohort_dirs must be non-empty")
        self.meta = list(meta)
        self.pretrain = list(pretrain or ["RETF", "SwinB", "VimS"])
        self.horizon = int(horizon)
        self.disease_names = list(JOINT_VOCAB)
        self.cohort_names: list[str] = []
        self._rows: list[tuple[str, UKBDatasetFast, int, np.ndarray]] = []
        # (cohort, base_ds, local_index_into_base.indices, joint_label_row)

        for name, path in cohort_dirs.items():
            path = Path(path)
            if not path.is_dir():
                raise FileNotFoundError(f"MultiCohort cache missing: {path}")
            ds = UKBDatasetFast(
                str(path),
                None,
                list(meta),
                disease=[0],
                use_pretrain=list(self.pretrain),
                incident_exclude_prior=False,
            )
            # Use all prevalence columns for projection source.
            ds.set_target(horizon, list(ds.disease_names), incident_exclude_prior=False)
            proj = build_joint_projection(name, ds.disease_names)
            for local_i, raw in enumerate(ds.indices):
                y_local = np.asarray(ds.yy[raw], dtype=np.float32).reshape(-1)
                y_joint = project_labels_to_joint(y_local.reshape(1, -1), proj)[0]
                self._rows.append((str(name), ds, local_i, y_joint))
            self.cohort_names.append(str(name))

        self.indices = list(range(len(self._rows)))
        self._disease_idx = list(range(len(self.disease_names)))
        # Fake yy / pid attributes for split helpers.
        self.yy = np.stack([r[3] for r in self._rows], axis=0)
        pids = []
        for cohort, ds, local_i, _y in self._rows:
            raw = ds.indices[local_i]
            base_pid = ds.pid[raw] if hasattr(ds, "pid") else raw
            pids.append(f"{cohort}:{base_pid}")
        self.pid = np.array(pids, dtype=object)
        # Meta matrix aligned with rows (for fairness helpers).
        self.m = np.stack(
            [np.asarray(r[1].m[r[1].indices[r[2]]], dtype=np.float32) for r in self._rows],
            axis=0,
        )
        self.meta_names = list(meta)
        self.ql = np.stack(
            [np.asarray(r[1].ql[r[1].indices[r[2]]], dtype=np.float32) for r in self._rows],
            axis=0,
        )
        self.qr = np.stack(
            [np.asarray(r[1].qr[r[1].indices[r[2]]], dtype=np.float32) for r in self._rows],
            axis=0,
        )

    def set_target(self, y: int, disease, incident_exclude_prior: bool = False) -> None:
        del incident_exclude_prior
        self.horizon = int(y)
        if disease and isinstance(disease[0], str):
            missing = [n for n in disease if n not in self.disease_names]
            if missing:
                raise ValueError(f"Unknown joint disease {missing}; have {self.disease_names}")
            self._disease_idx = [self.disease_names.index(n) for n in disease]
        else:
            self._disease_idx = [int(d) for d in disease] if disease else list(range(len(self.disease_names)))
        self.indices = list(range(len(self._rows)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        row_i = self.indices[index]
        cohort, ds, local_i, y_joint = self._rows[row_i]
        del cohort
        # Reuse underlying UKBDatasetFast __getitem__ via its indices list.
        sample = ds[local_i]
        # sample ends with labels; replace with joint projection slice.
        *front, _old_y = sample
        y = y_joint[np.asarray(self._disease_idx, dtype=np.int64)].astype(np.float32)
        return (*front, y)
