"""Deterministic train/validation index splits (sample- or patient-level)."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
from torch.utils.data import Subset

SplitName = Literal["train", "val", "test", "all"]


def _label_scalar(y) -> float:
    if hasattr(y, "numel"):
        return float(y.sum().item() if y.numel() > 1 else y.item())
    arr = np.asarray(y)
    return float(arr.sum() if arr.size > 1 else arr.item())


def dataset_labels(dataset, max_samples: int | None = None) -> np.ndarray:
    """Binary (or any-positive) labels aligned with Subset indices 0..len(dataset)-1.

    Multi-label targets are reduced with a sum so stratification still sees a
    positive if any class is positive. For per-class arrays use dataset_label_matrix.
    """
    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)
    ys = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ys[i] = _label_scalar(dataset[i][-1])
    return ys


def dataset_label_matrix(dataset, max_samples: int | None = None) -> np.ndarray:
    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)
    rows = [np.asarray(dataset[i][-1], dtype=np.float32).reshape(-1) for i in range(n)]
    return np.stack(rows, axis=0)


def unwrap_feature_dataset(dataset) -> tuple[object, list[int]]:
    """Map outer dataset indices to raw rows of UKBDatasetFast (or similar)."""
    positions = list(range(len(dataset)))
    while isinstance(dataset, Subset):
        positions = [dataset.indices[i] for i in positions]
        dataset = dataset.dataset
    if hasattr(dataset, "indices"):
        positions = [dataset.indices[i] for i in positions]
    return dataset, positions


def dataset_patient_ids(dataset) -> np.ndarray:
    """Patient ids aligned with dataset[i], one per row.

    Falls back to 0..N-1 (each sample is its own patient) when `pid` is absent.
    """
    n = len(dataset)
    base, raw_rows = unwrap_feature_dataset(dataset)
    if hasattr(base, "pid"):
        pids = np.asarray(base.pid)
        return pids[np.asarray(raw_rows, dtype=np.int64)]
    return np.arange(n)


def dataset_meta_matrix(dataset) -> np.ndarray | None:
    base, raw_rows = unwrap_feature_dataset(dataset)
    if not hasattr(base, "m"):
        return None
    return np.asarray(base.m)[np.asarray(raw_rows, dtype=np.int64)]


def dataset_meta_names(dataset) -> list[str]:
    base, _ = unwrap_feature_dataset(dataset)
    names = getattr(base, "meta_names", None)
    if names is None:
        return []
    return [str(x) for x in names]


def stratified_train_val_indices(
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return (train_idx, val_idx) as positions into the given label array."""
    n = len(labels)
    if n < 4:
        raise ValueError(f"Need at least 4 samples for a split; got {n}")
    val_fraction = float(np.clip(val_fraction, 0.05, 0.5))
    n_val = max(2, int(round(n * val_fraction)))
    n_val = min(n_val, n - 2)

    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels >= 0.5)
    neg = np.flatnonzero(labels < 0.5)
    if len(pos) == 0 or len(neg) == 0:
        perm = rng.permutation(n)
        val_idx = perm[:n_val].tolist()
        train_idx = perm[n_val:].tolist()
        return train_idx, val_idx

    n_val_pos = max(1, int(round(n_val * len(pos) / n)))
    n_val_pos = min(n_val_pos, len(pos) - 1)
    n_val_neg = n_val - n_val_pos
    n_val_neg = max(1, min(n_val_neg, len(neg) - 1))
    n_val_pos = n_val - n_val_neg

    val_pos = rng.choice(pos, size=n_val_pos, replace=False)
    val_neg = rng.choice(neg, size=n_val_neg, replace=False)
    val_idx = np.concatenate([val_pos, val_neg])
    rng.shuffle(val_idx)
    mask = np.ones(n, dtype=bool)
    mask[val_idx] = False
    train_idx = np.flatnonzero(mask).tolist()
    return train_idx, val_idx.tolist()


def _row_positive(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim > 1:
        return (arr >= 0.5).any(axis=1).astype(np.float32)
    return (arr.reshape(-1) >= 0.5).astype(np.float32)


def patient_level_train_val_indices(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split by unique patient id; every row of a patient stays in one fold."""
    patient_ids = np.asarray(patient_ids)
    row_pos = _row_positive(labels)
    if len(patient_ids) != len(row_pos):
        raise ValueError("patient_ids and labels must have the same length")

    uniq, inv = np.unique(patient_ids, return_inverse=True)
    if len(uniq) < 4:
        raise ValueError(f"Need at least 4 patients for a split; got {len(uniq)}")

    p_labels = np.zeros(len(uniq), dtype=np.float32)
    for row_i, p_i in enumerate(inv):
        if row_pos[row_i] >= 0.5:
            p_labels[p_i] = 1.0

    train_p, val_p = stratified_train_val_indices(p_labels, val_fraction, seed)
    train_patients = {uniq[i] for i in train_p}
    val_patients = {uniq[i] for i in val_p}
    if train_patients & val_patients:
        raise RuntimeError("Patient leaked between train and val")

    train_idx = [i for i, p in enumerate(patient_ids) if p in train_patients]
    val_idx = [i for i, p in enumerate(patient_ids) if p in val_patients]
    return train_idx, val_idx


def patient_level_train_val_test_indices(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Patient-level 3-way split. Test is carved first, then val from the remainder."""
    patient_ids = np.asarray(patient_ids)
    row_pos = _row_positive(labels)
    uniq, inv = np.unique(patient_ids, return_inverse=True)
    if len(uniq) < 6:
        raise ValueError(f"Need at least 6 patients for a 3-way split; got {len(uniq)}")

    p_labels = np.zeros(len(uniq), dtype=np.float32)
    for row_i, p_i in enumerate(inv):
        if row_pos[row_i] >= 0.5:
            p_labels[p_i] = 1.0

    hold_frac = float(np.clip(test_fraction, 0.05, 0.4))
    rest_idx, test_p = stratified_train_val_indices(p_labels, hold_frac, seed)
    rest_labels = p_labels[np.asarray(rest_idx)]
    inner_frac = float(val_fraction) / max(1e-6, 1.0 - hold_frac)
    train_local, val_local = stratified_train_val_indices(rest_labels, inner_frac, seed + 1)
    train_p = [rest_idx[i] for i in train_local]
    val_p = [rest_idx[i] for i in val_local]

    train_patients = {uniq[i] for i in train_p}
    val_patients = {uniq[i] for i in val_p}
    test_patients = {uniq[i] for i in test_p}
    if len(train_patients & val_patients) or len(train_patients & test_patients) or len(val_patients & test_patients):
        raise RuntimeError("Patient leaked across train/val/test")

    train_idx = [i for i, p in enumerate(patient_ids) if p in train_patients]
    val_idx = [i for i, p in enumerate(patient_ids) if p in val_patients]
    test_idx = [i for i, p in enumerate(patient_ids) if p in test_patients]
    return train_idx, val_idx, test_idx


def make_subsets(dataset, train_idx: list[int], val_idx: list[int]) -> tuple[Subset, Subset]:
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def save_split(
    path: str,
    train_idx: list[int],
    val_idx: list[int],
    seed: int,
    val_fraction: float,
    test_idx: list[int] | None = None,
    patient_ids: np.ndarray | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: dict = {
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
        "seed": np.int64(seed),
        "val_fraction": np.float64(val_fraction),
    }
    if test_idx is not None:
        payload["test_idx"] = np.asarray(test_idx, dtype=np.int64)
    if patient_ids is not None:
        payload["patient_ids"] = np.asarray(patient_ids)
    np.savez_compressed(path, **payload)


def load_split(path: str) -> tuple[list[int], list[int]]:
    data = np.load(path, allow_pickle=True)
    return data["train_idx"].tolist(), data["val_idx"].tolist()


def load_test_idx(path: str) -> list[int] | None:
    data = np.load(path, allow_pickle=True)
    if "test_idx" not in data.files:
        return None
    return data["test_idx"].tolist()


def resolve_split_indices(
    split: SplitName,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int] | None = None,
) -> list[int]:
    if split == "train":
        return train_idx
    if split == "val":
        return val_idx
    if split == "test":
        if test_idx is None:
            raise ValueError("split='test' requires test_idx in split.npz")
        return test_idx
    if test_idx is None:
        return train_idx + val_idx
    return train_idx + val_idx + test_idx


def subset_for_split(
    dataset,
    split: SplitName,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int] | None = None,
) -> Subset:
    idx = resolve_split_indices(split, train_idx, val_idx, test_idx)
    return Subset(dataset, idx)
