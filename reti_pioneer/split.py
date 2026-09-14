"""Deterministic train/validation/calibration/test index splits (patient-level)."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Literal

import numpy as np
from torch.utils.data import Subset

SplitName = Literal["train", "val", "calibration", "cal", "test", "all"]


def _label_scalar(y) -> float:
    """Any-positive among supervised entries (ignore missing <0 / NaN)."""
    if hasattr(y, "numel"):
        import torch

        yy = y.detach().float().reshape(-1)
        present = yy[torch.isfinite(yy) & (yy >= 0)]
        if present.numel() == 0:
            return 0.0
        return float((present >= 0.5).any().item())
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    present = arr[np.isfinite(arr) & (arr >= 0)]
    if present.size == 0:
        return 0.0
    return float((present >= 0.5).any())


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


def assert_calibration_disjoint(
    calibration_ids: list[int],
    evaluation_ids: list[int],
) -> None:
    leak = set(calibration_ids) & set(evaluation_ids)
    if leak:
        raise RuntimeError(
            f"Calibration/evaluation leakage: {len(leak)} shared indices. "
            "Fit temperature on calibration_ids only; score evaluation_ids only."
        )


def assert_split_label_coverage(
    labels: np.ndarray,
    indices: list[int] | np.ndarray,
    *,
    fold_name: str,
    require_both_classes: bool = True,
    per_class: bool = True,
) -> None:
    """Fail if a formal fold is missing positives/negatives (or per-class).

    For multilabel ``labels`` (N, K), when ``per_class`` is True each column with
    any finite supervised entries must contain both 0 and 1 among ``indices``.
    Missing labels (``<0`` / NaN) are ignored per column.
    """
    y = np.asarray(labels)
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        raise ValueError(f"{fold_name} fold is empty")
    if y.ndim == 1:
        vals = y[idx]
        present = vals[np.isfinite(vals) & (vals >= 0)]
        if present.size == 0:
            raise ValueError(f"{fold_name}: no supervised labels")
        if require_both_classes and len(np.unique((present >= 0.5).astype(int))) < 2:
            raise ValueError(
                f"{fold_name}: need both positive and negative labels for formal endpoints "
                f"(got unique={np.unique(present)})"
            )
        return
    # Multilabel
    sub = y[idx]
    for k in range(sub.shape[1]):
        col = sub[:, k]
        present = col[np.isfinite(col) & (col >= 0)]
        if present.size == 0:
            continue  # fully masked column — OK for partial-label
        if per_class and require_both_classes and len(np.unique((present >= 0.5).astype(int))) < 2:
            raise ValueError(
                f"{fold_name}: class column {k} missing pos or neg "
                f"(n={present.size}, unique={np.unique(present)})"
            )


def patient_level_multilabel_train_val_indices(
    patient_ids: np.ndarray,
    label_matrix: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Patient split stratified on any-positive OR first informative column.

    Improves multilabel balance vs collapsing only by sum: prefers patients that
    cover rare positive columns when possible via iterative rare-class priority.
    """
    patient_ids = np.asarray(patient_ids)
    y = np.asarray(label_matrix, dtype=np.float32)
    if y.ndim == 1:
        return patient_level_train_val_indices(patient_ids, y, val_fraction, seed)
    n = y.shape[0]
    if len(patient_ids) != n:
        raise ValueError("patient_ids and label_matrix length mismatch")

    # Prevalence per class among supervised entries.
    prev = []
    for k in range(y.shape[1]):
        col = y[:, k]
        present = col[np.isfinite(col) & (col >= 0)]
        if present.size == 0:
            prev.append(1.0)
        else:
            prev.append(float((present >= 0.5).mean()))
    # Stratify on the rarest supervised class (most fragile for AUROC).
    rare_k = int(np.argmin(prev)) if prev else 0
    rare_labels = np.zeros(n, dtype=np.float32)
    col = y[:, rare_k]
    mask = np.isfinite(col) & (col >= 0)
    rare_labels[mask] = (col[mask] >= 0.5).astype(np.float32)
    # Patients with no supervised rare label fall back to any-positive.
    any_pos = _row_positive(y)
    unset = ~mask
    rare_labels[unset] = any_pos[unset]
    return patient_level_train_val_indices(patient_ids, rare_labels, val_fraction, seed)


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


def patient_level_train_cal_val_indices(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    cal_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Patient-level train / calibration / val (eval) with pairwise disjoint patients.

    Order: carve ``val`` (evaluation) first, then ``calibration`` from the remainder,
    rest is ``train``. Calibration never overlaps the evaluation fold.
    """
    patient_ids = np.asarray(patient_ids)
    row_pos = _row_positive(labels)
    uniq, inv = np.unique(patient_ids, return_inverse=True)
    if len(uniq) < 6:
        raise ValueError(f"Need at least 6 patients for train/cal/val; got {len(uniq)}")

    p_labels = np.zeros(len(uniq), dtype=np.float32)
    for row_i, p_i in enumerate(inv):
        if row_pos[row_i] >= 0.5:
            p_labels[p_i] = 1.0

    val_frac = float(np.clip(val_fraction, 0.05, 0.45))
    rest_idx, val_p = stratified_train_val_indices(p_labels, val_frac, seed)
    rest_labels = p_labels[np.asarray(rest_idx)]
    # cal_fraction is relative to the full cohort; convert to fraction of remainder.
    cal_of_rest = float(cal_fraction) / max(1e-6, 1.0 - val_frac)
    cal_of_rest = float(np.clip(cal_of_rest, 0.05, 0.5))
    train_local, cal_local = stratified_train_val_indices(rest_labels, cal_of_rest, seed + 7)
    train_p = [rest_idx[i] for i in train_local]
    cal_p = [rest_idx[i] for i in cal_local]

    train_patients = {uniq[i] for i in train_p}
    cal_patients = {uniq[i] for i in cal_p}
    val_patients = {uniq[i] for i in val_p}
    if train_patients & cal_patients or train_patients & val_patients or cal_patients & val_patients:
        raise RuntimeError("Patient leaked across train/calibration/val")

    train_idx = [i for i, p in enumerate(patient_ids) if p in train_patients]
    cal_idx = [i for i, p in enumerate(patient_ids) if p in cal_patients]
    val_idx = [i for i, p in enumerate(patient_ids) if p in val_patients]
    return train_idx, cal_idx, val_idx


def patient_level_train_cal_val_test_indices(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    cal_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Four-way patient split: train / calibration / val / test."""
    patient_ids = np.asarray(patient_ids)
    row_pos = _row_positive(labels)
    uniq, inv = np.unique(patient_ids, return_inverse=True)
    if len(uniq) < 8:
        raise ValueError(f"Need at least 8 patients for 4-way split; got {len(uniq)}")

    p_labels = np.zeros(len(uniq), dtype=np.float32)
    for row_i, p_i in enumerate(inv):
        if row_pos[row_i] >= 0.5:
            p_labels[p_i] = 1.0

    hold_frac = float(np.clip(test_fraction, 0.05, 0.35))
    rest_p, test_p = stratified_train_val_indices(p_labels, hold_frac, seed)
    rest_labels = p_labels[np.asarray(rest_p)]
    # Fractions of the *full* cohort, re-expressed on the non-test remainder.
    scale = max(1e-6, 1.0 - hold_frac)
    val_of_rest = float(np.clip(val_fraction / scale, 0.05, 0.45))
    # Carve val from remainder, then cal from what's left.
    mid_p, val_p_local = stratified_train_val_indices(rest_labels, val_of_rest, seed + 1)
    mid_labels = rest_labels[np.asarray(mid_p)]
    cal_of_mid = float(np.clip(cal_fraction / max(1e-6, scale - val_fraction), 0.05, 0.5))
    train_local, cal_local = stratified_train_val_indices(mid_labels, cal_of_mid, seed + 2)

    mid_p_arr = np.asarray(mid_p, dtype=np.int64)
    rest_p_arr = np.asarray(rest_p, dtype=np.int64)
    train_patients = {uniq[int(rest_p_arr[int(mid_p_arr[i])])] for i in train_local}
    cal_patients = {uniq[int(rest_p_arr[int(mid_p_arr[i])])] for i in cal_local}
    val_patients = {uniq[int(rest_p_arr[i])] for i in val_p_local}
    test_patients = {uniq[i] for i in test_p}
    if (
        train_patients & cal_patients
        or train_patients & val_patients
        or train_patients & test_patients
        or cal_patients & val_patients
        or cal_patients & test_patients
        or val_patients & test_patients
    ):
        raise RuntimeError("Patient leaked across train/cal/val/test")

    train_idx = [i for i, p in enumerate(patient_ids) if p in train_patients]
    cal_idx = [i for i, p in enumerate(patient_ids) if p in cal_patients]
    val_idx = [i for i, p in enumerate(patient_ids) if p in val_patients]
    test_idx = [i for i, p in enumerate(patient_ids) if p in test_patients]
    return train_idx, cal_idx, val_idx, test_idx


def nested_calibration_from_val(
    patient_ids: np.ndarray,
    val_idx: list[int],
    labels: np.ndarray,
    cal_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split an existing val fold into (cal_fit, val_eval) without patient leakage.

    Returns ``(calibration_idx, evaluation_idx)`` as indices into the original dataset.
    """
    patient_ids = np.asarray(patient_ids)
    val_idx = list(val_idx)
    if len(val_idx) < 4:
        raise ValueError("Need at least 4 val rows to nest calibration")
    sub_pids = patient_ids[np.asarray(val_idx)]
    sub_labels = np.asarray(labels)[np.asarray(val_idx)]
    frac = float(np.clip(cal_fraction, 0.1, 0.5))
    uniq = np.unique(sub_pids)
    if len(uniq) < 4:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(val_idx))
        n_cal = max(1, int(round(len(val_idx) * frac)))
        n_cal = min(n_cal, len(val_idx) - 1)
        cal_local = perm[:n_cal].tolist()
        eval_local = perm[n_cal:].tolist()
    else:
        # val_fraction=frac → second return is the calibration fold
        eval_local, cal_local = patient_level_train_val_indices(
            sub_pids, sub_labels, frac, seed
        )
    cal_idx = [val_idx[i] for i in cal_local]
    eval_idx = [val_idx[i] for i in eval_local]
    assert_calibration_disjoint(cal_idx, eval_idx)
    return cal_idx, eval_idx


def make_subsets(dataset, train_idx: list[int], val_idx: list[int]) -> tuple[Subset, Subset]:
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def split_hash(
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int] | None = None,
    calibration_idx: list[int] | None = None,
    patient_ids: np.ndarray | None = None,
) -> str:
    payload = {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
        "test": sorted(int(i) for i in (test_idx or [])),
        "calibration": sorted(int(i) for i in (calibration_idx or [])),
        "patients": (
            [str(p) for p in np.asarray(patient_ids).tolist()]
            if patient_ids is not None
            else None
        ),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def save_split(
    path: str,
    train_idx: list[int],
    val_idx: list[int],
    seed: int,
    val_fraction: float,
    test_idx: list[int] | None = None,
    patient_ids: np.ndarray | None = None,
    calibration_idx: list[int] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: dict = {
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
        "seed": np.int64(seed),
        "val_fraction": np.float64(val_fraction),
        "split_hash": np.asarray(
            split_hash(train_idx, val_idx, test_idx, calibration_idx, patient_ids)
        ),
    }
    if test_idx is not None:
        payload["test_idx"] = np.asarray(test_idx, dtype=np.int64)
    if calibration_idx is not None:
        payload["calibration_idx"] = np.asarray(calibration_idx, dtype=np.int64)
    if patient_ids is not None:
        payload["patient_ids"] = np.asarray(patient_ids)
    # Disjointness check
    sets = {
        "train": set(train_idx),
        "val": set(val_idx),
    }
    if calibration_idx is not None:
        sets["calibration"] = set(calibration_idx)
    if test_idx is not None:
        sets["test"] = set(test_idx)
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            leak = sets[a] & sets[b]
            if leak:
                raise RuntimeError(f"Index leak between {a} and {b}: {len(leak)} rows")
    np.savez_compressed(path, **payload)


def load_split(path: str) -> tuple[list[int], list[int]]:
    data = np.load(path, allow_pickle=True)
    return data["train_idx"].tolist(), data["val_idx"].tolist()


def load_test_idx(path: str) -> list[int] | None:
    data = np.load(path, allow_pickle=True)
    if "test_idx" not in data.files:
        return None
    return data["test_idx"].tolist()


def load_calibration_idx(path: str) -> list[int] | None:
    data = np.load(path, allow_pickle=True)
    if "calibration_idx" not in data.files:
        return None
    return data["calibration_idx"].tolist()


def load_split_hash(path: str) -> str | None:
    data = np.load(path, allow_pickle=True)
    if "split_hash" not in data.files:
        return None
    v = data["split_hash"]
    if getattr(v, "shape", ()) == ():
        return str(v.item())
    return str(v)


def resolve_split_indices(
    split: SplitName,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int] | None = None,
    calibration_idx: list[int] | None = None,
) -> list[int]:
    if split == "train":
        return train_idx
    if split == "val":
        return val_idx
    if split in ("calibration", "cal"):
        if calibration_idx is None:
            raise ValueError("split='calibration' requires calibration_idx in split.npz")
        return calibration_idx
    if split == "test":
        if test_idx is None:
            raise ValueError("split='test' requires test_idx in split.npz")
        return test_idx
    # all
    parts = list(train_idx) + list(val_idx)
    if calibration_idx is not None:
        parts = parts + list(calibration_idx)
    if test_idx is not None:
        parts = parts + list(test_idx)
    return parts


def subset_for_split(
    dataset,
    split: SplitName,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int] | None = None,
    calibration_idx: list[int] | None = None,
) -> Subset:
    idx = resolve_split_indices(split, train_idx, val_idx, test_idx, calibration_idx)
    return Subset(dataset, idx)
