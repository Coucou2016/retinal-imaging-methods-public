"""Canonical head mapping for cross-dataset evaluation.

These pairs are *related* screening targets, not identical gold standards.
ODIR D is ocular-evidence diabetes; BRSET diabetes is a clinical/self-report
diagnosis; RFMiD DR is a retinal sign. Never report a mapped AUROC as UKB T2DM.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical task -> {dataset_key: label name as stored in UKB_y*.npz `yn`}
CROSS_TASKS: dict[str, dict[str, str]] = {
    "diabetes_related": {
        "odir": "D",
        "brset": "diabetes",
        "rfmid": "DR",
        "ukb": "t2dm",
        "demo": "t2dm",
    },
    "diabetes_ocular": {
        "odir": "D",
        "brset": "dr_referable",
        "rfmid": "DR",
    },
    "hypertension_ocular": {
        "odir": "H",
        "brset": "hypertensive_retinopathy",
        # RFMiD 1.0 default columns have no HR/HTN — do not invent a mapping.
        "ukb": "hypertension",
        "demo": "hypertension",
    },
}

DEFAULT_CROSS_HEADS = ("diabetes_related", "hypertension_ocular")

# Extra aliases if a cache used long names instead of ODIR letters.
_NAME_ALIASES = {
    "D": ("D", "diabetes_ocular", "diabetes"),
    "H": ("H", "hypertension_ocular", "hypertension"),
    "DR": ("DR", "dr", "diabetic_retinopathy"),
    "HR": ("HR", "HTN", "hypertensive_retinopathy", "hypertensive retinopathy"),
    "diabetes": ("diabetes", "diabetes_mellitus"),
    "dr_referable": ("dr_referable", "DR", "referable_dr"),
    "hypertensive_retinopathy": ("hypertensive_retinopathy", "HR", "HTN"),
    "t2dm": ("t2dm", "T2D", "diabetes"),
    "hypertension": ("hypertension", "H"),
}


@dataclass(frozen=True)
class MappedHead:
    task: str
    train_name: str
    test_name: str
    train_index: int
    test_index: int


def _index_in(names: list[str], wanted: str) -> int | None:
    lower = [n.lower() for n in names]
    for cand in _NAME_ALIASES.get(wanted, (wanted,)):
        if cand.lower() in lower:
            return lower.index(cand.lower())
    if wanted.lower() in lower:
        return lower.index(wanted.lower())
    return None


def resolve_cross_heads(
    train_names: list[str],
    test_names: list[str],
    train_dataset: str,
    test_dataset: str,
    tasks: list[str] | tuple[str, ...] | None = None,
) -> list[MappedHead]:
    """Return overlapping canonical heads that exist in both label vectors."""
    if tasks is None:
        tasks = DEFAULT_CROSS_HEADS
    mapped: list[MappedHead] = []
    for task in tasks:
        spec = CROSS_TASKS.get(task)
        if spec is None:
            raise ValueError(f"Unknown cross-dataset task {task!r}; known: {list(CROSS_TASKS)}")
        src = spec.get(train_dataset)
        dst = spec.get(test_dataset)
        if not src or not dst:
            continue
        ti = _index_in(train_names, src)
        vi = _index_in(test_names, dst)
        if ti is None or vi is None:
            print(
                f"[label_map] skip task={task}: "
                f"{train_dataset}:{src!r}->{ti} / {test_dataset}:{dst!r}->{vi}"
            )
            continue
        mapped.append(
            MappedHead(
                task=task,
                train_name=train_names[ti],
                test_name=test_names[vi],
                train_index=ti,
                test_index=vi,
            )
        )
    return mapped


def _as_2d(arr):
    import numpy as np

    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def take_columns(arr, indices: list[int]):
    import numpy as np

    arr = _as_2d(arr)
    return np.stack([arr[:, i] for i in indices], axis=1)


def slice_mapped(
    logits_or_probs,
    labels,
    heads: list[MappedHead],
):
    """Slice model outputs (train K) and test labels (test K) onto shared heads."""
    out_p = take_columns(logits_or_probs, [h.train_index for h in heads])
    out_y = take_columns(labels, [h.test_index for h in heads])
    return out_p, out_y
