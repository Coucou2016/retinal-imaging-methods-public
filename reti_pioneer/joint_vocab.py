"""Joint partial-label vocabulary across ODIR / BRSET / RFMiD.

Maps each cohort's local label columns onto a shared endpoint vocabulary.
Missing endpoints are marked ``-1`` so ``masked_bce`` supervises only present labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Canonical joint heads used for MultiCohort training (endpoint-aware).
JOINT_VOCAB: tuple[str, ...] = (
    "diabetes_ocular",
    "diabetes_systemic",
    "hypertension_ocular",
)

# Local label name → joint endpoint id, per dataset.
DATASET_LOCAL_TO_JOINT: dict[str, dict[str, str]] = {
    "odir": {
        "D": "diabetes_ocular",
        "H": "hypertension_ocular",
        "diabetes": "diabetes_ocular",
        "hypertension": "hypertension_ocular",
    },
    "brset": {
        "dr_referable": "diabetes_ocular",
        "diabetes": "diabetes_systemic",
        "hypertensive_retinopathy": "hypertension_ocular",
    },
    "rfmid": {
        "DR": "diabetes_ocular",
        "dr": "diabetes_ocular",
    },
    "demo": {
        "t2dm": "diabetes_systemic",
        "hypertension": "hypertension_ocular",
    },
    "ukb": {
        "t2dm": "diabetes_systemic",
        "hypertension": "hypertension_ocular",
    },
}


@dataclass(frozen=True)
class JointProjection:
    dataset: str
    local_names: tuple[str, ...]
    # For each joint column: local index or -1 if absent.
    local_indices: tuple[int, ...]
    joint_names: tuple[str, ...] = JOINT_VOCAB


def build_joint_projection(dataset: str, local_names: list[str] | tuple[str, ...]) -> JointProjection:
    mapping = DATASET_LOCAL_TO_JOINT.get(str(dataset).lower(), {})
    lower = {str(n).lower(): i for i, n in enumerate(local_names)}
    indices: list[int] = []
    for joint in JOINT_VOCAB:
        # Prefer explicit local→joint reverse lookup.
        local_idx = -1
        for local, jid in mapping.items():
            if jid == joint and local.lower() in lower:
                local_idx = lower[local.lower()]
                break
        # Also accept joint id as a local column name.
        if local_idx < 0 and joint.lower() in lower:
            local_idx = lower[joint.lower()]
        indices.append(local_idx)
    return JointProjection(
        dataset=str(dataset).lower(),
        local_names=tuple(str(x) for x in local_names),
        local_indices=tuple(indices),
        joint_names=JOINT_VOCAB,
    )


def project_labels_to_joint(
    labels: np.ndarray,
    projection: JointProjection,
    missing_value: float = -1.0,
) -> np.ndarray:
    """Project (N, K_local) → (N, K_joint) with missing_value where unmapped."""
    y = np.asarray(labels, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n = y.shape[0]
    out = np.full((n, len(projection.joint_names)), float(missing_value), dtype=np.float32)
    for j, li in enumerate(projection.local_indices):
        if li < 0:
            continue
        if li >= y.shape[1]:
            continue
        out[:, j] = y[:, li]
    return out


def joint_mask_from_labels(joint_labels: np.ndarray) -> np.ndarray:
    """Boolean present-mask (True = supervised)."""
    y = np.asarray(joint_labels, dtype=np.float32)
    return np.isfinite(y) & (y >= 0)
