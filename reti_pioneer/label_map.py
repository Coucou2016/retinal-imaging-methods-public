"""Endpoint ontology and cross-dataset head mapping.

Endpoints are *related* screening targets, not identical gold standards.
ODIR D is ocular-evidence diabetes; BRSET diabetes is clinical/self-report;
RFMiD DR is a retinal sign. Never report a mapped AUROC as UKB T2DM ICD.

Alignment levels
----------------
- ``direct``: same clinical concept (safe for clinical-claim tables)
- ``partial``: overlapping but incomplete (warn; not clinical-claim by default)
- ``related_not_equivalent``: related screening proxies only (refuse clinical tables)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Alignment = Literal["direct", "partial", "related_not_equivalent"]
EndpointKind = Literal["systemic", "ocular_manifestation"]


@dataclass(frozen=True)
class DatasetLabelRef:
    """One dataset's column mapped to a canonical endpoint."""

    dataset: str
    label: str
    alignment: Alignment


@dataclass(frozen=True)
class EndpointSpec:
    """Canonical screening endpoint with kind + per-dataset alignment metadata."""

    id: str
    kind: EndpointKind
    description: str
    refs: tuple[DatasetLabelRef, ...]

    def ref_for(self, dataset: str) -> DatasetLabelRef | None:
        for r in self.refs:
            if r.dataset == dataset:
                return r
        return None

    def label_for(self, dataset: str) -> str | None:
        r = self.ref_for(dataset)
        return None if r is None else r.label

    def alignment_pair(self, train_dataset: str, test_dataset: str) -> Alignment | None:
        a = self.ref_for(train_dataset)
        b = self.ref_for(test_dataset)
        if a is None or b is None:
            return None
        order = {"direct": 0, "partial": 1, "related_not_equivalent": 2}
        return a.alignment if order[a.alignment] >= order[b.alignment] else b.alignment


# Backward-compatible alias used in older docs / imports.
Endpoint = EndpointSpec


# Canonical endpoints (pre-registered). Keep related mappings only with explicit flags.
ENDPOINTS: dict[str, EndpointSpec] = {
    "diabetes_ocular": EndpointSpec(
        id="diabetes_ocular",
        kind="ocular_manifestation",
        description=(
            "Ocular manifestation of diabetes / DR findings. "
            "Not UKB T2DM ICD."
        ),
        refs=(
            DatasetLabelRef("odir", "D", "direct"),
            DatasetLabelRef("brset", "dr_referable", "partial"),
            DatasetLabelRef("rfmid", "DR", "direct"),
        ),
    ),
    "diabetes_systemic": EndpointSpec(
        id="diabetes_systemic",
        kind="systemic",
        description=(
            "Systemic diabetes diagnosis when the dataset records it. "
            "Still not UKB T2DM ICD unless the UKB column is used."
        ),
        refs=(
            DatasetLabelRef("brset", "diabetes", "direct"),
            DatasetLabelRef("ukb", "t2dm", "direct"),
            DatasetLabelRef("demo", "t2dm", "direct"),
        ),
    ),
    # Kept for exploratory cross-eval only — never a clinical-claim head.
    "diabetes_related": EndpointSpec(
        id="diabetes_related",
        kind="systemic",
        description=(
            "Exploratory related screening proxies across ocular and systemic "
            "diabetes labels. alignment=related_not_equivalent by construction."
        ),
        refs=(
            DatasetLabelRef("odir", "D", "related_not_equivalent"),
            DatasetLabelRef("brset", "diabetes", "related_not_equivalent"),
            DatasetLabelRef("rfmid", "DR", "related_not_equivalent"),
            DatasetLabelRef("ukb", "t2dm", "related_not_equivalent"),
            DatasetLabelRef("demo", "t2dm", "related_not_equivalent"),
        ),
    ),
    "hypertension_ocular": EndpointSpec(
        id="hypertension_ocular",
        kind="ocular_manifestation",
        description=(
            "Hypertensive retinopathy / ocular hypertension signs. "
            "Not UKB systemic hypertension ICD."
        ),
        refs=(
            DatasetLabelRef("odir", "H", "direct"),
            DatasetLabelRef("brset", "hypertensive_retinopathy", "direct"),
            DatasetLabelRef("ukb", "hypertension", "related_not_equivalent"),
            DatasetLabelRef("demo", "hypertension", "related_not_equivalent"),
        ),
    ),
    "hypertension_systemic": EndpointSpec(
        id="hypertension_systemic",
        kind="systemic",
        description=(
            "Systemic hypertension diagnosis when the dataset records it. "
            "Distinct from hypertensive retinopathy ocular signs."
        ),
        refs=(
            DatasetLabelRef("ukb", "hypertension", "direct"),
            DatasetLabelRef("demo", "hypertension", "direct"),
            # BRSET records hypertensive retinopathy, not systemic HTN diagnosis.
            DatasetLabelRef("brset", "hypertensive_retinopathy", "related_not_equivalent"),
        ),
    ),
}

# Backward-compatible task → {dataset: label} view (alignment lives on Endpoint).
CROSS_TASKS: dict[str, dict[str, str]] = {
    eid: {r.dataset: r.label for r in ep.refs} for eid, ep in ENDPOINTS.items()
}

# Prefer ocular-direct pairs for default cross-eval; keep diabetes_related opt-in.
DEFAULT_CROSS_HEADS = ("hypertension_ocular", "diabetes_ocular")
# Legacy default used in older docs/CLIs (explicitly related, not clinical).
LEGACY_CROSS_HEADS = ("diabetes_related", "hypertension_ocular")

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
    alignment: Alignment = "related_not_equivalent"
    kind: EndpointKind = "systemic"
    clinical_claim_allowed: bool = False


def _index_in(names: list[str], wanted: str) -> int | None:
    lower = [n.lower() for n in names]
    for cand in _NAME_ALIASES.get(wanted, (wanted,)):
        if cand.lower() in lower:
            return lower.index(cand.lower())
    if wanted.lower() in lower:
        return lower.index(wanted.lower())
    return None


def get_endpoint(task: str) -> EndpointSpec:
    if task not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint {task!r}; known: {list(ENDPOINTS)}")
    return ENDPOINTS[task]


def clinical_claim_allowed(alignment: Alignment) -> bool:
    """Alignment-only gate. Feature provenance is a separate AND (see provenance helpers)."""
    return alignment == "direct"


def cache_features_are_real(cache_dir: str | os.PathLike[str] | Path) -> bool:
    """True only when backbone features look real (no SYNTHETIC/STUB marker)."""
    from pathlib import Path as _Path

    root = _Path(cache_dir)
    marker = root / "SYNTHETIC_FEATURES.txt"
    if marker.is_file():
        text = marker.read_text(encoding="utf-8", errors="ignore").lower()
        if "synthetic" in text or "stub" in text:
            return False
    meta_path = root / "label_map.json"
    if meta_path.is_file():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("synthetic_features") is True:
            return False
        if meta.get("stub_features") is True:
            return False
        if meta.get("clinical_claim_allowed") is True and not marker.is_file():
            return True
    # Real extract removes the marker; require all three backbone caches.
    from reti_pioneer.data_paths import ukb_compressed_ready

    return ukb_compressed_ready(str(root)) and not marker.is_file()


def features_clinical_claim_allowed(cache_dir: str | os.PathLike[str] | Path) -> bool:
    """clinical_claim_allowed is true only after real (non-synthetic) features exist."""
    return cache_features_are_real(cache_dir)


def update_label_map_provenance(
    cache_dir: str | os.PathLike[str] | Path,
    *,
    synthetic_features: bool,
    stub_features: bool = False,
    extra: dict | None = None,
) -> Path:
    """Merge provenance flags into ``label_map.json`` (create if missing)."""
    import json
    from pathlib import Path as _Path

    root = _Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "label_map.json"
    meta: dict = {}
    if path.is_file():
        meta = json.loads(path.read_text(encoding="utf-8"))
    meta["synthetic_features"] = bool(synthetic_features)
    meta["stub_features"] = bool(stub_features)
    # Real features required before any clinical claim flag may be true.
    meta["clinical_claim_allowed"] = (not synthetic_features) and (not stub_features)
    if extra:
        meta.update(extra)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def assert_clinical_alignment(
    heads: list[MappedHead],
    *,
    clinical_tables: bool = False,
    paper_mode: bool = False,
) -> None:
    """Refuse or warn when non-direct alignments enter clinical tables."""
    bad = [h for h in heads if not clinical_claim_allowed(h.alignment)]
    if not bad:
        return
    detail = ", ".join(f"{h.task}({h.alignment})" for h in bad)
    msg = (
        f"[endpoint] non-direct alignment for clinical tables: {detail}. "
        "Related/partial mappings are exploratory only — not UKB ICD equivalents."
    )
    if clinical_tables or paper_mode:
        raise ValueError(msg)
    warnings.warn(msg, stacklevel=2)
    print(f"WARNING: {msg}")


def resolve_cross_heads(
    train_names: list[str],
    test_names: list[str],
    train_dataset: str,
    test_dataset: str,
    tasks: list[str] | tuple[str, ...] | None = None,
    *,
    require_clinical: bool = False,
    paper_mode: bool = False,
) -> list[MappedHead]:
    """Return overlapping canonical heads that exist in both label vectors."""
    if tasks is None:
        tasks = DEFAULT_CROSS_HEADS
    mapped: list[MappedHead] = []
    for task in tasks:
        ep = ENDPOINTS.get(task)
        if ep is None:
            raise ValueError(f"Unknown cross-dataset task {task!r}; known: {list(ENDPOINTS)}")
        src_ref = ep.ref_for(train_dataset)
        dst_ref = ep.ref_for(test_dataset)
        if src_ref is None or dst_ref is None:
            continue
        ti = _index_in(train_names, src_ref.label)
        vi = _index_in(test_names, dst_ref.label)
        if ti is None or vi is None:
            print(
                f"[label_map] skip task={task}: "
                f"{train_dataset}:{src_ref.label!r}->{ti} / {test_dataset}:{dst_ref.label!r}->{vi}"
            )
            continue
        alignment = ep.alignment_pair(train_dataset, test_dataset) or "related_not_equivalent"
        mapped.append(
            MappedHead(
                task=task,
                train_name=train_names[ti],
                test_name=test_names[vi],
                train_index=ti,
                test_index=vi,
                alignment=alignment,
                kind=ep.kind,
                clinical_claim_allowed=clinical_claim_allowed(alignment),
            )
        )
    if require_clinical or paper_mode:
        assert_clinical_alignment(mapped, clinical_tables=True, paper_mode=paper_mode)
    elif mapped:
        assert_clinical_alignment(mapped, clinical_tables=False)
    return mapped


def harmonization_table_rows() -> list[dict[str, str]]:
    """Rows for manuscript / PUBLIC_DATA endpoint harmonization tables."""
    rows: list[dict[str, str]] = []
    for ep in ENDPOINTS.values():
        for r in ep.refs:
            rows.append(
                {
                    "endpoint": ep.id,
                    "kind": ep.kind,
                    "dataset": r.dataset,
                    "label": r.label,
                    "alignment": r.alignment,
                    "clinical_claim_allowed": str(clinical_claim_allowed(r.alignment)).lower(),
                    "description": ep.description,
                }
            )
    return rows


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
