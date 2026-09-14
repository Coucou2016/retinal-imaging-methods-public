"""Shared helpers for public fundus manifests → UKB-style records."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PatientRecord:
    patient_id: str
    age: float
    sex: float  # 0 female / 1 male (UKB-style); 0 if unknown
    labels: np.ndarray
    left_path: str | None = None
    right_path: str | None = None
    weight: float = 0.0  # missing on public sets; padded for UKB_mqd layout
    ethnicity: int = 0  # missing on public sets; index 0 of 7-way one-hot
    ql: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.2, 0.1], dtype=np.float32))
    qr: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.2, 0.1], dtype=np.float32))
    split: str | None = None  # train / val / test when the source provides it


def find_column(row: dict, *names: str, default: str | None = None) -> str | None:
    if not row:
        return default
    keys = {str(k).strip(): k for k in row}
    lower = {str(k).strip().lower(): k for k in row}
    for name in names:
        if name in keys:
            val = row[keys[name]]
            if val is not None and str(val).strip() != "":
                return str(val).strip()
        if name.lower() in lower:
            val = row[lower[name.lower()]]
            if val is not None and str(val).strip() != "":
                return str(val).strip()
    return default


def parse_float(value, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_sex(value) -> float:
    """Map common sex encodings to UKB-style {0, 1} (1 = male)."""
    if value is None:
        return 0.0
    s = str(value).strip().lower()
    if s in {"1", "m", "male", "man"}:
        return 1.0
    if s in {"0", "2", "f", "female", "woman"}:
        return 0.0
    return parse_float(value, 0.0)


def parse_binary(value) -> float:
    if value is None or str(value).strip() == "":
        return 0.0
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "positive"}:
        return 1.0
    if s in {"0", "false", "no", "n", "negative"}:
        return 0.0
    try:
        return 1.0 if float(s) >= 0.5 else 0.0
    except ValueError:
        return 0.0


def default_quality() -> np.ndarray:
    return np.array([0.7, 0.2, 0.1], dtype=np.float32)


def quality_from_scores(good=None, usable=None, bad=None, **named) -> np.ndarray:
    """Build a 3-way (good, usable, bad) quality vector.

    Named extras (focus/illumination/artifacts) from BRSET are mapped when the
    3-way EyeQ-style scores are absent: high artifact → more mass on `bad`.
    """
    if good is not None or usable is not None or bad is not None:
        q = np.array(
            [parse_float(good, 0.7), parse_float(usable, 0.2), parse_float(bad, 0.1)],
            dtype=np.float32,
        )
    else:
        focus = parse_float(named.get("focus"), 1.0)
        illum = parse_float(named.get("illumination"), 1.0)
        art = parse_float(named.get("artifacts"), 0.0)
        good_s = 0.5 * (focus + illum) * (1.0 - 0.5 * art)
        bad_s = 0.2 + 0.6 * art
        usable_s = max(0.05, 1.0 - good_s - bad_s)
        q = np.array([good_s, usable_s, bad_s], dtype=np.float32)
    s = float(q.sum())
    if s <= 0:
        return default_quality()
    return (q / s).astype(np.float32)


def ukb_meta_matrix(records: list[PatientRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Pad public metadata into UKB_mqd columns: age, gender, weight, ethnicity."""
    m = np.column_stack(
        [
            np.array([r.age for r in records], dtype=np.float32),
            np.array([r.sex for r in records], dtype=np.float32),
            np.array([r.weight for r in records], dtype=np.float32),
            np.array([r.ethnicity for r in records], dtype=np.float32),
        ]
    )
    mn = np.array(["baselineage", "gender", "weight", "ethnicity"])
    return m, mn


def synthetic_backbone_features(
    labels: np.ndarray,
    dim: int,
    seed: int,
    stream: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic fake RETF/Swin/Vim features for CI when images are absent.

    SYNTHETIC: not fundus-derived. A weak linear signal of the labels is mixed
    into Gaussian noise so a smoke-train can overfit tiny N. Do not report as
    a clinical AUROC.
    """
    labels = np.asarray(labels, dtype=np.float32)
    n, k = labels.shape
    rng = np.random.default_rng(int(seed) + 1009 * int(stream))
    left = rng.standard_normal((n, dim)).astype(np.float32) * 0.25
    right = rng.standard_normal((n, dim)).astype(np.float32) * 0.25
    k_use = min(k, dim)
    left[:, :k_use] += labels[:, :k_use]
    right[:, :k_use] += labels[:, :k_use]
    return left, right


_IMAGE_INDEX: dict[str, dict[str, "Path"]] = {}


def _basename_index(root: "Path") -> dict[str, "Path"]:
    """Build basename → path map once per root (skips rm_images sample folders)."""
    key = str(root.resolve())
    if key in _IMAGE_INDEX:
        return _IMAGE_INDEX[key]
    index: dict[str, "Path"] = {}
    for hit in root.rglob("*"):
        if not hit.is_file():
            continue
        if hit.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            continue
        if "rm_images" in str(hit).lower():
            continue
        # Prefer Training/Validation/Test paths over duplicates.
        name = hit.name
        if name not in index:
            index[name] = hit
        else:
            cur = str(index[name]).lower()
            new = str(hit).lower()
            score = lambda s: (
                ("/training/" in s or "\\training\\" in s)
                + 2 * ("/validation/" in s or "\\validation\\" in s)
                + 3 * ("/test/" in s or "\\test\\" in s)
            )
            # Keep first; do not overwrite — callers with split context should pass fuller relative paths.
            del score
    _IMAGE_INDEX[key] = index
    return index


def resolve_image_path(path: str | None, root: str | Path | None = None) -> Path | None:
    """Resolve a record path against an optional dataset root.

    For RFMiD-style bare filenames (``1.png``), also searches common nested
    folders under ``root`` (Training / Validation / Test).
    """
    from pathlib import Path as _Path

    if path is None or str(path).strip() == "":
        return None
    p = _Path(path)
    if p.is_file():
        return p
    if root is None:
        return None
    root_p = _Path(root)
    cand = root_p / path
    if cand.is_file():
        return cand
    name = p.name
    # RFMiD HF layout: Training_Set/Training_Set/Training/<id>.png etc.
    for rel in (
        _Path("Training_Set") / "Training_Set" / "Training" / name,
        _Path("Evaluation_Set") / "Evaluation_Set" / "Validation" / name,
        _Path("Test_Set") / "Test_Set" / "Test" / name,
        _Path("Training") / name,
        _Path("Validation") / name,
        _Path("Test") / name,
        _Path("images") / name,
    ):
        hit = root_p / rel
        if hit.is_file():
            return hit
    index = _basename_index(root_p)
    return index.get(name)


def pair_paths_for_record(
    record: PatientRecord,
    root: str | Path | None = None,
) -> tuple[str, str] | None:
    """Return (left, right) absolute paths; duplicate the available eye if one is missing.

    Returns None when neither eye resolves to an existing file.
    """
    left = resolve_image_path(record.left_path, root)
    right = resolve_image_path(record.right_path, root)
    if left is None and right is None:
        return None
    if left is None:
        left = right
    if right is None:
        right = left
    assert left is not None and right is not None
    return str(left.resolve()), str(right.resolve())


def write_pairs_manifest(
    records: list[PatientRecord],
    out_path: str | Path,
    root: str | Path | None = None,
    skip_missing: bool = True,
) -> list[int]:
    """Write ``left_path,right_path`` CSV aligned with ``records`` order.

    Returns the list of record indices that were written. When ``skip_missing``
    is True, rows without any readable image are omitted (caller must keep
    labels/mqd in the same filtered order, or extract only after filtering).
    When False, missing pairs raise FileNotFoundError.
    """
    from pathlib import Path as _Path

    out_path = _Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept: list[int] = []
    lines = ["# left_path,right_path  # same row order as filtered PatientRecord list\n"]
    for i, rec in enumerate(records):
        pair = pair_paths_for_record(rec, root)
        if pair is None:
            if skip_missing:
                continue
            raise FileNotFoundError(
                f"No image for patient_id={rec.patient_id!r} "
                f"(left={rec.left_path!r} right={rec.right_path!r})"
            )
        kept.append(i)
        lines.append(f"{pair[0]},{pair[1]}\n")
    out_path.write_text("".join(lines), encoding="utf-8")
    return kept


def load_pairs_manifest(path: str | Path) -> list[tuple[str, str]]:
    """Parse a pairs.csv written by ``write_pairs_manifest`` or hand-authored."""
    from pathlib import Path as _Path

    pairs: list[tuple[str, str]] = []
    with open(_Path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                raise ValueError(f"Bad manifest line (need left,right): {line!r}")
            pairs.append((parts[0], parts[1]))
    return pairs


def assert_feature_row_count(cache_dir: str | Path, n_expected: int) -> None:
    """Ensure every UKB_* backbone npz (if present) has n_expected rows."""
    from pathlib import Path as _Path

    cache_dir = _Path(cache_dir)
    for name in ("UKB_RETF.npz", "UKB_swin.npz", "UKB_vim.npz"):
        path = cache_dir / name
        if not path.is_file():
            continue
        with np.load(path) as data:
            left = data["left"]
            right = data["right"]
            if left.shape[0] != n_expected or right.shape[0] != n_expected:
                raise ValueError(
                    f"{path.name} has left={left.shape} right={right.shape}; "
                    f"expected N={n_expected} to match UKB_mqd / labels"
                )
