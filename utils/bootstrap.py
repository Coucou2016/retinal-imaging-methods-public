"""Patient-level bootstrap confidence intervals for discrimination metrics."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score


def _unique_patients(patient_ids: np.ndarray) -> np.ndarray:
    pids = np.asarray(patient_ids)
    # Preserve first-seen order for determinism.
    _, idx = np.unique(pids, return_index=True)
    return pids[np.sort(idx)]


def _patient_mask(patient_ids: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """Boolean mask of rows belonging to any patient in ``chosen`` (with multiplicity via concat)."""
    # For bootstrap with replacement we concat per-draw patient blocks.
    blocks = []
    pid_arr = np.asarray(patient_ids)
    for p in chosen:
        blocks.append(np.where(pid_arr == p)[0])
    if not blocks:
        return np.array([], dtype=np.int64)
    return np.concatenate(blocks)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim == 1:
        return _safe_auroc(y_true, y_score)
    vals = [_safe_auroc(y_true[:, k], y_score[:, k]) for k in range(y_true.shape[1])]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def patient_level_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    patient_ids: np.ndarray,
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Patient-level bootstrap CI: resample patients with replacement, score rows.

    Returns point estimate on the full sample plus percentile CI.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    patient_ids = np.asarray(patient_ids)
    if len(patient_ids) != (len(y_true) if y_true.ndim == 1 else y_true.shape[0]):
        raise ValueError("patient_ids length must match number of rows")
    fn = metric_fn or _macro_auroc
    point = float(fn(y_true, y_score))
    patients = _unique_patients(patient_ids)
    if patients.size < 2 or n_boot < 1:
        return {
            "estimate": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot": 0,
            "n_patients": int(patients.size),
            "alpha": float(alpha),
        }
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(n_boot)):
        draw = rng.choice(patients, size=patients.size, replace=True)
        idx = _patient_mask(patient_ids, draw)
        if idx.size == 0:
            samples.append(float("nan"))
            continue
        if y_true.ndim == 1:
            samples.append(float(fn(y_true[idx], y_score[idx])))
        else:
            samples.append(float(fn(y_true[idx], y_score[idx])))
    arr = np.asarray(samples, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        lo = hi = float("nan")
    else:
        lo = float(np.quantile(finite, alpha / 2.0))
        hi = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return {
        "estimate": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": int(n_boot),
        "n_boot_finite": int(finite.size),
        "n_patients": int(patients.size),
        "alpha": float(alpha),
    }


def delong_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float]:
    """DeLong asymptotic AUROC 95% CI (binary). Falls back to NaN CI if degenerate.

    Implements the structural-components variance of DeLong et al. (1988) for a
    single classifier. For multilabel input, returns macro-mean of per-class CIs
    estimates (estimate/ci averaged over finite classes).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim == 2:
        rows = [
            delong_auroc_ci(y_true[:, k], y_score[:, k], alpha=alpha)
            for k in range(y_true.shape[1])
        ]
        finite = [r for r in rows if np.isfinite(r["estimate"])]
        if not finite:
            return {
                "estimate": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "se": float("nan"),
                "method": "delong_macro",
                "alpha": float(alpha),
                "n_classes": int(y_true.shape[1]),
            }
        return {
            "estimate": float(np.mean([r["estimate"] for r in finite])),
            "ci_low": float(np.mean([r["ci_low"] for r in finite if np.isfinite(r["ci_low"])])),
            "ci_high": float(np.mean([r["ci_high"] for r in finite if np.isfinite(r["ci_high"])])),
            "se": float(np.mean([r["se"] for r in finite if np.isfinite(r["se"])])),
            "method": "delong_macro",
            "alpha": float(alpha),
            "n_classes": int(y_true.shape[1]),
            "n_classes_finite": len(finite),
        }

    y = np.asarray(y_true, dtype=np.float64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    pos = s[y >= 0.5]
    neg = s[y < 0.5]
    m, n = int(pos.size), int(neg.size)
    point = _safe_auroc(y, s)
    if m < 1 or n < 1 or not np.isfinite(point):
        return {
            "estimate": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "se": float("nan"),
            "method": "delong",
            "alpha": float(alpha),
            "n_pos": m,
            "n_neg": n,
        }
    # Structural components V10 / V01
    v10 = np.empty(m, dtype=np.float64)
    for i, xi in enumerate(pos):
        v10[i] = (np.sum(neg < xi) + 0.5 * np.sum(neg == xi)) / n
    v01 = np.empty(n, dtype=np.float64)
    for j, yj in enumerate(neg):
        v01[j] = (np.sum(pos > yj) + 0.5 * np.sum(pos == yj)) / m
    s10 = float(np.var(v10, ddof=1)) if m > 1 else 0.0
    s01 = float(np.var(v01, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(s10 / m + s01 / n))
    try:
        from scipy.stats import norm

        z = float(norm.ppf(1.0 - alpha / 2.0))
    except Exception:
        z = 1.95996398454  # Phi^{-1}(0.975)
    lo = float(np.clip(point - z * se, 0.0, 1.0))
    hi = float(np.clip(point + z * se, 0.0, 1.0))
    return {
        "estimate": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "se": se,
        "method": "delong",
        "alpha": float(alpha),
        "n_pos": m,
        "n_neg": n,
    }


def paired_delta_auroc_bootstrap(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    patient_ids: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired patient-level bootstrap CI for ΔAUROC = AUROC(A) − AUROC(B)."""
    y_true = np.asarray(y_true)
    ya = np.asarray(y_score_a)
    yb = np.asarray(y_score_b)
    patient_ids = np.asarray(patient_ids)
    n = len(y_true) if y_true.ndim == 1 else y_true.shape[0]
    if not (len(ya) == n and len(yb) == n and len(patient_ids) == n):
        raise ValueError("y_true / scores / patient_ids length mismatch")

    point = _macro_auroc(y_true, ya) - _macro_auroc(y_true, yb)
    patients = _unique_patients(patient_ids)
    if patients.size < 2 or n_boot < 1:
        return {
            "delta_auroc": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boot": 0,
            "n_patients": int(patients.size),
            "alpha": float(alpha),
            "auroc_a": _macro_auroc(y_true, ya),
            "auroc_b": _macro_auroc(y_true, yb),
        }
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(n_boot)):
        draw = rng.choice(patients, size=patients.size, replace=True)
        idx = _patient_mask(patient_ids, draw)
        if y_true.ndim == 1:
            d = _safe_auroc(y_true[idx], ya[idx]) - _safe_auroc(y_true[idx], yb[idx])
        else:
            d = _macro_auroc(y_true[idx], ya[idx]) - _macro_auroc(y_true[idx], yb[idx])
        samples.append(float(d))
    arr = np.asarray(samples, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        lo = hi = float("nan")
    else:
        lo = float(np.quantile(finite, alpha / 2.0))
        hi = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return {
        "delta_auroc": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": int(n_boot),
        "n_boot_finite": int(finite.size),
        "n_patients": int(patients.size),
        "alpha": float(alpha),
        "auroc_a": _macro_auroc(y_true, ya),
        "auroc_b": _macro_auroc(y_true, yb),
    }
