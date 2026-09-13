"""Calibration and decision-curve helpers (paper evaluation metrics)."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_curve


def _as_1d(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), 0.0, 1.0)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true and y_prob shape mismatch: {y_true.shape} vs {y_prob.shape}")
    return y_true, y_prob


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true, y_prob = _as_1d(y_true, y_prob)
    if y_true.size == 0:
        return float("nan")
    return float(brier_score_loss(y_true, y_prob))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width ECE; empty bins are skipped. Finite on small synthetic sets."""
    y_true, y_prob = _as_1d(y_true, y_prob)
    n = int(y_true.size)
    if n == 0:
        return float("nan")
    n_bins = max(1, min(int(n_bins), n))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        m = int(mask.sum())
        if m == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (m / n) * abs(acc - conf)
    return float(ece)


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_prob = _as_1d(y_true, y_prob)
    n_bins = max(1, min(int(n_bins), max(1, y_true.size)))
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
    except ValueError:
        return np.array([]), np.array([])
    return prob_true, prob_pred


def treat_all_net_benefit(y_true: np.ndarray, threshold: float = 0.10) -> float:
    """Net benefit of treating everyone at threshold (may be negative)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    if y_true.size == 0:
        return float("nan")
    pt = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    prevalence = float(y_true.mean())
    return float(prevalence - (1.0 - prevalence) * (pt / (1.0 - pt)))


def treat_none_net_benefit(y_true: np.ndarray | None = None, threshold: float = 0.10) -> float:
    """Net benefit of treating no one (always 0 by definition)."""
    del y_true, threshold
    return 0.0


def decision_curve_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Decision-curve net benefit (binary). Returns raw NB; may be negative."""
    y_true, y_prob = _as_1d(y_true, y_prob)
    n = len(y_true)
    if n == 0:
        return np.full(len(np.asarray(thresholds)), np.nan)
    benefits = []
    for pt in np.asarray(thresholds, dtype=np.float64):
        pt = float(np.clip(pt, 1e-6, 1.0 - 1e-6))
        pred = y_prob >= pt
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        nb = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        benefits.append(float(nb))
    return np.array(benefits, dtype=np.float64)


def net_benefit_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.10) -> float:
    arr = decision_curve_net_benefit(y_true, y_prob, np.array([threshold], dtype=np.float64))
    return float(arr[0]) if arr.size else float("nan")


def _finite_threshold(thr: float, y_prob: np.ndarray) -> float:
    """roc_curve may emit +inf for the first threshold; map to a usable cut."""
    if np.isfinite(thr):
        return float(thr)
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    if y_prob.size == 0:
        return 1.0
    return float(np.max(y_prob) + 1e-6)


def _roc_sens_spec(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    y_true, y_prob = _as_1d(y_true, y_prob)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    sens = np.asarray(tpr, dtype=np.float64)
    spec = 1.0 - np.asarray(fpr, dtype=np.float64)
    return sens, spec, np.asarray(thresholds, dtype=np.float64)


def youden_operating_point(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Threshold maximizing Youden's J = sensitivity + specificity - 1."""
    nan = {
        "threshold": float("nan"),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "youden_j": float("nan"),
    }
    roc = _roc_sens_spec(y_true, y_prob)
    if roc is None:
        return nan
    sens, spec, thresholds = roc
    j = sens + spec - 1.0
    idx = int(np.argmax(j))
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    return {
        "threshold": _finite_threshold(float(thresholds[idx]), y_prob),
        "sensitivity": float(sens[idx]),
        "specificity": float(spec[idx]),
        "youden_j": float(j[idx]),
    }


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_specificity: float = 0.95,
) -> dict[str, float]:
    """Sensitivity at specificity >= target, else nearest specificity on the ROC.

    When multiple ROC points meet the specificity floor, pick highest sensitivity.
    """
    nan = {
        "threshold": float("nan"),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "target_specificity": float(target_specificity),
        "met_target": float("nan"),
    }
    roc = _roc_sens_spec(y_true, y_prob)
    if roc is None:
        return nan
    sens, spec, thresholds = roc
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    target = float(np.clip(target_specificity, 0.0, 1.0))
    qualify = np.where(spec >= target - 1e-12)[0]
    if qualify.size:
        # Highest sensitivity among points meeting the floor; tie-break: closest spec.
        sub_sens = sens[qualify]
        best_local = int(np.argmax(sub_sens))
        ties = qualify[np.isclose(sub_sens, sub_sens[best_local])]
        idx = int(ties[int(np.argmin(np.abs(spec[ties] - target)))])
        met = 1.0
    else:
        idx = int(np.argmin(np.abs(spec - target)))
        met = 0.0
    return {
        "threshold": _finite_threshold(float(thresholds[idx]), y_prob),
        "sensitivity": float(sens[idx]),
        "specificity": float(spec[idx]),
        "target_specificity": target,
        "met_target": met,
    }


def _bce_with_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(y_true, dtype=np.float64).ravel()
    # stable: max(z,0) - z*y + log(1+exp(-|z|))
    return float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit a scalar temperature T>0 on validation logits (shared across classes)."""
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if logits.size == 0:
        return 1.0

    def nll(t: float) -> float:
        t = max(float(t), 1e-3)
        return _bce_with_logits(logits / t, y_true)

    try:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded", options={"xatol": 1e-4})
        t = float(res.x)
    except Exception:
        grid = np.linspace(0.05, 10.0, 64)
        t = float(grid[int(np.argmin([nll(v) for v in grid]))])
    return float(np.clip(t, 0.05, 10.0))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return sigmoid(logits / T)."""
    t = max(float(temperature), 1e-3)
    z = np.asarray(logits, dtype=np.float64) / t
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def fit_calibration_intercept_slope(
    logits: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float]:
    """Fit logistic intercept/slope: P = sigmoid(slope * logit + intercept).

    Returns ``(intercept, slope)``. Falls back to (0, 1) on degenerate data.
    """
    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(y_true, dtype=np.float64).ravel()
    if z.size < 4 or len(np.unique(y)) < 2:
        return 0.0, 1.0
    # Design matrix [1, z]
    x = np.column_stack([np.ones_like(z), z])

    def nll(beta: np.ndarray) -> float:
        return _bce_with_logits(x @ beta, y)

    try:
        from scipy.optimize import minimize

        res = minimize(nll, x0=np.array([0.0, 1.0]), method="L-BFGS-B")
        intercept, slope = float(res.x[0]), float(res.x[1])
    except Exception:
        intercept, slope = 0.0, 1.0
    slope = float(np.clip(slope, 1e-3, 50.0))
    return intercept, slope


def apply_intercept_slope(
    logits: np.ndarray,
    intercept: float,
    slope: float,
) -> np.ndarray:
    z = slope * np.asarray(logits, dtype=np.float64) + float(intercept)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def per_class_decision_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Per-disease DCA curves; preferred over macro NB@0.10 as primary narrative."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.50, 10)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out: dict[str, dict[str, list[float]]] = {}
    if y_true.ndim == 1:
        names = class_names or ["class_0"]
        nb = decision_curve_net_benefit(y_true, y_prob, thresholds)
        out[names[0]] = {
            "thresholds": thresholds.tolist(),
            "net_benefit": nb.tolist(),
            "treat_all": [treat_all_net_benefit(y_true, float(t)) for t in thresholds],
            "treat_none": [0.0 for _ in thresholds],
        }
        return out
    for k in range(y_true.shape[1]):
        name = class_names[k] if class_names and k < len(class_names) else f"class_{k}"
        nb = decision_curve_net_benefit(y_true[:, k], y_prob[:, k], thresholds)
        out[name] = {
            "thresholds": thresholds.tolist(),
            "net_benefit": nb.tolist(),
            "treat_all": [treat_all_net_benefit(y_true[:, k], float(t)) for t in thresholds],
            "treat_none": [0.0 for _ in thresholds],
        }
    return out
