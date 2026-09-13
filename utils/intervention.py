"""Intervention / counterfactual quality-vector helpers (smoke tests)."""

from __future__ import annotations

from typing import Callable

import numpy as np


def canonical_quality_vectors() -> dict[str, np.ndarray]:
    """One-hot-ish prototypes: good / usable / bad."""
    return {
        "good": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "usable": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "bad": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }


def confidence_monotonicity_report(
    score_fn: Callable[[np.ndarray], float],
) -> dict[str, float | bool]:
    """Check that score(good) ≥ score(usable) ≥ score(bad) for a scalar confidence.

    ``score_fn`` maps a length-3 quality probability vector to a scalar
    (e.g. model confidence, gate magnitude, or router weight).
    """
    vecs = canonical_quality_vectors()
    sg = float(score_fn(vecs["good"]))
    su = float(score_fn(vecs["usable"]))
    sb = float(score_fn(vecs["bad"]))
    mono = (sg + 1e-8 >= su) and (su + 1e-8 >= sb)
    return {
        "score_good": sg,
        "score_usable": su,
        "score_bad": sb,
        "monotonic": bool(mono),
        "gap_good_usable": sg - su,
        "gap_usable_bad": su - sb,
    }
