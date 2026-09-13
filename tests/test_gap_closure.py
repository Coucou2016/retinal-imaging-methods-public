"""Gap-closure tests: threshold lock, bootstrap CI, quality aux, intervention, E5 gate."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from model.RetiPioneer import get_reti_pioneer
from model.quality_gate import QualityConditionedGate, soft_quality_ce
from scripts.evaluate import (
    resolve_threshold_fit_ids,
    score_predictions,
)
from utils.bootstrap import paired_delta_auroc_bootstrap, patient_level_bootstrap_ci
from utils.calibration import (
    binary_metrics_at_threshold,
    fit_operating_point_thresholds,
)
from utils.intervention import confidence_monotonicity_report


class TestThresholdFitNotEval(unittest.TestCase):
    def test_fit_thresholds_differ_from_eval_application(self):
        rng = np.random.default_rng(0)
        # Fit fold: well-separated scores → high Youden threshold ~0.5
        y_fit = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
        p_fit = np.array([0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95], dtype=np.float64)
        # Eval fold: different score distribution
        y_eval = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
        p_eval = np.array([0.3, 0.4, 0.45, 0.55, 0.6, 0.7], dtype=np.float64)

        thr = fit_operating_point_thresholds(y_fit, p_fit)
        frozen = score_predictions(
            y_eval, p_eval, fit_labels=y_fit, fit_probs=p_fit, allow_fit_on_eval=False
        )
        leaky = score_predictions(y_eval, p_eval, allow_fit_on_eval=True)

        self.assertEqual(frozen["operating_points_source"], "fit_fold")
        self.assertEqual(leaky["operating_points_source"], "eval")
        # Frozen thresholds must equal fit-fold selection, not eval-fitted ones.
        self.assertAlmostEqual(
            frozen["youden_threshold"], thr["youden_threshold"], places=6
        )
        self.assertNotAlmostEqual(
            frozen["youden_threshold"], leaky["youden_threshold"], places=5
        )
        # Applied sens/spec use frozen cut on eval labels.
        applied = binary_metrics_at_threshold(
            y_eval, p_eval, thr["youden_threshold"]
        )
        self.assertAlmostEqual(frozen["youden_sensitivity"], applied["sensitivity"], places=6)
        self.assertAlmostEqual(frozen["youden_specificity"], applied["specificity"], places=6)

    def test_paper_mode_rejects_fit_on_eval(self):
        bundle = {
            "train_idx": [0, 1, 2, 3],
            "val_idx": [4, 5, 6, 7],
            "test_idx": None,
            "calibration_idx": None,
        }
        with self.assertRaises(ValueError):
            resolve_threshold_fit_ids(bundle, "val", paper_mode=True)

    def test_test_split_uses_val_for_thresholds(self):
        bundle = {
            "train_idx": [0, 1, 2, 3],
            "val_idx": [4, 5, 6, 7],
            "test_idx": [8, 9, 10, 11],
            "calibration_idx": None,
        }
        ids, src = resolve_threshold_fit_ids(bundle, "test", paper_mode=True)
        self.assertEqual(src, "val")
        self.assertEqual(ids, [4, 5, 6, 7])

    def test_allow_fit_on_eval_false_without_fit_raises(self):
        y = np.array([0, 1, 0, 1], dtype=np.float64)
        p = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64)
        with self.assertRaises(ValueError):
            score_predictions(y, p, allow_fit_on_eval=False)


class TestBootstrapCI(unittest.TestCase):
    def test_patient_level_bootstrap_covers_point(self):
        rng = np.random.default_rng(1)
        # 20 patients × 2 eyes; strong signal
        pids = np.array([f"p{i // 2}" for i in range(40)])
        y = np.array([i % 2 for i in range(40)], dtype=np.float64)
        p = y * 0.7 + rng.uniform(0.05, 0.25, size=40)
        out = patient_level_bootstrap_ci(y, p, pids, n_boot=200, seed=0)
        self.assertTrue(np.isfinite(out["estimate"]))
        self.assertTrue(np.isfinite(out["ci_low"]))
        self.assertTrue(np.isfinite(out["ci_high"]))
        self.assertLessEqual(out["ci_low"], out["estimate"] + 1e-9)
        self.assertGreaterEqual(out["ci_high"], out["estimate"] - 1e-9)
        self.assertEqual(out["n_patients"], 20)

    def test_paired_delta_bootstrap_sign(self):
        pids = np.array([f"p{i}" for i in range(30)])
        y = np.array([i % 2 for i in range(30)], dtype=np.float64)
        # Model A clearly better than chance-ish B
        pa = y * 0.9 + 0.05
        pb = np.full_like(y, 0.5)
        out = paired_delta_auroc_bootstrap(y, pa, pb, pids, n_boot=100, seed=2)
        self.assertGreater(out["delta_auroc"], 0.2)
        self.assertGreater(out["ci_low"], 0.0)


class TestQualityAuxAndGate(unittest.TestCase):
    def test_soft_quality_ce_prefers_correct(self):
        target = torch.tensor([[0.8, 0.15, 0.05], [0.1, 0.2, 0.7]])
        good = torch.log(target + 1e-8)
        bad = torch.zeros_like(good)
        self.assertLess(float(soft_quality_ce(good, target)), float(soft_quality_ce(bad, target)))

    def test_quality_gate_shapes_and_default_off(self):
        gate = QualityConditionedGate(feat_dim=8, enabled=True)
        xf = torch.randn(4, 8)
        q = F.softmax(torch.randn(4, 3), dim=-1)
        out = gate(xf, q)
        self.assertEqual(tuple(out.shape), (4, 8))
        # Identity path when disabled
        off = QualityConditionedGate(feat_dim=8, enabled=False)
        self.assertTrue(torch.allclose(off(xf, q), xf))

    def test_e5_flag_does_not_break_e0_shapes(self):
        m0 = get_reti_pioneer(True, num_classes=1, learnable_q=False, quality_gating=False)
        m5 = get_reti_pioneer(
            True, num_classes=1, learnable_q=True, quality_router="monotone", quality_gating=True
        )
        B = 2
        l = [torch.randn(B, 1024), torch.randn(B, 1024), torch.randn(B, 384)]
        r = [torch.randn(B, 1024), torch.randn(B, 1024), torch.randn(B, 384)]
        m = torch.randn(B, 10)
        q = (torch.ones(B, 3) / 3, torch.ones(B, 3) / 3)
        y0 = m0(((l, r), m, q))
        y5 = m5(((l, r), m, q))
        self.assertEqual(tuple(y0.shape), (B, 1))
        self.assertEqual(tuple(y5.shape), (B, 1))


class TestMonotoneScalarOrder(unittest.TestCase):
    def test_quality_scalar_good_gt_usable_gt_bad(self):
        from model.QualityAware import QualityAware

        qa = QualityAware(8, 4, enable_q=True, learnable_q=True, quality_router="monotone")
        with torch.no_grad():
            g = float(qa.quality_scalar(torch.tensor([[1.0, 0.0, 0.0]])).item())
            u = float(qa.quality_scalar(torch.tensor([[0.0, 1.0, 0.0]])).item())
            b = float(qa.quality_scalar(torch.tensor([[0.0, 0.0, 1.0]])).item())
        self.assertGreaterEqual(g, u - 1e-5)
        self.assertGreaterEqual(u, b - 1e-5)


class TestInterventionMonotonicity(unittest.TestCase):
    def test_good_usable_bad_confidence_order(self):
        # Synthetic scalar weight aligned to q=(good, usable, bad) → good > usable > bad
        def score_fn(q_vec: np.ndarray) -> float:
            w = np.array([1.0, 0.5, 0.0])
            return float(q_vec @ w)

        report = confidence_monotonicity_report(score_fn)
        self.assertTrue(report["monotonic"])
        self.assertGreater(report["score_good"], report["score_usable"])
        self.assertGreater(report["score_usable"], report["score_bad"])


if __name__ == "__main__":
    unittest.main()
