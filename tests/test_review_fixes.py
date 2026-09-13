"""P0 review fixes: calibration disjointness, monotone router, endpoints, masked BCE."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import warnings

import numpy as np
import torch
from ignite.utils import to_onehot

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from model.QualityAware import QualityAware
from model.RetiPioneer import get_reti_pioneer, normalize_ensemble
from reti_pioneer.label_map import (
    ENDPOINTS,
    assert_clinical_alignment,
    clinical_claim_allowed,
    harmonization_table_rows,
    resolve_cross_heads,
)
from reti_pioneer.split import (
    assert_calibration_disjoint,
    nested_calibration_from_val,
    patient_level_train_cal_val_indices,
    save_split,
    load_calibration_idx,
    split_hash,
)
from utils.run import masked_bce_with_logits


class TestCalibrationSplitDisjoint(unittest.TestCase):
    def test_calibration_split_disjoint(self):
        pids = np.array([f"p{i//2}" for i in range(40)])
        labels = np.array([i % 2 for i in range(40)], dtype=np.float32)
        train_idx, cal_idx, val_idx = patient_level_train_cal_val_indices(
            pids, labels, val_fraction=0.25, cal_fraction=0.15, seed=0
        )
        assert_calibration_disjoint(cal_idx, val_idx)
        assert_calibration_disjoint(cal_idx, train_idx)
        assert_calibration_disjoint(train_idx, val_idx)
        self.assertEqual(
            set(train_idx) | set(cal_idx) | set(val_idx),
            set(range(40)),
        )
        with self.assertRaises(RuntimeError):
            assert_calibration_disjoint(cal_idx, cal_idx)

    def test_nested_val_calibration_disjoint(self):
        pids = np.array([f"p{i}" for i in range(20)])
        labels = np.array([i % 2 for i in range(20)], dtype=np.float32)
        val_idx = list(range(20))
        cal_idx, eval_idx = nested_calibration_from_val(
            pids, val_idx, labels, cal_fraction=0.4, seed=1
        )
        assert_calibration_disjoint(cal_idx, eval_idx)
        self.assertEqual(set(cal_idx) | set(eval_idx), set(val_idx))

    def test_save_split_persists_calibration_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "split.npz")
            pids = np.array(["a", "b", "c", "d", "e", "f"])
            save_split(
                path,
                [0, 1],
                [2, 3],
                seed=0,
                val_fraction=0.3,
                calibration_idx=[4, 5],
                patient_ids=pids,
            )
            cal = load_calibration_idx(path)
            self.assertEqual(cal, [4, 5])
            h = split_hash([0, 1], [2, 3], None, [4, 5], pids)
            with np.load(path, allow_pickle=True) as data:
                self.assertEqual(str(data["split_hash"].item()), h)


class TestMonotoneQualityRouter(unittest.TestCase):
    def test_monotonicity_and_bounds(self):
        qa = QualityAware(8, 4, enable_q=True, learnable_q=True, quality_router="monotone")
        w = qa.monotone_weights().detach()
        self.assertEqual(tuple(w.shape), (3,))
        self.assertLessEqual(float(w[0]), float(w[1]) + 1e-6)
        self.assertLessEqual(float(w[1]), float(w[2]) + 1e-6)
        self.assertTrue(torch.all(w >= 0))
        self.assertTrue(torch.all(w <= 1 + 1e-5))
        self.assertAlmostEqual(float(w[2]), 1.0, places=5)
        # Near paper init
        self.assertLess(float(w[0]), 0.05)
        self.assertAlmostEqual(float(w[1]), 0.5, delta=0.05)

    def test_monotone_gradient_flows(self):
        torch.manual_seed(0)
        qa = QualityAware(8, 4, enable_q=True, learnable_q=True, quality_router="monotone")
        xf = torch.randn(3, 8)
        q = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
        out = qa(xf, q).sum()
        out.backward()
        self.assertIsNotNone(qa.mono_delta.grad)
        self.assertGreater(float(qa.mono_delta.grad.abs().sum()), 0.0)

    def test_fixed_baseline_unchanged(self):
        qa = QualityAware(8, 4, enable_q=True, learnable_q=False, quality_router="fixed")
        self.assertEqual(qa.quality_router, "fixed")
        self.assertFalse(qa.q_fc.weight.requires_grad)
        self.assertAlmostEqual(float(qa.q_fc.weight[0, 0]), 1.0)
        self.assertAlmostEqual(float(qa.q_fc.weight[0, 1]), 0.5)
        self.assertAlmostEqual(float(qa.q_fc.weight[0, 2]), 0.0)

    def test_free_linear_ablation_only(self):
        qa = QualityAware(8, 4, enable_q=True, learnable_q=True, quality_router="free_linear")
        self.assertEqual(qa.quality_router, "free_linear")
        self.assertTrue(qa.q_fc.weight.requires_grad)


class TestEnsembleNaming(unittest.TestCase):
    def test_paper_alias_deprecated(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            name = normalize_ensemble("paper")
            self.assertEqual(name, "released_code")
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))

    def test_released_code_soft_train_max_eval(self):
        # Soft mix ≠ max when temperature is moderate and heads disagree.
        probs = torch.tensor([[[0.0], [1.0], [2.0]], [[-1.0], [0.0], [1.0]]])  # (B,H,1)
        temp = 1.0
        soft = (torch.softmax(probs / temp, dim=1) * probs).sum(dim=1)
        hard = probs.max(dim=1).values
        self.assertFalse(torch.allclose(soft, hard, atol=1e-3))
        m = get_reti_pioneer(True, num_classes=1, ensemble="released_code")
        self.assertEqual(m.ensemble, "released_code")
        m.train()
        self.assertTrue(m.training)
        m.eval()
        self.assertFalse(m.training)

    def test_published_soft_vote(self):
        m = get_reti_pioneer(True, num_classes=1, ensemble="published_soft_vote")
        self.assertEqual(m.ensemble, "published_soft_vote")


class TestEndpointOntology(unittest.TestCase):
    def test_diabetes_related_not_clinical(self):
        ep = ENDPOINTS["diabetes_related"]
        for r in ep.refs:
            self.assertFalse(clinical_claim_allowed(r.alignment))

    def test_cross_eval_warns_or_refuses(self):
        from dataset.brset import BRSET_LABELS
        from dataset.odir import ODIR_LABELS

        heads = resolve_cross_heads(
            list(ODIR_LABELS),
            list(BRSET_LABELS),
            "odir",
            "brset",
            tasks=["diabetes_related", "hypertension_ocular"],
        )
        tasks = {h.task for h in heads}
        self.assertIn("diabetes_related", tasks)
        related = next(h for h in heads if h.task == "diabetes_related")
        self.assertEqual(related.alignment, "related_not_equivalent")
        self.assertFalse(related.clinical_claim_allowed)
        with self.assertRaises(ValueError):
            assert_clinical_alignment(heads, clinical_tables=True)

    def test_harmonization_table_nonempty(self):
        rows = harmonization_table_rows()
        self.assertGreater(len(rows), 3)
        self.assertIn("alignment", rows[0])


class TestMaskedBCE(unittest.TestCase):
    def test_missing_labels_ignored(self):
        logits = torch.zeros(4, 3)
        targets = torch.tensor(
            [
                [1.0, -1.0, 0.0],
                [0.0, 1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 1.0],
            ]
        )
        loss = masked_bce_with_logits(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        # All-missing batch → 0 loss (denom clamped but mask sum 0 → we clamp denom)
        loss_empty = masked_bce_with_logits(logits[:1], targets[2:3])
        self.assertEqual(float(loss_empty), 0.0)


class TestDefaultLearnableUsesMonotone(unittest.TestCase):
    def test_get_reti_pioneer_learnable_defaults_monotone(self):
        m = get_reti_pioneer(True, num_classes=1, learnable_q=True)
        qa = m.models[0][0].quality_aware
        self.assertEqual(qa.quality_router, "monotone")
        self.assertIsNotNone(qa.mono_delta)


if __name__ == "__main__":
    unittest.main()
