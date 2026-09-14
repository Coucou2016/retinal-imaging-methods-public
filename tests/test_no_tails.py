"""No-tails closure tests: E5 backbone router, MultiCohort, DeLong, predict path."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from model.RetiPioneer import get_reti_pioneer
from model.quality_gate import QualityBackboneRouter
from reti_pioneer.config import build_model_from_config, load_config
from reti_pioneer.joint_vocab import JOINT_VOCAB, build_joint_projection, project_labels_to_joint
from reti_pioneer.split import assert_split_label_coverage
from utils.bootstrap import delong_auroc_ci


class TestE5BackboneRouter(unittest.TestCase):
    def test_softmax_over_backbones(self):
        router = QualityBackboneRouter(n_backbones=3, enabled=True)
        q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        w = router(q)
        self.assertEqual(tuple(w.shape), (2, 3))
        self.assertTrue(torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-5))

    def test_e5_uses_backbone_router_not_ensemble_max(self):
        model = get_reti_pioneer(
            True,
            num_classes=2,
            learnable_q=True,
            quality_router="monotone",
            quality_gating=True,
            lambda_q=0.1,
        )
        self.assertTrue(model.quality_gating)
        self.assertTrue(model.backbone_router.enabled)
        B = 3
        l = [torch.randn(B, 1024), torch.randn(B, 1024), torch.randn(B, 384)]
        r = [torch.randn(B, 1024), torch.randn(B, 1024), torch.randn(B, 384)]
        m = torch.randn(B, 10)
        q = (torch.ones(B, 3) / 3, torch.ones(B, 3) / 3)
        out = model(((l, r), m, q))
        self.assertEqual(tuple(out.shape), (B, 2))
        self.assertIsNotNone(model._last_backbone_weights)
        self.assertEqual(tuple(model._last_backbone_weights.shape), (B, 3))
        aux = model.collect_quality_aux_logits()
        self.assertIsNotNone(aux)
        self.assertEqual(tuple(aux.shape), (B, 3))


class TestJointVocabMultiCohort(unittest.TestCase):
    def test_project_odir_masks_systemic(self):
        proj = build_joint_projection("odir", ["N", "D", "G", "C", "A", "H", "M", "O"])
        y = np.array([[0, 1, 0, 0, 0, 1, 0, 0]], dtype=np.float32)
        joint = project_labels_to_joint(y, proj)
        self.assertEqual(joint.shape, (1, len(JOINT_VOCAB)))
        # diabetes_ocular from D, hypertension_ocular from H, systemic missing
        self.assertEqual(float(joint[0, 0]), 1.0)
        self.assertEqual(float(joint[0, 1]), -1.0)
        self.assertEqual(float(joint[0, 2]), 1.0)

    def test_multicohort_dataset_smoke(self):
        from dataset.multi_cohort import MultiCohortDataset
        from reti_pioneer.data_paths import ukb_compressed_ready

        roots = {
            "odir": os.path.join(ROOT, "data", "odir"),
            "brset": os.path.join(ROOT, "data", "brset"),
            "rfmid": os.path.join(ROOT, "data", "rfmid"),
        }
        present = {k: v for k, v in roots.items() if ukb_compressed_ready(v)}
        if len(present) < 2:
            self.skipTest("need ≥2 public caches")
        ds = MultiCohortDataset(
            present,
            ["baselineage", "gender", "weight", "ethnicity"],
            pretrain=["RETF", "SwinB", "VimS"],
        )
        ds.set_target(0, list(JOINT_VOCAB))
        self.assertGreaterEqual(len(ds), 4)
        sample = ds[0]
        y = sample[-1]
        self.assertEqual(int(np.asarray(y).shape[-1]), len(JOINT_VOCAB))


class TestDeLongAndCoverage(unittest.TestCase):
    def test_delong_ci_covers_point(self):
        rng = np.random.default_rng(0)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 5, dtype=np.float64)
        p = y * 0.8 + rng.uniform(0.05, 0.15, size=y.size)
        out = delong_auroc_ci(y, p)
        self.assertTrue(np.isfinite(out["estimate"]))
        self.assertLessEqual(out["ci_low"], out["estimate"] + 1e-9)
        self.assertGreaterEqual(out["ci_high"], out["estimate"] - 1e-9)

    def test_assert_split_missing_class_fails(self):
        y = np.zeros((10, 2), dtype=np.float32)
        y[:, 0] = 1.0  # class0 all-pos
        y[:, 1] = np.array([0, 1] * 5, dtype=np.float32)
        with self.assertRaises(ValueError):
            assert_split_label_coverage(y, list(range(10)), fold_name="test")


class TestAllConfigsBuildAndOneEpoch(unittest.TestCase):
    def test_every_yaml_builds(self):
        cfg_dir = Path(ROOT) / "configs"
        for path in sorted(cfg_dir.glob("*.yaml")):
            cfg = load_config(path)
            model = build_model_from_config(cfg)
            self.assertIsNotNone(model)

    def test_e5_one_epoch_synthetic_smoke(self):
        # Tiny forward+backward to prove E5+λ_q path is wired.
        model = get_reti_pioneer(
            True,
            num_classes=2,
            learnable_q=True,
            quality_router="monotone",
            quality_gating=True,
            lambda_q=0.1,
            quality_aux=True,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        B = 4
        l = [torch.randn(B, 1024), torch.randn(B, 1024), torch.randn(B, 384)]
        r = [torch.randn(B, 1024), torch.randn(B, 1024), torch.randn(B, 384)]
        m = torch.randn(B, 10)
        q = (torch.ones(B, 3) / 3, torch.ones(B, 3) / 3)
        y = torch.tensor([[1.0, -1.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
        from utils.run import masked_bce_with_logits
        from model.quality_gate import soft_quality_ce

        opt.zero_grad()
        logits = model(((l, r), m, q))
        loss = masked_bce_with_logits(logits, y)
        aux = model.collect_quality_aux_logits()
        if aux is not None:
            loss = loss + 0.1 * soft_quality_ce(aux, 0.5 * (q[0] + q[1]))
        loss.backward()
        opt.step()
        self.assertTrue(torch.isfinite(loss.detach()))


class TestPredictExtensionImport(unittest.TestCase):
    def test_script_imports(self):
        import scripts.predict_extension as pe

        self.assertTrue(callable(pe.main))


if __name__ == "__main__":
    unittest.main()
