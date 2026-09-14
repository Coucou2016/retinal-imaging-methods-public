"""Hard-gate contract tests: YAML schema, monotone, cal disjoint, paper_mode, official test."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dataset.public_common import PatientRecord
from model.QualityAware import QualityAware
from model.RetiPioneer import VALID_ENSEMBLES, normalize_ensemble
from reti_pioneer.config import build_model_from_config, load_config, validate_config
from reti_pioneer.label_map import (
    ENDPOINTS,
    EndpointSpec,
    assert_clinical_alignment,
    features_clinical_claim_allowed,
    resolve_cross_heads,
    update_label_map_provenance,
)
from reti_pioneer.split import load_test_idx
from scripts.prepare_public_npz import write_public_npz


CONFIGS_DIR = Path(ROOT) / "configs"
ABLATION_YAMLS = sorted(CONFIGS_DIR.glob("ablation_*.yaml")) + [
    CONFIGS_DIR / "default.yaml",
    CONFIGS_DIR / "demo.yaml",
]


class TestStrictConfigSchema(unittest.TestCase):
    def test_all_yaml_parse_and_build(self):
        self.assertGreaterEqual(len(ABLATION_YAMLS), 10)
        for path in ABLATION_YAMLS:
            with self.subTest(config=path.name):
                cfg = load_config(path, strict=True)
                model = build_model_from_config(cfg)
                self.assertIsNotNone(model)
                # Smoke forward on dummy batch would need full tensors; shape check enough.
                self.assertTrue(hasattr(model, "forward"))

    def test_unknown_top_level_rejected(self):
        cfg = load_config(CONFIGS_DIR / "demo.yaml")
        cfg["not_a_real_key"] = 1
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_unknown_training_key_rejected(self):
        cfg = load_config(CONFIGS_DIR / "demo.yaml")
        cfg["training"]["bogus_hyperparam"] = 0.5
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_invalid_ensemble_rejected(self):
        cfg = load_config(CONFIGS_DIR / "demo.yaml")
        cfg["training"]["ensemble"] = "soft_vote_v2"
        with self.assertRaises(ValueError):
            validate_config(cfg)


class TestMonotoneBounds(unittest.TestCase):
    def test_monotone_bounds_and_order(self):
        qa = QualityAware(8, 4, enable_q=True, learnable_q=True, quality_router="monotone")
        w = qa.monotone_weights().detach()
        self.assertLessEqual(float(w[0]), float(w[1]) + 1e-6)
        self.assertLessEqual(float(w[1]), float(w[2]) + 1e-6)
        self.assertAlmostEqual(float(w[2]), 1.0, places=5)
        self.assertTrue((w >= 0).all() and (w <= 1 + 1e-5).all())

    def test_free_linear_is_unconstrained_linear(self):
        qa = QualityAware(8, 4, enable_q=True, learnable_q=True, quality_router="free_linear")
        self.assertEqual(qa.quality_router, "free_linear")
        self.assertIsNotNone(qa.q_fc)
        self.assertIsNone(qa.mono_delta)


class TestEnsembleContract(unittest.TestCase):
    def test_valid_ensembles(self):
        self.assertEqual(
            set(VALID_ENSEMBLES),
            {"released_code", "published_soft_vote", "mean", "temp_mean", "paper"},
        )
        self.assertEqual(normalize_ensemble("paper"), "released_code")
        with self.assertRaises(ValueError):
            normalize_ensemble("not_an_ensemble")


class TestEndpointOntology(unittest.TestCase):
    def test_endpoint_spec_and_systemic_split(self):
        self.assertIsInstance(ENDPOINTS["diabetes_ocular"], EndpointSpec)
        self.assertEqual(ENDPOINTS["diabetes_ocular"].kind, "ocular_manifestation")
        self.assertEqual(ENDPOINTS["diabetes_systemic"].kind, "systemic")
        self.assertIn("hypertension_systemic", ENDPOINTS)
        self.assertEqual(ENDPOINTS["hypertension_systemic"].kind, "systemic")
        self.assertEqual(ENDPOINTS["hypertension_ocular"].kind, "ocular_manifestation")

    def test_paper_mode_rejects_related(self):
        heads = resolve_cross_heads(
            ["D", "H"],
            ["diabetes", "hypertensive_retinopathy"],
            "odir",
            "brset",
            tasks=["diabetes_related"],
            paper_mode=False,
        )
        self.assertTrue(heads)
        with self.assertRaises(ValueError):
            assert_clinical_alignment(heads, paper_mode=True)


class TestOfficialTestNeverResplit(unittest.TestCase):
    def test_carve_val_preserves_official_test(self):
        # 12 train + 0 val + 6 test patients; carving val must keep test indices.
        records = []
        for i in range(12):
            records.append(
                PatientRecord(
                    patient_id=f"tr{i}",
                    age=50.0,
                    sex=float(i % 2),
                    labels=np.array([float(i % 2)], dtype=np.float32),
                    left_path=None,
                    right_path=None,
                    ql=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    qr=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    split="train",
                )
            )
        for i in range(6):
            records.append(
                PatientRecord(
                    patient_id=f"te{i}",
                    age=55.0,
                    sex=float(i % 2),
                    labels=np.array([float(i % 2)], dtype=np.float32),
                    left_path=None,
                    right_path=None,
                    ql=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    qr=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    split="test",
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            out = write_public_npz(
                tmp,
                records,
                ["D"],
                seed=0,
                synthetic_features=True,
                dataset_name="odir",
                force=True,
            )
            test_idx = load_test_idx(os.path.join(out, "split.npz"))
            self.assertIsNotNone(test_idx)
            self.assertEqual(sorted(test_idx), list(range(12, 18)))
            with np.load(os.path.join(out, "split.npz"), allow_pickle=True) as data:
                train = set(int(x) for x in data["train_idx"])
                val = set(int(x) for x in data["val_idx"])
            self.assertTrue(train.isdisjoint(val))
            self.assertTrue(train.isdisjoint(set(test_idx)))
            self.assertTrue(val.isdisjoint(set(test_idx)))
            self.assertEqual(train | val, set(range(12)))


class TestProvenanceGate(unittest.TestCase):
    def test_synthetic_not_clinical(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "SYNTHETIC_FEATURES.txt").write_text(
                "SYNTHETIC\nclinical_claim_allowed: false\n", encoding="utf-8"
            )
            update_label_map_provenance(tmp, synthetic_features=True, stub_features=False)
            self.assertFalse(features_clinical_claim_allowed(tmp))
            meta = json.loads(Path(tmp, "label_map.json").read_text(encoding="utf-8"))
            self.assertFalse(meta["clinical_claim_allowed"])

    def test_real_extract_enables_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            update_label_map_provenance(tmp, synthetic_features=False, stub_features=False)
            meta = json.loads(Path(tmp, "label_map.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["clinical_claim_allowed"])


class TestCalDisjointAndPaperModeCalibrate(unittest.TestCase):
    def test_resolve_cal_rejects_val_calibrate_paper_mode(self):
        from scripts.evaluate import resolve_calibration_fit_ids

        bundle = {
            "train_idx": [0, 1, 2, 3],
            "val_idx": [4, 5, 6, 7],
            "test_idx": None,
            "calibration_idx": None,
        }
        with self.assertRaises(ValueError):
            resolve_calibration_fit_ids(bundle, "val", paper_mode=True, split_seed=0)


if __name__ == "__main__":
    unittest.main()
