"""Follow-up paper pieces: multi-task model, public loaders, calibration, splits."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from ignite.utils import to_onehot

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dataset.UKBDataset import UKBDatasetFast
from dataset.brset import BRSET_LABELS, make_synthetic_brset_records, parse_brset_csv
from dataset.odir import ODIR_LABELS, make_synthetic_odir_records, parse_odir_csv
from dataset.rfmid import parse_rfmid_csv
from model.RetiPioneer import get_reti_pioneer
from reti_pioneer.label_map import resolve_cross_heads, slice_mapped
from reti_pioneer.split import dataset_patient_ids, patient_level_train_val_indices
from scripts.prepare_public_npz import write_public_npz
from utils.calibration import (
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    net_benefit_at_threshold,
    sensitivity_at_specificity,
    youden_operating_point,
)
from utils.functional import pb_l_r_m_q_y


def _batch(b=4, k=1):
    l = [torch.randn(b, 1024), torch.randn(b, 1024), torch.randn(b, 384)]
    r = [x.clone() for x in l]
    m = torch.cat(
        [torch.tensor([[50.0, 1.0, 70.0]] * b), to_onehot(torch.zeros(b, dtype=torch.long), 7)],
        1,
    )
    ql = torch.tensor([[0.7, 0.2, 0.1]] * b)
    return ((l, r), m, (ql, ql.clone())), torch.zeros(b, k)


def _q_modules(model):
    return [seq[0].quality_aware for seq in model.models]


class TestMultitaskShapes(unittest.TestCase):
    def test_k1_and_k_gt1_forward(self):
        device = torch.device("cpu")
        xmq, _ = _batch(4, 1)
        m1 = get_reti_pioneer(True, num_classes=1).to(device)
        m1.train()
        self.assertEqual(m1(xmq).shape, (4, 1))
        m1.eval()
        self.assertEqual(m1(xmq).shape, (4, 1))

        xmq8, _ = _batch(4, 8)
        m8 = get_reti_pioneer(True, num_classes=8).to(device)
        m8.train()
        self.assertEqual(m8(xmq8).shape, (4, 8))
        m8.eval()
        self.assertEqual(m8(xmq8).shape, (4, 8))

    def test_learnable_q_grad_vs_frozen(self):
        torch.manual_seed(0)
        xmq, _ = _batch(3, 1)
        frozen = get_reti_pioneer(True, num_classes=1, learnable_q=False)
        learn = get_reti_pioneer(
            True, num_classes=1, learnable_q=True, quality_router="free_linear"
        )
        for qa in _q_modules(frozen):
            self.assertFalse(qa.q_fc.weight.requires_grad)
        for qa in _q_modules(learn):
            self.assertTrue(qa.q_fc.weight.requires_grad)

        w0 = _q_modules(frozen)[0].q_fc.weight.detach().clone()
        frozen.train()
        opt_f = torch.optim.SGD(frozen.parameters(), lr=0.5)
        opt_f.zero_grad()
        frozen(xmq).sum().backward()
        opt_f.step()
        self.assertTrue(torch.equal(w0, _q_modules(frozen)[0].q_fc.weight))

        q_fc = _q_modules(learn)[0].q_fc
        w1 = q_fc.weight.detach().clone()
        learn.train()
        opt_l = torch.optim.SGD(q_fc.parameters(), lr=0.5)
        opt_l.zero_grad()
        learn(xmq).sum().backward()
        self.assertIsNotNone(q_fc.weight.grad)
        self.assertGreater(float(q_fc.weight.grad.detach().abs().sum()), 0.0)
        opt_l.step()
        self.assertFalse(torch.equal(w1, q_fc.weight))


class TestPatientLevelSplit(unittest.TestCase):
    def test_same_patient_never_in_train_and_val(self):
        pids = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"])
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=np.float32)
        train_idx, val_idx = patient_level_train_val_indices(pids, labels, 0.3, seed=0)
        train_p = set(pids[train_idx])
        val_p = set(pids[val_idx])
        self.assertTrue(train_p.isdisjoint(val_p))
        self.assertEqual(train_p | val_p, set(pids))
        self.assertGreater(len(val_p), 0)
        self.assertGreater(len(train_p), 0)


class TestCalibrationFinite(unittest.TestCase):
    def test_metrics_finite_on_synthetic_labels(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=80).astype(np.float64)
        p = np.clip(y * 0.6 + rng.random(80) * 0.4, 0, 1)
        ece = expected_calibration_error(y, p, n_bins=10)
        brier = brier_score(y, p)
        nb = net_benefit_at_threshold(y, p, 0.10)
        self.assertTrue(np.isfinite(ece))
        self.assertTrue(np.isfinite(brier))
        self.assertTrue(np.isfinite(nb))
        logits = np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1 - 1e-4))
        t = fit_temperature(logits, y)
        cal = apply_temperature(logits, t)
        self.assertTrue(np.isfinite(t) and t > 0)
        self.assertTrue(np.isfinite(cal).all())
        self.assertAlmostEqual(expected_calibration_error(y, y), 0.0, places=5)


class TestOperatingPoints(unittest.TestCase):
    def test_youden_on_perfect_separator(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
        p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
        op = youden_operating_point(y, p)
        self.assertAlmostEqual(op["youden_j"], 1.0, places=5)
        self.assertAlmostEqual(op["sensitivity"], 1.0, places=5)
        self.assertAlmostEqual(op["specificity"], 1.0, places=5)
        self.assertTrue(0.4 < op["threshold"] <= 0.6 + 1e-9)

    def test_sens_at_95_spec_prefers_high_spec_floor(self):
        # Negatives clustered low; positives high — can hit >=95% specificity.
        y = np.array([0] * 20 + [1] * 20, dtype=np.float64)
        p = np.concatenate(
            [np.linspace(0.01, 0.40, 20), np.linspace(0.55, 0.99, 20)]
        ).astype(np.float64)
        s95 = sensitivity_at_specificity(y, p, target_specificity=0.95)
        self.assertGreaterEqual(s95["specificity"], 0.95 - 1e-9)
        self.assertEqual(s95["met_target"], 1.0)
        self.assertTrue(0.0 <= s95["sensitivity"] <= 1.0)
        self.assertTrue(np.isfinite(s95["threshold"]))

    def test_sens_at_95_spec_met_via_high_specificity_corner(self):
        # Even with few negatives, ROC includes the predict-all-negative corner (spec=1).
        y = np.array([0, 0, 1, 1, 1, 1], dtype=np.float64)
        p = np.array([0.2, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
        s95 = sensitivity_at_specificity(y, p, target_specificity=0.95)
        self.assertEqual(s95["met_target"], 1.0)
        self.assertGreaterEqual(s95["specificity"], 0.95 - 1e-9)
        self.assertTrue(np.isfinite(s95["sensitivity"]))
        self.assertTrue(np.isfinite(s95["threshold"]))

    def test_sens_at_100_spec_uses_fpr_zero_corner(self):
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
        p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=np.float64)
        s100 = sensitivity_at_specificity(y, p, target_specificity=1.0)
        self.assertEqual(s100["met_target"], 1.0)
        self.assertAlmostEqual(s100["specificity"], 1.0, places=5)

    def test_score_predictions_includes_operating_points(self):
        from scripts.evaluate import score_predictions

        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
        p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=np.float64)
        scores = score_predictions(y, p)
        for key in (
            "youden_j",
            "youden_threshold",
            "youden_sensitivity",
            "youden_specificity",
            "sens@95%spec",
            "sens@95%spec_threshold",
            "sens@95%spec_specificity",
            "sens@95%spec_met_target",
        ):
            self.assertIn(key, scores)
            self.assertTrue(np.isfinite(scores[key]), msg=key)

    def test_operating_points_nan_on_single_class(self):
        y = np.zeros(10, dtype=np.float64)
        p = np.linspace(0.1, 0.9, 10)
        op = youden_operating_point(y, p)
        self.assertTrue(np.isnan(op["youden_j"]))
        s95 = sensitivity_at_specificity(y, p)
        self.assertTrue(np.isnan(s95["sensitivity"]))


class TestPublicLoaders(unittest.TestCase):
    def test_parse_tiny_csvs_without_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            odir = os.path.join(tmp, "odir.csv")
            with open(odir, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ID", "Patient Age", "Patient Sex", "Left-Fundus", "Right-Fundus",
                            "N", "D", "G", "C", "A", "H", "M", "O"])
                w.writerow([0, 65, "Male", "0_left.jpg", "0_right.jpg", 0, 1, 0, 0, 0, 0, 0, 0])
                w.writerow([1, 50, "Female", "1_left.jpg", "1_right.jpg", 1, 0, 0, 0, 0, 0, 0, 0])
            recs = parse_odir_csv(odir)
            self.assertEqual(len(recs), 2)
            self.assertEqual(len(ODIR_LABELS), 8)
            self.assertEqual(recs[0].labels[1], 1.0)  # D

            br = os.path.join(tmp, "brset.csv")
            with open(br, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "patient_id", "laterality", "age", "sex",
                            "diabetes", "hypertensive_retinopathy", "quality_focus"])
                w.writerow(["i0", "p0", "left", 60, "F", 1, 0, 1])
                w.writerow(["i1", "p0", "right", 60, "F", 1, 0, 1])
                w.writerow(["i2", "p1", "left", 45, "M", 0, 1, 1])
            brecs = parse_brset_csv(br)
            self.assertGreaterEqual(len(brecs), 2)
            p0 = [r for r in brecs if r.patient_id == "p0"][0]
            self.assertEqual(p0.labels[0], 1.0)
            self.assertEqual(len(BRSET_LABELS), 3)

            rf = os.path.join(tmp, "rfmid.csv")
            with open(rf, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ID", "Disease_Risk", "DR", "ARMD", "OTHER"])
                w.writerow([1, 1, 1, 0, 0])
                w.writerow([2, 0, 0, 0, 0])
            rrecs, names = parse_rfmid_csv(rf)
            self.assertEqual(len(rrecs), 2)
            self.assertIn("DR", names)
            self.assertEqual(rrecs[0].left_path, rrecs[0].right_path)


class TestPublicSmokeTrainEval(unittest.TestCase):
    def test_synthetic_odir_npz_train_eval(self):
        from dataset.odir import make_synthetic_odir_records
        from scripts.evaluate import collect_probs, score_predictions
        from utils.run import single_fastds_run

        with tempfile.TemporaryDirectory() as tmp:
            recs = make_synthetic_odir_records(48, seed=3)
            write_public_npz(tmp, recs, list(ODIR_LABELS), seed=3, dataset_name="odir", force=True)
            for name in ["UKB_RETF", "UKB_swin", "UKB_vim", "UKB_mqd", "UKB_y0"]:
                self.assertTrue(os.path.isfile(os.path.join(tmp, f"{name}.npz")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "SYNTHETIC_FEATURES.txt")))

            ds = UKBDatasetFast(
                tmp,
                meta=["baselineage", "gender", "weight", "ethnicity"],
                disease=list(ODIR_LABELS),
                use_pretrain=["RETF", "SwinB", "VimS"],
                incident_exclude_prior=False,
            )
            pids = dataset_patient_ids(ds)
            self.assertEqual(len(pids), len(ds))
            labels = np.array([float(np.asarray(ds[i][-1]).sum() > 0) for i in range(len(ds))])
            train_idx, val_idx = patient_level_train_val_indices(pids, labels, 0.25, seed=3)
            self.assertTrue(set(pids[train_idx]).isdisjoint(set(pids[val_idx])))

            device = torch.device("cpu")
            model = get_reti_pioneer(True, num_classes=len(ODIR_LABELS), learnable_q=True).to(device)

            def pb(batch, device, non_blocking):
                (l, r), m, (ql, qr), y = batch
                eth = m[:, -1]
                e = to_onehot(eth.long(), 7)
                m = torch.cat([m[:, :-1], e], 1)
                if y.dim() == 1:
                    y = y.view(-1, 1)
                return pb_l_r_m_q_y(((l, r), m, (ql, qr), y), device, non_blocking)

            from torch.utils.data import Subset

            tds = Subset(ds, train_idx)
            vds = Subset(ds, val_idx)
            ckpt_dir = os.path.join(tmp, "run")
            single_fastds_run(
                model,
                lr=1e-3,
                epochs=2,
                epochs_factor=1,
                warmup_lr=1e-5,
                warmup_epochs=2,
                tds=tds,
                vds=vds,
                xdss=[],
                pb=pb,
                bs=8,
                device=device,
                tbdir=ckpt_dir,
                save=True,
                tensorboard=False,
                simple_metrics=True,
            )
            self.assertTrue(any(f.endswith(".pt") for f in os.listdir(os.path.join(ckpt_dir, "ckpt"))))
            probs, y_true = collect_probs(model, vds, device, batch_size=8)
            self.assertEqual(probs.shape[0], len(vds))
            self.assertEqual(probs.shape[1], len(ODIR_LABELS))
            scores = score_predictions(y_true, probs)
            self.assertTrue(np.isfinite(scores["ece"]))
            self.assertTrue(np.isfinite(scores["brier"]))


class TestCrossDatasetMap(unittest.TestCase):
    def test_collect_probs_keeps_multilabel_when_model_k1(self):
        """K=1 model on K>1 label cache must not flatten labels to n*K."""
        from scripts.evaluate import collect_probs

        with tempfile.TemporaryDirectory() as tmp:
            write_public_npz(
                tmp,
                make_synthetic_brset_records(16, seed=3),
                list(BRSET_LABELS),
                seed=3,
                dataset_name="brset",
                force=True,
            )
            ds = UKBDatasetFast(
                tmp,
                meta=["baselineage", "gender", "weight", "ethnicity"],
                y=0,
                disease=[0],
                use_pretrain=["RETF", "SwinB", "VimS"],
                incident_exclude_prior=False,
            )
            ds.set_target(0, list(ds.disease_names), incident_exclude_prior=False)
            self.assertGreater(len(ds.disease_names), 1)
            model = get_reti_pioneer(True, num_classes=1)
            probs, labels = collect_probs(model, ds, torch.device("cpu"), batch_size=8)
            self.assertEqual(probs.shape, (len(ds),))
            self.assertEqual(labels.shape, (len(ds), len(ds.disease_names)))

    def test_odir_brset_overlap(self):
        heads = resolve_cross_heads(
            list(ODIR_LABELS), list(BRSET_LABELS), "odir", "brset"
        )
        tasks = {h.task for h in heads}
        self.assertIn("diabetes_ocular", tasks)
        self.assertIn("hypertension_ocular", tasks)
        d = next(h for h in heads if h.task == "diabetes_ocular")
        self.assertEqual(d.train_name, "D")
        self.assertEqual(d.test_name, "dr_referable")
        logits = np.arange(16, dtype=np.float32).reshape(2, 8)
        labels = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        p, y = slice_mapped(logits, labels, heads)
        self.assertEqual(p.shape, (2, 2))
        self.assertEqual(y.shape, (2, 2))
        h_htn = next(h for h in heads if h.task == "hypertension_ocular")
        h_dm = next(h for h in heads if h.task == "diabetes_ocular")
        # Column order follows `heads` list order.
        i_htn = heads.index(h_htn)
        i_dm = heads.index(h_dm)
        self.assertEqual(p[0, i_dm], logits[0, list(ODIR_LABELS).index("D")])
        self.assertEqual(
            y[0, i_htn],
            labels[0, list(BRSET_LABELS).index("hypertensive_retinopathy")],
        )
        self.assertEqual(y[0, i_dm], labels[0, list(BRSET_LABELS).index("dr_referable")])

    def test_evaluate_cli_cross_dataset(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            odir_dir = os.path.join(tmp, "odir")
            brset_dir = os.path.join(tmp, "brset")
            write_public_npz(
                odir_dir,
                make_synthetic_odir_records(24, seed=1),
                list(ODIR_LABELS),
                seed=1,
                dataset_name="odir",
                force=True,
            )
            write_public_npz(
                brset_dir,
                make_synthetic_brset_records(20, seed=2),
                list(BRSET_LABELS),
                seed=2,
                dataset_name="brset",
                force=True,
            )
            run = os.path.join(tmp, "run")
            os.makedirs(os.path.join(run, "ckpt"), exist_ok=True)
            model = get_reti_pioneer(True, num_classes=len(ODIR_LABELS), learnable_q=False)
            torch.save(model.state_dict(), os.path.join(run, "ckpt", "model.pt"))
            with open(os.path.join(run, "run_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "num_classes": len(ODIR_LABELS),
                        "diseases": list(ODIR_LABELS),
                        "horizon": 0,
                        "learnable_q": False,
                        "enable_q": True,
                        "ensemble": "released_code",
                        "multitask": True,
                        "dataset": "odir",
                        "fast_mode": True,
                    },
                    f,
                )
            out = subprocess.check_output(
                [
                    sys.executable,
                    os.path.join(ROOT, "scripts", "evaluate.py"),
                    "--ckpt",
                    run,
                    "--data-dir",
                    odir_dir,
                    "--dataset",
                    "odir",
                    "--test-data-dir",
                    brset_dir,
                    "--test-dataset",
                    "brset",
                    "--split",
                    "all",
                ],
                cwd=ROOT,
                text=True,
            )
            self.assertIn("Endpoint-aware cross-cohort odir -> brset", out)
            self.assertIn("diabetes_ocular", out)
            self.assertIn("AUROC=", out)


class TestPreparePublicCli(unittest.TestCase):
    def test_prepare_synthetic_demo_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.check_call(
                [
                    sys.executable,
                    os.path.join(ROOT, "scripts", "prepare_public_npz.py"),
                    "--dataset",
                    "odir",
                    "--out",
                    tmp,
                    "--synthetic-demo",
                    "--n-samples",
                    "16",
                    "--force",
                ],
                cwd=ROOT,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmp, "UKB_mqd.npz")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "label_map.json")))


class TestAblationSummaryWriter(unittest.TestCase):
    def test_write_summary_marks_synthetic(self):
        from scripts.run_ablations import _metric_row, write_summary

        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _metric_row(
                    "baseline",
                    "odir_val",
                    {
                        "raw": {
                            "auroc": 0.55,
                            "ap": 0.4,
                            "ece": 0.1,
                            "brier": 0.2,
                            "nb@0.10": 0.01,
                        },
                        "calibrated": {"auroc": 0.56, "ece": 0.05},
                        "temperature": 1.2,
                        "n": 10,
                    },
                ),
                _metric_row("learnq", "odir_val", None, error="train_failed"),
            ]
            csv_path = os.path.join(tmp, "ablation_summary.csv")
            md_path = os.path.join(tmp, "ablation_summary.md")
            write_summary(rows, csv_path, md_path)
            with open(csv_path, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("SYNTHETIC", text)
            self.assertIn("baseline", text)
            self.assertIn("train_failed", text)
            with open(md_path, encoding="utf-8") as f:
                md = f.read()
            self.assertIn("SYNTHETIC", md)
            self.assertIn("baseline", md)
            self.assertIn("| baseline |", md)


class TestEvaluateOutJson(unittest.TestCase):
    def test_out_json_written(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            odir_dir = os.path.join(tmp, "odir")
            write_public_npz(
                odir_dir,
                make_synthetic_odir_records(24, seed=5),
                list(ODIR_LABELS),
                seed=5,
                dataset_name="odir",
                force=True,
            )
            run = os.path.join(tmp, "run")
            os.makedirs(os.path.join(run, "ckpt"), exist_ok=True)
            model = get_reti_pioneer(True, num_classes=1, learnable_q=False)
            torch.save(model.state_dict(), os.path.join(run, "ckpt", "model.pt"))
            with open(os.path.join(run, "run_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "num_classes": 1,
                        "diseases": ["D"],
                        "horizon": 0,
                        "learnable_q": False,
                        "enable_q": True,
                        "ensemble": "released_code",
                        "multitask": False,
                        "dataset": "odir",
                        "fast_mode": True,
                    },
                    f,
                )
            out_json = os.path.join(tmp, "metrics.json")
            subprocess.check_call(
                [
                    sys.executable,
                    os.path.join(ROOT, "scripts", "evaluate.py"),
                    "--ckpt",
                    run,
                    "--data-dir",
                    odir_dir,
                    "--dataset",
                    "odir",
                    "--disease",
                    "D",
                    "--split",
                    "all",
                    "--out-json",
                    out_json,
                ],
                cwd=ROOT,
            )
            with open(out_json, encoding="utf-8") as f:
                payload = json.load(f)
            self.assertIn("raw", payload)
            self.assertIn("auroc", payload["raw"])
            self.assertIn("disclaimer", payload)


class TestDownloadHelper(unittest.TestCase):
    def test_download_helper_exits_zero_without_creds(self):
        out = subprocess.check_output(
            [
                sys.executable,
                os.path.join(ROOT, "scripts", "download_public_data.py"),
                "--dest",
                tempfile.mkdtemp(),
            ],
            cwd=ROOT,
            text=True,
        )
        self.assertIn("BRSET", out)
        self.assertIn("PhysioNet", out)


class TestAblationQuickSmoke(unittest.TestCase):
    def test_run_ablations_quick_one_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            odir = os.path.join(tmp, "odir")
            brset = os.path.join(tmp, "brset")
            ckpt = os.path.join(tmp, "ckpt")
            out = os.path.join(tmp, "results")
            subprocess.check_call(
                [
                    sys.executable,
                    os.path.join(ROOT, "scripts", "run_ablations.py"),
                    "--quick",
                    "--no-cross",
                    "--only",
                    "baseline",
                    "--n-samples",
                    "32",
                    "--odir-dir",
                    odir,
                    "--brset-dir",
                    brset,
                    "--ckpt-dir",
                    ckpt,
                    "--out-dir",
                    out,
                    "--force-cache",
                ],
                cwd=ROOT,
            )
            csv_path = os.path.join(out, "ablation_summary.csv")
            md_path = os.path.join(out, "ablation_summary.md")
            self.assertTrue(os.path.isfile(csv_path))
            self.assertTrue(os.path.isfile(md_path))
            with open(md_path, encoding="utf-8") as f:
                md = f.read()
            self.assertIn("SYNTHETIC", md)
            with open(csv_path, encoding="utf-8") as f:
                body = f.read()
            self.assertIn("baseline", body)
            self.assertIn("ok", body)


class TestFrozenSplitAndDCA(unittest.TestCase):
    def test_net_benefit_can_be_negative(self):
        y = np.array([0, 0, 0, 0, 1], dtype=np.float64)
        # Always predict positive → many FPs → negative NB at high threshold.
        p = np.ones_like(y)
        nb = net_benefit_at_threshold(y, p, 0.5)
        self.assertLess(nb, 0.0)

    def test_train_copies_frozen_test_idx(self):
        from reti_pioneer.split import load_test_idx, save_split

        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "odir")
            ckpt_root = os.path.join(tmp, "ckpt")
            subprocess.check_call(
                [
                    sys.executable,
                    os.path.join(ROOT, "scripts", "prepare_public_npz.py"),
                    "--dataset",
                    "odir",
                    "--synthetic-demo",
                    "--n-samples",
                    "48",
                    "--out",
                    cache,
                    "--force",
                ],
                cwd=ROOT,
            )
            with np.load(os.path.join(cache, "split.npz"), allow_pickle=True) as data:
                train_idx = data["train_idx"].tolist()
                val_idx = data["val_idx"].tolist()
            test_idx = train_idx[:4]
            train_idx = train_idx[4:]
            save_split(
                os.path.join(cache, "split.npz"),
                train_idx,
                val_idx,
                0,
                0.25,
                test_idx=test_idx,
            )
            cfg_path = os.path.join(tmp, "train.yaml")
            with open(os.path.join(ROOT, "configs", "ablation_baseline.yaml"), encoding="utf-8") as f:
                body = f.read()
            lines = []
            for line in body.splitlines():
                if line.startswith("ckpt_dir:"):
                    lines.append(f"ckpt_dir: {ckpt_root.replace(os.sep, '/')}")
                elif line.startswith("data_dir:"):
                    lines.append(f"data_dir: {cache.replace(os.sep, '/')}")
                else:
                    lines.append(line)
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            subprocess.check_call(
                [
                    sys.executable,
                    os.path.join(ROOT, "scripts", "train.py"),
                    "--config",
                    cfg_path,
                    "--dataset",
                    "odir",
                    "--data-dir",
                    cache,
                    "--demo",
                    "--disease",
                    "D",
                    "--horizon",
                    "0",
                ],
                cwd=ROOT,
            )
            found = False
            for root, _dirs, files in os.walk(ckpt_root):
                if "split.npz" in files:
                    t = load_test_idx(os.path.join(root, "split.npz"))
                    if t is not None and len(t) == 4:
                        found = True
                        break
            self.assertTrue(found, "expected run split.npz with frozen test_idx length 4")

    def test_labels_only_preserves_backbone(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "odir")
            subprocess.check_call(
                [
                    sys.executable,
                    os.path.join(ROOT, "scripts", "prepare_public_npz.py"),
                    "--dataset",
                    "odir",
                    "--synthetic-demo",
                    "--n-samples",
                    "24",
                    "--out",
                    out,
                    "--force",
                ],
                cwd=ROOT,
            )
            # Pretend features are real by removing synthetic marker.
            marker = os.path.join(out, "SYNTHETIC_FEATURES.txt")
            self.assertTrue(os.path.isfile(marker))
            os.remove(marker)
            retf = os.path.join(out, "UKB_RETF.npz")
            before = open(retf, "rb").read()
            subprocess.check_call(
                [
                    sys.executable,
                    os.path.join(ROOT, "scripts", "prepare_public_npz.py"),
                    "--dataset",
                    "odir",
                    "--synthetic-demo",
                    "--n-samples",
                    "24",
                    "--out",
                    out,
                    "--labels-only",
                    "--force",
                ],
                cwd=ROOT,
            )
            after = open(retf, "rb").read()
            self.assertEqual(before, after)


class TestExtractPrepareBridge(unittest.TestCase):
    def test_manifest_aligns_and_stub_extract(self):
        from PIL import Image

        from dataset.public_common import (
            PatientRecord,
            assert_feature_row_count,
            load_pairs_manifest,
            write_pairs_manifest,
        )
        from dataset.odir import ODIR_LABELS

        with tempfile.TemporaryDirectory() as tmp:
            img_dir = os.path.join(tmp, "images")
            os.makedirs(img_dir)
            recs = []
            for i in range(4):
                lp = os.path.join(img_dir, f"L{i}.png")
                rp = os.path.join(img_dir, f"R{i}.png")
                Image.new("RGB", (64, 64), (i * 40, 20, 10)).save(lp)
                Image.new("RGB", (64, 64), (10, i * 40, 20)).save(rp)
                y = np.zeros(len(ODIR_LABELS), dtype=np.float32)
                y[i % len(ODIR_LABELS)] = 1.0
                recs.append(
                    PatientRecord(
                        patient_id=f"p{i}",
                        age=50.0 + i,
                        sex=float(i % 2),
                        labels=y,
                        left_path=lp,
                        right_path=rp,
                    )
                )
            cache = os.path.join(tmp, "cache")
            # Labels + synthetic features first, then overwrite with stub extract.
            write_public_npz(
                cache, recs, list(ODIR_LABELS), seed=0, dataset_name="odir", force=True
            )
            man = os.path.join(cache, "pairs.csv")
            kept = write_pairs_manifest(recs, man, root=None, skip_missing=False)
            self.assertEqual(len(kept), 4)
            pairs = load_pairs_manifest(man)
            self.assertEqual(len(pairs), 4)

            dry = subprocess.check_output(
                [
                    sys.executable,
                    os.path.join(ROOT, "scripts", "extract_features.py"),
                    "--cache-dir",
                    cache,
                    "--dry-run",
                ],
                cwd=ROOT,
                text=True,
            )
            self.assertIn("dry-run OK", dry)

            subprocess.check_call(
                [
                    sys.executable,
                    os.path.join(ROOT, "scripts", "extract_features.py"),
                    "--cache-dir",
                    cache,
                    "--all-backbones",
                    "--stub-identity",
                    "--batch-size",
                    "2",
                ],
                cwd=ROOT,
            )
            assert_feature_row_count(cache, 4)
            marker = os.path.join(cache, "SYNTHETIC_FEATURES.txt")
            self.assertTrue(os.path.isfile(marker))
            with open(marker, encoding="utf-8") as f:
                self.assertIn("STUB-IDENTITY", f.read())
            for name, dim in (("UKB_RETF", 1024), ("UKB_swin", 1024), ("UKB_vim", 384)):
                with np.load(os.path.join(cache, f"{name}.npz")) as data:
                    self.assertEqual(data["left"].shape, (4, dim))
                    self.assertEqual(data["right"].shape, (4, dim))

    def test_prepare_cli_write_manifest_require_images(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.makedirs(src)
            # Minimal ODIR-like CSV with absolute image paths.
            csv_path = os.path.join(src, "full_df.csv")
            rows = [["ID", "Patient Age", "Patient Sex", "Left-Fundus", "Right-Fundus"] + list(ODIR_LABELS)]
            for i in range(3):
                lp = os.path.join(src, f"L{i}.png")
                rp = os.path.join(src, f"R{i}.png")
                Image.new("RGB", (32, 32), (i, i, i)).save(lp)
                Image.new("RGB", (32, 32), (i, 0, 0)).save(rp)
                lab = ["0"] * len(ODIR_LABELS)
                lab[0] = "1"
                rows.append([str(i), "55", "Male", lp, rp] + lab)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
            out = os.path.join(tmp, "out")
            subprocess.check_call(
                [
                    sys.executable,
                    os.path.join(ROOT, "scripts", "prepare_public_npz.py"),
                    "--dataset",
                    "odir",
                    "--src",
                    src,
                    "--csv",
                    csv_path,
                    "--out",
                    out,
                    "--labels-only",
                    "--force",
                    "--write-manifest",
                    "--require-images",
                ],
                cwd=ROOT,
            )
            # labels-only on empty cache still needs backbones for ukb_compressed_ready —
            # labels-only without existing cache may not write backbones. Check manifest + mqd.
            self.assertTrue(os.path.isfile(os.path.join(out, "pairs.csv")))
            self.assertTrue(os.path.isfile(os.path.join(out, "UKB_mqd.npz")))
            with np.load(os.path.join(out, "UKB_mqd.npz")) as mqd:
                n_mqd = int(mqd["m"].shape[0])
            with open(os.path.join(out, "pairs.csv"), encoding="utf-8") as fh:
                n_pairs = sum(1 for line in fh if line.strip() and not line.startswith("#"))
            self.assertEqual(n_mqd, n_pairs)
            self.assertEqual(n_mqd, 3)


if __name__ == "__main__":
    unittest.main()
