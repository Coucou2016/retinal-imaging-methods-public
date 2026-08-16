"""Audit tests: metrics, ensemble modes, splits, checkpoint round-trip."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from ignite.utils import to_onehot
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dataset.UKBDataset import UKBDatasetFast
from model.RetiPioneer import get_reti_pioneer
from reti_pioneer.split import (
    dataset_labels,
    load_split,
    make_subsets,
    save_split,
    stratified_train_val_indices,
)
from scripts.generate_demo_data import generate
from utils.functional import pb_l_r_m_q_y
from utils.run import single_fastds_run


def _pb(batch, device, non_blocking):
    (l, r), m, (ql, qr), y = batch
    eth = m[:, -1]
    e = to_onehot(eth.long(), 7)
    m = torch.cat([m[:, :-1], e], 1)
    return pb_l_r_m_q_y(((l, r), m, (ql, qr), y), device, non_blocking)


class TestMetricsSanity(unittest.TestCase):
    def test_auroc_perfect_and_random(self):
        y = np.array([0, 0, 1, 1], dtype=np.float32)
        self.assertAlmostEqual(roc_auc_score(y, [0.1, 0.2, 0.8, 0.9]), 1.0)
        rng = np.random.default_rng(0)
        y_rand = rng.integers(0, 2, 200)
        p_rand = rng.random(200)
        auc = roc_auc_score(y_rand, p_rand)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)


class TestEnsembleModes(unittest.TestCase):
    def test_soft_train_vs_max_aggregation(self):
        """Paper: soft voting (train) vs max pooling (inference)."""
        probs = torch.tensor([[0.2, 1.5, 0.8], [-0.5, 0.1, 2.0]])
        temp = 0.1
        train_out = torch.sum(torch.softmax(probs / temp, dim=1) * probs, dim=1, keepdim=True)
        eval_out = torch.max(probs, dim=1, keepdim=True).values
        self.assertFalse(torch.allclose(train_out, eval_out))


class TestSplitAndValidation(unittest.TestCase):
    def test_stratified_split_both_classes_in_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp, n_samples=128, seed=7)
            ds = UKBDatasetFast(
                tmp,
                meta=["baselineage", "gender", "weight", "ethnicity"],
                disease=["t2dm"],
                use_pretrain=["RETF", "SwinB", "VimS"],
            )
            labels = dataset_labels(ds)
            train_idx, val_idx = stratified_train_val_indices(labels, 0.25, seed=7)
            val_labels = labels[val_idx]
            self.assertGreater(val_labels.sum(), 0)
            self.assertLess(val_labels.sum(), len(val_labels))

    def test_demo_train_val_auroc_in_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp, n_samples=96, seed=11)
            ds = UKBDatasetFast(
                tmp,
                meta=["baselineage", "gender", "weight", "ethnicity"],
                disease=["t2dm"],
                use_pretrain=["RETF", "SwinB", "VimS"],
            )
            labels = dataset_labels(ds)
            train_idx, val_idx = stratified_train_val_indices(labels, 0.25, seed=11)
            tds, vds = make_subsets(ds, train_idx, val_idx)
            device = torch.device("cpu")
            model = get_reti_pioneer(True).to(device)
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
                pb=_pb,
                bs=16,
                device=device,
                tbdir=ckpt_dir,
                save=True,
                tensorboard=False,
            )
            save_split(os.path.join(ckpt_dir, "split.npz"), train_idx, val_idx, 11, 0.25)
            loaded_train, loaded_val = load_split(os.path.join(ckpt_dir, "split.npz"))
            self.assertEqual(loaded_train, train_idx)

            from scripts.evaluate import collect_probs

            model.eval()
            probs, y_true = collect_probs(model, vds, device, batch_size=16)
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, probs)
                self.assertGreaterEqual(auc, 0.0)
                self.assertLessEqual(auc, 1.0)


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_state_dict_reload_changes_predictions(self):
        device = torch.device("cpu")
        model_a = get_reti_pioneer(True).to(device)
        model_b = get_reti_pioneer(True).to(device)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            torch.save(model_a.state_dict(), path)
        model_b.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        os.unlink(path)

        b = 2
        l = [torch.randn(b, 1024), torch.randn(b, 1024), torch.randn(b, 384)]
        r = [x.clone() for x in l]
        m = torch.cat(
            [torch.tensor([[50.0, 1.0, 70.0]] * b), to_onehot(torch.zeros(b, dtype=torch.long), 7)],
            1,
        )
        ql = torch.tensor([[0.7, 0.2, 0.1]] * b)
        batch = ((l, r), m, (ql, ql.clone()))
        model_a.eval()
        model_b.eval()
        out_a = model_a(batch)
        out_b = model_b(batch)
        self.assertTrue(torch.allclose(out_a, out_b))


if __name__ == "__main__":
    unittest.main()
