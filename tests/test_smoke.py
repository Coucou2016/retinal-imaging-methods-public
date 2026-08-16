"""Smoke tests: demo data generation and one training epoch on CPU."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestRetiPioneerSmoke(unittest.TestCase):
    def test_demo_data_and_imports(self):
        sys.path.insert(0, ROOT)
        from scripts.generate_demo_data import generate
        from reti_pioneer.constants import DISEASE_NAMES

        self.assertEqual(len(DISEASE_NAMES), 6)
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp, n_samples=32, seed=0)
            for name in ["UKB_RETF", "UKB_swin", "UKB_vim", "UKB_mqd", "UKB_y0"]:
                self.assertTrue(os.path.isfile(os.path.join(tmp, f"{name}.npz")))

    def test_model_forward_fast_mode(self):
        """Forward pass on synthetic tensors (no torchvision dataset import)."""
        import torch
        from ignite.utils import to_onehot

        from model.RetiPioneer import get_reti_pioneer
        from utils.functional import pb_l_r_m_q_y

        device = torch.device("cpu")
        model = get_reti_pioneer(True).to(device)
        b = 4
        l = [torch.randn(b, 1024), torch.randn(b, 1024), torch.randn(b, 384)]
        r = [torch.randn(b, 1024), torch.randn(b, 1024), torch.randn(b, 384)]
        m = torch.tensor([[50.0, 1.0, 70.0, 3.0]] * b)
        eth = to_onehot(m[:, -1].long(), 7)
        m = torch.cat([m[:, :-1], eth], 1)
        ql = torch.tensor([[0.7, 0.2, 0.1]] * b)
        qr = ql.clone()
        xmq, _ = pb_l_r_m_q_y(((l, r), m, (ql, qr), torch.zeros(b, 1)), device, False)
        out = model(xmq)
        self.assertEqual(out.shape, (b, 1))

    def test_train_one_disease_cpu(self):
        """Train on pre-extracted features (no torchvision required)."""
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.check_call(
                [
                    sys.executable,
                    os.path.join(ROOT, "scripts", "generate_demo_data.py"),
                    "--out-dir",
                    tmp,
                    "--n-samples",
                    "64",
                ],
                cwd=ROOT,
            )
            import torch
            from datetime import datetime
            from ignite.utils import to_onehot

            from dataset.UKBDataset import UKBDatasetFast
            from model.RetiPioneer import get_reti_pioneer
            from utils.functional import pb_l_r_m_q_y
            from utils.run import single_fastds_run

            def pb(batch, device, non_blocking):
                (l, r), m, (ql, qr), y = batch
                eth = m[:, -1]
                e = to_onehot(eth.long(), 7)
                m = torch.cat([m[:, :-1], e], 1)
                return pb_l_r_m_q_y(((l, r), m, (ql, qr), y), device, non_blocking)

            device = torch.device("cpu")
            ds = UKBDatasetFast(
                tmp,
                meta=["baselineage", "gender", "weight", "ethnicity"],
                disease=[0],
                use_pretrain=["RETF", "SwinB", "VimS"],
            )
            ds.set_target(0, ["t2dm"])
            model = get_reti_pioneer(True).to(device)
            ftime = datetime.now().strftime("%Y%m%d%H%M%S")
            single_fastds_run(
                model,
                lr=1e-3,
                epochs=2,
                epochs_factor=1,
                warmup_lr=1e-5,
                warmup_epochs=2,
                tds=ds,
                vds=None,
                xdss=[],
                pb=pb,
                bs=16,
                device=device,
                tbdir=os.path.join(tmp, "ckpt", ftime, "t2dm", "y0"),
                save=True,
                tensorboard=False,
            )
            ckpt_dir = os.path.join(tmp, "ckpt", ftime, "t2dm", "y0", "ckpt")
            self.assertTrue(any(f.endswith(".pt") for f in os.listdir(ckpt_dir)))

    def test_evaluate_checkpoint(self):
        sys.path.insert(0, ROOT)
        from scripts.evaluate import collect_probs, resolve_ckpt_path
        from scripts.generate_demo_data import generate
        from dataset.UKBDataset import UKBDatasetFast
        from model.RetiPioneer import get_reti_pioneer
        from utils.functional import pb_l_r_m_q_y
        from ignite.utils import to_onehot
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp, n_samples=48, seed=1)
            device = torch.device("cpu")
            ds = UKBDatasetFast(
                tmp,
                meta=["baselineage", "gender", "weight", "ethnicity"],
                disease=["t2dm"],
                use_pretrain=["RETF", "SwinB", "VimS"],
            )
            model = get_reti_pioneer(True).to(device)
            probs, labels = collect_probs(model, ds, device, batch_size=16)
            self.assertEqual(probs.shape[0], len(ds))
            self.assertEqual(labels.shape[0], len(ds))

            run_dir = os.path.join(tmp, "run")
            os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)
            pt = os.path.join(run_dir, "ckpt", "1.pt")
            torch.save(model.state_dict(), pt)
            self.assertEqual(resolve_ckpt_path(run_dir), os.path.abspath(pt))
            self.assertEqual(resolve_ckpt_path(pt), os.path.abspath(pt))


if __name__ == "__main__":
    unittest.main()
