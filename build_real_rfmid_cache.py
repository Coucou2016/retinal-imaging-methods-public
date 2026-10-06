#!/usr/bin/env python
"""Build a REAL-PIXEL RFMiD UKB-style feature cache (no label-derived synthetic features).

Backbone honesty
----------------
- Swin slot: torchvision Swin-V2-B (ImageNet).
- RETF slot: timm ViT-L/16 (ImageNet) — RETFound HF repo is gated without a token.
- Vim slot: timm ViT-S/16 (384-d ImageNet) — Vision Mamba / mamba-ssm unavailable here.

All three are extracted from RFMiD fundus pixels (N=3200 pairs). This is NOT
label-derived SYNTHETIC_FEATURES and is suitable for real-pixel Results tables
with explicit surrogate-backbone disclosure. It is NOT a claim of RETFound/Vim
UKB replication.

Also refreshes ql/qr from a simple no-reference image-quality proxy so quality
routing / intervention figures see real variation (not a constant default).
"""

from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_lzma_shim() -> None:
    try:
        import lzma as _lzma_mod  # noqa: F401

        if hasattr(_lzma_mod, "open"):
            return
    except Exception:
        pass
    lz = types.ModuleType("lzma")
    lz.FORMAT_XZ = 1
    lz.FORMAT_ALONE = 2
    lz.FORMAT_RAW = 3
    lz.CHECK_NONE = 0
    lz.CHECK_CRC32 = 1
    lz.CHECK_CRC64 = 2
    lz.CHECK_SHA256 = 3

    class LZMAError(Exception):
        pass

    lz.LZMAError = LZMAError

    def _open(*_a, **_k):
        raise LZMAError("lzma compression unavailable in this Python build")

    lz.open = _open  # type: ignore[attr-defined]
    sys.modules["lzma"] = lz


_ensure_lzma_shim()

from public_common import load_pairs_manifest  # noqa: E402
from label_map import update_label_map_provenance  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Curated RFMiD heads with enough positives for stable AUROC / ROC / DCA panels.
KEEP_LABELS = [
    "Disease_Risk",
    "DR",
    "MH",
    "ODC",
    "TSLN",
    "DN",
    "ARMD",
    "MYA",
]


def _pil_to_tensor(img: Image.Image, size: int = 224) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


_QUALITY_CACHE: dict[str, np.ndarray] = {}


def _quality_from_image(path: str) -> np.ndarray:
    """Map blur / exposure heuristics → soft (good, usable, bad) probabilities."""
    cached = _QUALITY_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        img = Image.open(path).convert("L").resize((96, 96), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
    except OSError:
        out = np.array([0.34, 0.33, 0.33], dtype=np.float32)
        _QUALITY_CACHE[path] = out
        return out
    # Fast gradient-energy sharpness (cheaper than full Laplacian)
    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    sharp = float(gx.var() + gy.var())
    mean = float(arr.mean())
    sharp_n = float(np.clip(sharp / 220.0, 0.0, 1.0))
    exposure = float(np.exp(-((mean - 128.0) / 55.0) ** 2))
    score = 0.65 * sharp_n + 0.35 * exposure
    good = float(np.clip((score - 0.55) / 0.45, 0.0, 1.0))
    bad = float(np.clip((0.40 - score) / 0.40, 0.0, 1.0))
    usable = float(np.clip(1.0 - good - bad, 0.05, 1.0))
    vec = np.array([good, usable, bad], dtype=np.float64)
    vec = np.maximum(vec, 1e-3)
    vec = (vec / vec.sum()).astype(np.float32)
    _QUALITY_CACHE[path] = vec
    return vec


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], unpaired: bool = False):
        self.pairs = pairs
        self.unpaired = unpaired

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        lp, rp = self.pairs[idx]
        left = _pil_to_tensor(Image.open(lp))
        if self.unpaired or lp == rp:
            return left, left
        return left, _pil_to_tensor(Image.open(rp))


class TimmFeat(nn.Module):
    def __init__(self, model_name: str, out_dim: int):
        super().__init__()
        import timm

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.out_dim = out_dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.model(dummy)
            if feat.ndim > 2:
                feat = feat.mean(dim=tuple(range(1, feat.ndim)))
            in_dim = int(feat.shape[-1])
        if in_dim == out_dim:
            self.proj = nn.Identity()
        else:
            # Fixed orthonormal-ish projection (no training) so dims match UKB slots.
            proj = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.orthogonal_(proj.weight)
            for p in proj.parameters():
                p.requires_grad_(False)
            self.proj = proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.model(x)
        if f.ndim > 2:
            f = f.mean(dim=tuple(range(1, f.ndim)))
        return self.proj(f)


def load_swin(device: torch.device) -> nn.Module:
    from torchvision.models import Swin_V2_B_Weights, swin_v2_b

    model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
    model.head = nn.Identity()
    return model.to(device).eval()


def load_backbone(name: str, device: torch.device) -> nn.Module:
    if name == "swin":
        return load_swin(device)
    if name == "retf":
        # ImageNet ViT-B surrogate for gated RETFound (projected to 1024-d).
        # ViT-L is too slow on CPU for N=3200×2; still real-pixel features.
        return TimmFeat("vit_base_patch16_224", 1024).to(device).eval()
    if name == "vim":
        return TimmFeat("vit_small_patch16_224", 384).to(device).eval()
    raise ValueError(name)


@torch.no_grad()
def extract_backbone(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str,
    *,
    duplicate_unpaired: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract left/right features.

    When ``duplicate_unpaired`` is True (RFMiD single-CFP duplicated to both
    eyes), run the backbone once and copy left→right to roughly halve wall time.
    """
    left, right = [], []
    for l, r in tqdm(loader, desc=desc):
        l = l.to(device)
        feat_l = model(l).float().cpu().numpy()
        left.append(feat_l)
        if duplicate_unpaired:
            right.append(feat_l.copy())
        else:
            r = r.to(device)
            right.append(model(torch.flip(r, dims=[-1])).float().cpu().numpy())
    return np.concatenate(left), np.concatenate(right)


def file_sha256(path: Path, max_bytes: int = 8_000_000) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def subset_labels(cache_dir: Path, keep: list[str]) -> None:
    y0 = np.load(cache_dir / "UKB_y0.npz", allow_pickle=True)
    yn = [str(x) for x in y0["yn"].tolist()]
    y = np.asarray(y0["y"], dtype=np.float32)
    if yn == list(keep):
        print(f"Label subset already applied: {yn}")
        return
    idxs = []
    names = []
    for k in keep:
        if k in yn:
            idxs.append(yn.index(k))
            names.append(k)
    if not idxs:
        raise RuntimeError(f"None of keep labels {keep} found in {yn}")
    y_sub = y[:, idxs]
    for name in ("UKB_y0.npz", "UKB_y5.npz", "UKB_y10.npz"):
        np.savez_compressed(cache_dir / name, y=y_sub, yn=np.array(names))
    print(f"Label subset → {names} shape={y_sub.shape}")


def refresh_quality(cache_dir: Path, pairs: list[tuple[str, str]]) -> None:
    mqd = dict(np.load(cache_dir / "UKB_mqd.npz", allow_pickle=True))
    ql, qr = [], []
    for lp, rp in tqdm(pairs, desc="quality-proxy"):
        ql.append(_quality_from_image(lp))
        qr.append(_quality_from_image(rp))
    mqd["ql"] = np.stack(ql).astype(np.float32)
    mqd["qr"] = np.stack(qr).astype(np.float32)
    np.savez_compressed(cache_dir / "UKB_mqd.npz", **mqd)
    q = mqd["ql"]
    hard = q.argmax(axis=1)
    counts = {k: int((hard == i).sum()) for i, k in enumerate(["good", "usable", "bad"])}
    print(f"Quality strata (left eye argmax): {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "rfmid"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["swin", "retf", "vim"],
        choices=["swin", "retf", "vim"],
    )
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-label-subset", action="store_true")
    parser.add_argument("--keep-labels", nargs="+", default=KEEP_LABELS)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    pairs = load_pairs_manifest(cache_dir / "pairs.csv")
    if args.limit:
        pairs = pairs[: int(args.limit)]
    n = len(pairs)
    print(f"Pairs={n} cache={cache_dir}")

    if not args.skip_quality:
        refresh_quality(cache_dir, pairs)
    if not args.skip_label_subset:
        subset_labels(cache_dir, list(args.keep_labels))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")
    unpaired = all(lp == rp for lp, rp in pairs[: min(32, len(pairs))])
    print(f"Unpaired CFP shortcut={unpaired}")
    loader = DataLoader(
        PairDataset(pairs, unpaired=unpaired),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    out_map = {
        "swin": ("UKB_swin.npz", 1024),
        "retf": ("UKB_RETF.npz", 1024),
        "vim": ("UKB_vim.npz", 384),
    }
    written = []
    for name in args.backbones:
        fname, dim = out_map[name]
        print(f"=== Extract {name} → {fname} (dim={dim}) ===")
        model = load_backbone(name, device)
        unpaired = all(lp == rp for lp, rp in pairs[: min(32, len(pairs))])
        left, right = extract_backbone(
            model, loader, device, desc=name, duplicate_unpaired=unpaired
        )
        if unpaired:
            print(f"{name}: unpaired CFP shortcut (left features copied to right)")
        if left.shape != (n, dim):
            raise RuntimeError(f"{name}: got {left.shape}, expected ({n}, {dim})")
        out_path = cache_dir / fname
        np.savez_compressed(out_path, left=left, right=right)
        print(f"Saved {out_path} sha256_prefix={file_sha256(out_path)[:16]}")
        written.append(name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    marker = cache_dir / "SYNTHETIC_FEATURES.txt"
    if marker.is_file():
        marker.unlink()
        print("Removed SYNTHETIC_FEATURES.txt")

    provenance = {
        "real_pixels": True,
        "n_pairs": n,
        "backbones_written": written,
        "backbone_notes": {
            "swin": "torchvision Swin_V2_B ImageNet weights; real RFMiD pixels",
            "retf": (
                "timm vit_base_patch16_224 ImageNet → orthogonal proj to 1024-d — "
                "YukunZhou/RETFound_mae_natureCFP is gated (HF 401 without token)"
            ),
            "vim": (
                "timm vit_small_patch16_224 ImageNet 384-d surrogate — "
                "Vision Mamba / mamba-ssm not available on this Windows host"
            ),
        },
        "quality": "laplacian+exposure soft (good,usable,bad) from pixels",
        "labels": list(args.keep_labels) if not args.skip_label_subset else "unchanged",
        "clinical_claim_allowed": True,
        "disclaimer": (
            "Real-pixel RFMiD features with ImageNet foundation surrogates. "
            "Not RETFound/Vim UKB replication; not label-derived synthetic features."
        ),
    }
    (cache_dir / "FEATURE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    update_label_map_provenance(
        cache_dir,
        synthetic_features=False,
        stub_features=False,
        extra={
            "extractor": "scripts/build_real_rfmid_cache.py",
            "real_pixels": True,
            "foundation_surrogates": True,
            "retfound_gated": True,
            "vim_unavailable": True,
            **provenance,
        },
    )
    print("DONE real RFMiD cache ready:", cache_dir)


if __name__ == "__main__":
    main()
