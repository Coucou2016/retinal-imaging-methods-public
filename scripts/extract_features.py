#!/usr/bin/env python
"""Extract foundation-model features from fundus images into UKB_*.npz caches.

Bridge with ``prepare_public_npz.py``:

1. ``prepare_public_npz.py --labels-only --write-manifest`` writes UKB_mqd/y*
   and ``pairs.csv`` in the **same row order**.
2. This script reads that manifest and writes UKB_RETF / UKB_swin / UKB_vim
   with matching N; refuses to leave a length mismatch.

Requires GPU + pretrained weights for real extraction (RETFound via Hugging Face;
Swin via torchvision; Vision Mamba optional). See docs/DATA.md.

For CI without CUDA/weights, ``--stub-identity`` writes deterministic random
features of the correct shape (NOT for manuscript AUROC).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Windows Miniconda often ships a broken `_lzma` DLL; torchvision imports lzma
# via datasets.utils. Install a no-op shim before any torchvision/timm import.
def _ensure_lzma_shim() -> None:
    try:
        import lzma as _lzma_mod  # noqa: F401

        if hasattr(_lzma_mod, "open"):
            return
    except Exception:
        pass
    import types

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

from dataset.public_common import assert_feature_row_count, load_pairs_manifest

BACKBONE_OUT = {
    "retf": ("UKB_RETF.npz", 1024),
    "swin": ("UKB_swin.npz", 1024),
    "vim": ("UKB_vim.npz", 384),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pil_to_tensor(img: Image.Image, size: int = 224) -> torch.Tensor:
    """Minimal resize+normalize without importing torchvision (Windows lzma workaround)."""
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def build_transform(source: str, stub: bool):
    if stub:
        return _pil_to_tensor
    from dataset.transforms import fundus_transform

    return fundus_transform(source=source)


class ImagePairDataset(Dataset):
    def __init__(self, manifest: list[tuple[str, str]], transform):
        self.manifest = manifest
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        lp, rp = self.manifest[idx]
        l = self.transform(Image.open(lp).convert("RGB"))
        r = self.transform(Image.open(rp).convert("RGB"))
        return l, r


class StubIdentity(nn.Module):
    """Flatten + linear projection to fixed dim for CPU smoke tests only."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(3 * 224 * 224, out_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


@torch.no_grad()
def extract(backbone, loader, device):
    backbone.eval()
    left, right = [], []
    for l, r in tqdm(loader, desc="extract"):
        l, r = l.to(device), r.to(device)
        left.append(backbone(l).cpu().numpy())
        right.append(backbone(r.flip(-1)).cpu().numpy())
    return np.concatenate(left), np.concatenate(right)


def mqd_row_count(cache_dir: Path) -> int | None:
    path = cache_dir / "UKB_mqd.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        return int(data["m"].shape[0])


def load_backbone(name: str, stub: bool, device: torch.device) -> nn.Module:
    dim = BACKBONE_OUT[name][1]
    if stub:
        return StubIdentity(dim).to(device)
    from model.base import get_RETFound, get_SwinB

    if name == "retf":
        return get_RETFound().to(device)
    if name == "swin":
        return get_SwinB().to(device)
    from model.base import get_VimS

    return get_VimS().to(device)


def write_backbone_npz(out_path: Path, left: np.ndarray, right: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, left=left, right=right)
    print(f"Saved {out_path}  shape left={left.shape} right={right.shape}")


def clear_synthetic_marker(cache_dir: Path, stub: bool, *, complete: bool = True) -> None:
    marker = cache_dir / "SYNTHETIC_FEATURES.txt"
    if stub:
        marker.write_text(
            "STUB-IDENTITY features from scripts/extract_features.py --stub-identity.\n"
            "Not foundation-model features. Not for manuscript AUROC.\n"
            "clinical_claim_allowed: false\n",
            encoding="utf-8",
        )
        return
    if not complete:
        marker.write_text(
            "PARTIAL extract: not all foundation backbones replaced.\n"
            "clinical_claim_allowed: false\n",
            encoding="utf-8",
        )
        return
    if marker.is_file():
        marker.unlink()
        print(f"Removed {marker.name} (real backbone features written)")


def update_cache_provenance(
    cache_dir: Path,
    *,
    stub: bool,
    complete: bool = True,
    extra: dict | None = None,
) -> None:
    """Keep label_map.json in sync with feature reality after extract."""
    from reti_pioneer.label_map import update_label_map_provenance

    payload = {
        "extractor": "scripts/extract_features.py",
        "stub_identity": bool(stub),
        "extract_complete": bool(complete) and not stub,
    }
    if extra:
        payload.update(extra)
    update_label_map_provenance(
        cache_dir,
        synthetic_features=False if (complete and not stub) else True,
        stub_features=bool(stub),
        extra=payload,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract fundus features into UKB_* backbone npz (aligned with prepare_public_npz)"
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="CSV: left_path,right_path per line (default: <cache-dir>/pairs.csv)",
    )
    parser.add_argument("--out", default=None, help="Single-backbone output .npz (legacy)")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="UKB cache directory; writes UKB_RETF.npz / UKB_swin.npz / UKB_vim.npz",
    )
    parser.add_argument("--backbone", choices=["retf", "swin", "vim"], default="retf")
    parser.add_argument(
        "--all-backbones",
        action="store_true",
        help="Extract retf+swin+vim into --cache-dir (requires --cache-dir)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--source", default="hospital", choices=["ukb", "hospital", "generic"])
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU when CUDA is unavailable (slow; prefer --stub-identity for CI)",
    )
    parser.add_argument(
        "--stub-identity",
        action="store_true",
        help="CPU stub backbone (correct shapes only). NOT for paper tables.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max pairs to extract (CPU smoke / subset pilot).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest ↔ UKB_mqd row counts; do not load models",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    manifest_path = args.manifest
    if manifest_path is None:
        if cache_dir is None:
            raise SystemExit("Pass --manifest and/or --cache-dir")
        manifest_path = str(cache_dir / "pairs.csv")
    pairs = load_pairs_manifest(manifest_path)
    if not pairs:
        raise SystemExit(f"Empty manifest: {manifest_path}")
    if args.limit is not None:
        pairs = pairs[: max(1, int(args.limit))]
        print(f"Limiting extract to first {len(pairs)} pairs (--limit)")

    n = len(pairs)
    if cache_dir is not None and args.limit is None:
        n_mqd = mqd_row_count(cache_dir)
        if n_mqd is not None and n_mqd != n:
            raise SystemExit(
                f"Manifest has {n} rows but {cache_dir}/UKB_mqd.npz has {n_mqd}. "
                "Re-run prepare_public_npz with --write-manifest so labels and pairs align."
            )

    if args.dry_run:
        print(f"dry-run OK: manifest={manifest_path} n={n} cache_dir={cache_dir}")
        if cache_dir is not None:
            try:
                assert_feature_row_count(cache_dir, n)
                print("existing backbone npz row counts match")
            except ValueError as exc:
                print(f"note: {exc}")
        return

    if args.all_backbones and cache_dir is None:
        raise SystemExit("--all-backbones requires --cache-dir")

    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.allow_cpu and not args.stub_identity:
        raise RuntimeError(
            "Feature extraction expects CUDA. Pass --allow-cpu or --stub-identity for local smoke, "
            "or use pre-built UKB_*.npz."
        )
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.stub_identity:
        print("WARNING: --stub-identity produces non-foundation features (CI only)")

    transform = build_transform(args.source, stub=args.stub_identity)
    loader = DataLoader(
        ImagePairDataset(pairs, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    names = list(BACKBONE_OUT) if args.all_backbones else [args.backbone]
    for name in names:
        fname, dim = BACKBONE_OUT[name]
        model = load_backbone(name, stub=args.stub_identity, device=device)
        left, right = extract(model, loader, device)
        if left.shape[0] != n or left.shape[-1] != dim:
            raise RuntimeError(f"{name}: unexpected left shape {left.shape}, expected ({n}, {dim})")
        if cache_dir is not None:
            out_path = cache_dir / fname
        elif args.out:
            out_path = Path(args.out)
        else:
            raise SystemExit("Pass --out or --cache-dir")
        write_backbone_npz(out_path, left, right)

    if cache_dir is not None:
        complete = bool(args.all_backbones) or args.limit is None
        # Single-backbone or --limit extracts are not a full clinical feature cache.
        if args.limit is not None or not args.all_backbones:
            complete = False
        clear_synthetic_marker(cache_dir, stub=args.stub_identity, complete=complete)
        update_cache_provenance(
            cache_dir,
            stub=args.stub_identity,
            complete=complete,
            extra={"backbones_written": names, "n": n},
        )
        if args.limit is None and complete:
            assert_feature_row_count(cache_dir, n)
        print(f"Aligned cache OK under {cache_dir} (N={n}, complete={complete})")


if __name__ == "__main__":
    main()
