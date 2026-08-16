"""Fundus preprocessing aligned with Reti-Pioneer / paper Methods."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image
from torchvision import transforms


SourceType = Literal["ukb", "hospital", "generic"]


def fundus_transform(
    image_size: int = 224,
    source: SourceType = "generic",
) -> transforms.Compose:
    """Build train/val transforms.

    UKB: center-crop 1400x1400 then resize (paper Methods).
    Hospital/other: pad to square then resize.
    """
    if source == "ukb":
        return transforms.Compose(
            [
                transforms.Lambda(lambda img: _ukb_crop(img)),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: _pad_square(img)),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _pad_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new(img.mode, (side, side), (0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def _ukb_crop(img: Image.Image, crop: int = 1400) -> Image.Image:
    w, h = img.size
    side = min(w, h, crop)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def synthetic_fundus(size: int = 512, seed: int | None = None) -> np.ndarray:
    """RGB fundus-like array for tests (H, W, 3) uint8."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size // 2, size // 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disc = np.exp(-((r - size * 0.08) ** 2) / (2 * (size * 0.03) ** 2))
    vessel = (rng.random((size, size)) > 0.92).astype(float) * 0.4
    base = 0.35 + 0.25 * np.sin(r / size * 8)
    rgb = np.stack([base + disc * 0.5 + vessel] * 3, axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
