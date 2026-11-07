#!/usr/bin/env python3
"""Render display equations to transparent PNGs at print resolution."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper_assets" / "equations"

EQUATIONS = {
    1: r"$u_i=\mathrm{softplus}(\delta_i)+\varepsilon,\quad i=1,2,3$",
    2: r"$c_i=\sum_{j=1}^{i}u_j$",
    3: r"$w_i=\dfrac{c_i}{c_3}$",
    4: (
        r"$\mathcal{L}=-\dfrac{1}{\sum_{k=1}^{K}m_k}"
        r"\sum_{k=1}^{K}m_k\left[y_k\log\sigma(z_k)+(1-y_k)\log(1-\sigma(z_k))\right]$"
    ),
    5: (
        r"$z=\sum_{h=1}^{3}\alpha_h(q)\,z^{(h)},\quad "
        r"\alpha(q)=\mathrm{softmax}\left(W_2\,\mathrm{SELU}(W_1q+b_1)+b_2\right)$"
    ),
    6: (
        r"$\mathrm{ECE}=\sum_{m=1}^{M}\dfrac{|B_m|}{n}"
        r"\left|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\right|$"
    ),
    7: r"$z'=z+\log\dfrac{\pi_{\mathrm{val}}}{1-\pi_{\mathrm{val}}}$",
    8: r"$p=\sigma\left(a\,z+b\right)$",
    9: r"$\mathrm{NB}=\dfrac{\mathrm{TP}}{n}-\dfrac{\mathrm{FP}}{n}\cdot\dfrac{p_t}{1-p_t}$",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["mathtext.fontset"] = "stix"
    meta = {}
    for num, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=13, color="black")
        path = OUT / f"eq{num}.png"
        fig.savefig(
            path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.03,
            transparent=True,
            facecolor="white",
        )
        plt.close(fig)
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
        # width in inches when displayed at 600 dpi
        meta[num] = {"px": [w, h], "inches": [w / 600, h / 600]}
        print(f"eq{num}: {w}x{h}px -> {w/600:.2f}x{h/600:.2f} in")

    (OUT / "equations.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
