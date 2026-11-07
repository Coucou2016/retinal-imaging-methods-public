#!/usr/bin/env python
"""Convert the manuscript's display equations to native Word math (OMML).

The earlier exporter embedded each equation as a 600 dpi PNG. That prints
correctly but leaves the reader with a picture: it cannot be selected, searched,
or edited, and it does not rescale with the paragraph. Word stores math as
OMML, so the LaTeX here is routed through MathML and inserted as a real math
object. The PNG path stays as a fallback for hosts without the two converters.

Exports
-------
DISPLAY_LATEX   the nine numbered display equations, keyed by number
to_omml(tex)    OMML element for one equation, or None when conversion fails
"""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


from xml.etree import ElementTree as ET

try:  # lxml is what python-docx expects to append
    from lxml import etree as LET
except Exception:  # pragma: no cover
    LET = None

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

DISPLAY_LATEX = {
    1: r"u_i=\mathrm{softplus}(\delta_i)+\varepsilon,\quad i=1,2,3",
    2: r"c_i=\sum_{j=1}^{i}u_j",
    3: r"w_i=\frac{c_i}{c_3}",
    4: (
        r"\mathcal{L}=-\frac{1}{\sum_{k=1}^{K}m_k}\sum_{k=1}^{K}m_k"
        r"\left[y_k\log\sigma(z_k)+(1-y_k)\log(1-\sigma(z_k))\right]"
    ),
    5: (
        r"z=\sum_{h=1}^{3}\alpha_h(q)\,z^{(h)},\qquad "
        r"\alpha(q)=\mathrm{softmax}\left(W_2\,\mathrm{SELU}(W_1q+b_1)+b_2\right)"
    ),
    6: (
        r"\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}"
        r"\left|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\right|"
    ),
    7: r"z'=z+\log\frac{\pi_{\mathrm{val}}}{1-\pi_{\mathrm{val}}}",
    8: r"p=\sigma\left(a\,z+b\right)",
    9: (
        r"\mathrm{NB}=\frac{\mathrm{TP}}{n}-\frac{\mathrm{FP}}{n}"
        r"\cdot\frac{p_t}{1-p_t}"
    ),
}


def to_omml(tex: str):
    """OMML element for ``tex``, or ``None`` if the converters are absent.

    ``mathml2omml`` emits fragments that use the ``m:`` prefix without declaring
    it, so the string is wrapped once in a root element that carries the
    namespace before it is parsed.
    """
    try:
        import latex2mathml.converter as l2m
        import mathml2omml
    except Exception:
        return None
    try:
        mathml = l2m.convert(tex)
        omml = mathml2omml.convert(mathml)
        wrapped = f'<root xmlns:m="{M_NS}">{omml}</root>'
        parser = LET.XMLParser() if LET is not None else None
        root = LET.fromstring(wrapped, parser) if LET is not None else ET.fromstring(wrapped)
        for child in root:
            return child
    except Exception:
        return None
    return None
