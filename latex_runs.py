#!/usr/bin/env python3
"""LaTeX -> styled run segments for Word, covering the constructs used here.

Styles: 'i' italic math variable, 'u' upright, 'sub' subscript, 'sup' superscript.
"""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import re

SYMBOLS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "Delta": "Δ",
    "varepsilon": "ε",
    "epsilon": "ε",
    "sigma": "σ",
    "Sigma": "Σ",
    "sum": "Σ",
    "in": "∈",
    "times": "×",
    "approx": "≈",
    "top": "⊤",
    "cdot": "·",
    "pm": "±",
    "le": "≤",
    "ge": "≥",
    "neq": "≠",
    "mid": "|",
    "qquad": "\u2003\u2003",
    "quad": "\u2003",
    ",": "\u2009",
    ";": "\u2009",
    "{": "{",
    "}": "}",
    "left": "",
    "right": "",
    "lvert": "|",
    "rvert": "|",
    "log": "log",
    "exp": "exp",
    "max": "max",
    "min": "min",
}

UPRIGHT = {"log", "exp", "max", "min", "softplus", "softmax", "sigmoid", "SELU"}

TOK = re.compile(r"\\[A-Za-z]+|\\.|[{}_^]|[^\s{}_^\\]")


def tokenize(latex: str):
    return [m.group(0) for m in TOK.finditer(latex)]


class P:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def group(self):
        """Read { ... } with nesting, or a single token."""
        tok = self.peek()
        if tok == "{":
            self.take()
            depth = 1
            out = []
            while self.i < len(self.t):
                cur = self.take()
                if cur == "{":
                    depth += 1
                elif cur == "}":
                    depth -= 1
                    if depth == 0:
                        return out
                out.append(cur)
            return out
        if tok is None:
            return []
        return [self.take()]

    def run(self, toks=None):
        if toks is not None:
            saved = self.t, self.i
            self.t, self.i = toks, 0
            out = self.run()
            self.t, self.i = saved
            return out
        out = []
        while self.i < len(self.t):
            tok = self.take()
            if tok == "}":
                break
            if tok in ("{",):
                out.extend(self.run(self.group()))
                continue
            if tok in ("_", "^"):
                inner = self.group()
                style = "sub" if tok == "_" else "sup"
                sub = self.run(inner)
                for text, st in sub:
                    if st in ("sub", "sup"):
                        out.append((text, st))
                    else:
                        out.append((text, style))
                continue
            if tok.startswith("\\"):
                name = tok[1:]
                if name in ("mathrm", "text", "operatorname"):
                    inner = self.group()
                    txt = "".join(self.plain(inner))
                    out.append((txt, "u"))
                    continue
                if name == "mathbb":
                    inner = self.group()
                    txt = "".join(self.plain(inner))
                    out.append(("ℝ" if txt == "R" else txt, "u"))
                    continue
                if name == "mathcal":
                    inner = self.group()
                    txt = "".join(self.plain(inner))
                    out.append((txt, "i"))
                    continue
                if name == "frac":
                    num = self.run(self.group())
                    den = self.run(self.group())
                    for t, s in num:
                        out.append((t, "i" if s == "u" else s))
                    out.append(("/", "u"))
                    for t, s in den:
                        out.append((t, "i" if s == "u" else s))
                    continue
                if name in SYMBOLS:
                    out.append((SYMBOLS[name], "u"))
                    continue
                out.append((name, "u"))
                continue
            if tok.isalpha() and tok.isascii():
                out.append((tok, "i" if tok not in UPRIGHT else "u"))
                continue
            out.append((tok, "u"))
        return out

    def plain(self, toks):
        return [t for t, _ in self.run(toks)]


def segments(latex: str):
    return P(tokenize(latex)).run()


def plain_text(latex: str) -> str:
    return "".join(t for t, _ in segments(latex))


if __name__ == "__main__":
    tests = [
        r"q=(q_{\mathrm{good}},q_{\mathrm{usable}},q_{\mathrm{bad}})",
        r"a\in\mathbb{R}^{3}",
        r"1\times10^{-4}",
        r"p=\sigma(z/T)",
        r"m_k\in\{0,1\}",
        r"0<w_1<w_2<w_3=1",
        r"w\approx(10^{-4},0.5,1)",
        r"z^{(h)}",
        r"s=q^{\top}w",
        r"\varepsilon=10^{-6}",
        r"w_i=\frac{c_i}{c_3}",
        r"c_i=\sum_{j=1}^{i}u_j",
        r"\mathcal{L}=-\frac{1}{\sum_{k=1}^{K}m_k}",
    ]
    for t in tests:
        print(f"{t:55s} -> {plain_text(t)}")
