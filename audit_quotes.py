#!/usr/bin/env python3
"""Flag paired quotation marks in prose.

The house style for this manuscript avoids quotation marks around ordinary
phrases, because they read as editorial emphasis rather than as a citation.
Quotes that mark a literal symbol or a verbatim dataset name are fine, so this
script only reports the occurrences and leaves the decision to the author.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MD = ROOT / "manuscript_zh.md"

QUOTES = re.compile(r"[\u201c\u201d\u2018\u2019\"]")


def main() -> int:
    hits = 0
    for lineno, line in enumerate(MD.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith(("|", "#", "```")):
            continue
        for m in QUOTES.finditer(line):
            hits += 1
            lo, hi = max(0, m.start() - 28), min(len(line), m.end() + 28)
            print(f"{lineno}\t{m.group(0)}\t{line[lo:hi]}")
    print(f"\n{hits} quotation mark(s) in prose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
