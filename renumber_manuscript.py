#!/usr/bin/env python3
"""Renumber tables and figures so that they appear in ascending order in the text.

The manuscript is hand-written prose with machine-generated tables pasted in, so
an inserted table can leave the numbering out of step with reading order.  This
script rebuilds the numbering from the captions themselves.

1. collect caption anchors (``**表 N**`` / ``**图 N**``) in document order
2. build old -> new maps
3. rewrite every ``表 N`` / ``图 N`` mention, captions included

Usage
-----
    python scripts/renumber_manuscript.py [--check]
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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MD = ROOT / "manuscript_zh.md"

TBL = re.compile(r"\*\*表\s*(\d+)\*\*")
FIG = re.compile(r"\*\*图\s*(\d+)\*\*")


def build_map(text: str, pattern: re.Pattern[str]) -> dict[int, int]:
    order = [int(m.group(1)) for m in pattern.finditer(text)]
    return {old: new for new, old in enumerate(order, 1)}


def rewrite(text: str, word: str, mapping: dict[int, int]) -> str:
    def sub(m: re.Match[str]) -> str:
        old = int(m.group(1))
        new = mapping.get(old, old)
        return f"{word} {new}"

    # a numbered mention is "表 12" / "图 3"; captions are "**表 12**"
    return re.sub(rf"{word}\s+(\d+)", sub, text)


def main() -> int:
    text = MD.read_text(encoding="utf-8")
    tmap = build_map(text, TBL)
    fmap = build_map(text, FIG)

    tbl_changed = {k: v for k, v in tmap.items() if k != v}
    fig_changed = {k: v for k, v in fmap.items() if k != v}

    if "--check" in sys.argv:
        if not tbl_changed and not fig_changed:
            print("numbering already in reading order")
            return 0
        print("table remap", tbl_changed)
        print("figure remap", fig_changed)
        return 1

    # Apply the figure map first: it cannot collide with the table tokens.
    if fig_changed:
        text = rewrite(text, "图", fmap)
    if tbl_changed:
        text = rewrite(text, "表", tmap)

    MD.write_text(text, encoding="utf-8")
    print("table remap", tbl_changed or "unchanged")
    print("figure remap", fig_changed or "unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
