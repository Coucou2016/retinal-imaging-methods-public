#!/usr/bin/env python3
"""List prose sentences over a character budget, with line numbers.

Usage: python scripts/list_long_sentences.py [budget]
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

MD = Path(__file__).resolve().parent / "manuscript_zh.md"


def main() -> int:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    text = MD.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    hits = []
    for i, ln in enumerate(text.splitlines(), 1):
        s = ln.strip()
        if not s or s.startswith("|") or s.startswith("![") or s.startswith("**图"):
            continue
        for sent in re.split(r"[。！？]", s):
            if len(sent) > budget:
                hits.append((i, len(sent), sent))
    for i, n, sent in hits:
        print(f"{i}\t{n}\t{sent}")
    print(f"\n{len(hits)} sentences over {budget}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
