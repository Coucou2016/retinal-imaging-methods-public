#!/usr/bin/env python3
"""Number the references by first appearance and rewrite the reference list.

In-text markers use the form ``{{key}}``.  The script replaces each marker with
``[n]`` where n follows the order of first appearance, then rebuilds the
"## 参考文献" section from ``scripts/citations.py``.

Exit code 1 if a marker has no entry, or if an entry is never cited.
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

import citations

MD = Path(__file__).resolve().parent / "manuscript_zh.md"
MARK = re.compile(r"\{\{([a-z0-9_]+)\}\}")
HEAD = "## 参考文献"


def main() -> int:
    text = MD.read_text(encoding="utf-8")
    head_at = text.index(HEAD)
    body, tail = text[:head_at], text[head_at:]

    order: list[str] = []
    for m in MARK.finditer(body):
        key = m.group(1)
        if key not in citations.REFS:
            print(f"missing reference entry: {key}")
            return 1
        if key not in order:
            order.append(key)

    if not order:
        print("no {{key}} markers found; nothing to do (already numbered?)")
        return 0

    number = {k: i for i, k in enumerate(order, 1)}

    def sub(m: re.Match[str]) -> str:
        return f"[{number[m.group(1)]}]"

    body = MARK.sub(sub, body)

    # merge adjacent brackets produced by "{{a}}{{b}}" style runs
    body = re.sub(r"\]\[", ",", body)

    lines = [HEAD, ""]
    for key in order:
        lines.append(f"{number[key]}. {citations.REFS[key]}")
    new_tail = "\n".join(lines) + "\n"

    MD.write_text(body + new_tail, encoding="utf-8")

    unused = sorted(set(citations.REFS) - set(order))
    print(f"cited {len(order)} references")
    for key in order:
        print(f"  [{number[key]}] {key}")
    if unused:
        print("never cited:", unused)
        return 1
    if tail.strip().split("\n")[0] != HEAD:
        print("reference heading lost")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
