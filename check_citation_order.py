"""Check that reference numbers are first cited in ascending order and all are cited."""

# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent / "manuscript_zh.md"
text = SRC.read_text(encoding="utf-8")
body, refs = text.split("## 参考文献")

first = {}
for m in re.finditer(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]", body):
    raw = m.group(1)
    for part in re.split(r"[,，]", raw):
        part = part.strip()
        if not part.isdigit():
            continue
        n = int(part)
        if n not in first:
            first[n] = m.start()

order = [n for n, _ in sorted(first.items(), key=lambda kv: kv[1])]
print("first-citation order:", order)
print("ascending:", order == sorted(order))

declared = [int(m.group(1)) for m in re.finditer(r"^(\d+)\.\s", refs, flags=re.M)]
print("declared refs:", declared)
missing = [n for n in declared if n not in first]
extra = [n for n in first if n not in declared]
print("declared but never cited:", missing)
print("cited but not declared:", extra)
