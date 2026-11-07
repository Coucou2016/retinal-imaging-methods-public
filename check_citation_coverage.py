"""Check that every figure and table caption is cited at least once in prose."""

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
SRC = ROOT / "manuscript_zh.md"
lines = SRC.read_text(encoding="utf-8").split("\n")

fig_caps, tab_caps = [], []
cites = {"图": [], "表": []}
in_refs = False
for idx, line in enumerate(lines, 1):
    s = line.strip()
    if s.startswith("## 参考文献"):
        in_refs = True
    if in_refs or not s:
        continue
    if s.startswith("**图") or s.startswith("**表"):
        kind = s[2]
        m = re.match(r"\*\*([图表])\s*(\d+)\*\*", s)
        if m:
            (fig_caps if kind == "图" else tab_caps).append((int(m.group(2)), idx, s[:40]))
        continue
    if s.startswith("!["):
        continue
    for m in re.finditer(r"图\s*(\d+)", s):
        cites["图"].append((int(m.group(1)), idx))
    for m in re.finditer(r"表\s*(\d+)", s):
        cites["表"].append((int(m.group(1)), idx))

for kind, caps in (("图", fig_caps), ("表", tab_caps)):
    cited = {n for n, _ in cites[kind]}
    missing = [c for c in caps if c[0] not in cited]
    print("%s captions %d, cited %d" % (kind, len(caps), len(caps) - len(missing)))
    for num, ln, head in missing:
        print("  never cited: %s %d (line %d) %s" % (kind, num, ln, head))
