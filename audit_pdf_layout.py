
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------
import pymupdf

d = pymupdf.open("_preview_v3.pdf")
issues = []
for i, page in enumerate(d):
    r = page.rect
    w, h = r.width, r.height
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
        if x1 > w - 20 or x0 < 20 or y1 > h - 10 or y0 < 10:
            issues.append((i + 1, round(x0), round(y0), round(x1), round(y1), txt[:50].replace("\n", " ")))
    for img in page.get_images(full=True):
        pass
print("total pages", d.page_count)
print("overflow blocks", len(issues))
for it in issues[:20]:
    print(it)

# check page fill ratios
print("\n-- fill --")
for i, page in enumerate(d):
    blocks = page.get_text("blocks")
    if not blocks:
        print(i + 1, "EMPTY PAGE")
        continue
    ymax = max(b[3] for b in blocks)
    if ymax < page.rect.height * 0.55:
        print(i + 1, "short page, content ends y=", round(ymax), "of", round(page.rect.height))
