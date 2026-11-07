
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------
import pymupdf

d = pymupdf.open("_preview_v3.pdf")
print("pages", d.page_count)

def overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

problems = 0
for i, page in enumerate(d):
    words = page.get_text("words")
    # group by line y
    for x in range(len(words)):
        for y in range(x + 1, min(x + 12, len(words))):
            a, b = words[x][:4], words[y][:4]
            if overlap(a, b):
                ov = min(a[2], b[2]) - max(a[0], b[0])
                if ov > 2:
                    problems += 1
                    if problems < 15:
                        print("page", i + 1, "overlap", round(ov, 1), words[x][4], "|", words[y][4])
print("word overlaps:", problems)

# check margins
for i, page in enumerate(d):
    w, h = page.rect.width, page.rect.height
    for b in page.get_text("blocks"):
        if b[2] > w - 15:
            print("right overflow page", i + 1, round(b[2]), "of", round(w), b[4][:40].replace("\n", " "))
