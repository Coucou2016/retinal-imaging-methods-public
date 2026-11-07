"""Audit figure/table caption pairing and page layout in the exported docx."""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCX = ROOT / "manuscript_zh.docx"


def main() -> int:
    import win32com.client as win32

    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = word.Documents.Open(str(DOCX), ReadOnly=True)
    try:
        wd_page = 3  # wdActiveEndPageNumber
        print(f"pages: {doc.ComputeStatistics(2)}")
        bad = 0
        for idx in range(1, doc.Tables.Count + 1):
            tbl = doc.Tables(idx)
            tpage = tbl.Range.Information(wd_page)
            prev = tbl.Range.Paragraphs(1).Previous()
            text = prev.Range.Text.strip()
            # walk back to the caption paragraph
            tries = 0
            while tries < 3 and not text.startswith("表"):
                prev = prev.Previous()
                text = prev.Range.Text.strip()
                tries += 1
            cpage = prev.Range.Information(wd_page)
            flag = "" if cpage == tpage else "  <== SPLIT"
            if cpage != tpage:
                bad += 1
            print(f"tab {idx:2d} caption p{cpage:3d} table p{tpage:3d} rows {tbl.Rows.Count:2d} {flag}")

        inline = doc.InlineShapes
        print(f"inline shapes: {inline.Count}")
        for idx in range(1, inline.Count + 1):
            shp = inline(idx)
            ipage = shp.Range.Information(wd_page)
            nxt = shp.Range.Paragraphs(1).Next().Range.Text.strip()
            tries = 0
            while tries < 3 and not nxt.startswith("图"):
                nxt = shp.Range.Paragraphs(1).Next().Range.Text.strip()
                tries += 1
            cpage = shp.Range.Paragraphs(1).Next().Range.Information(wd_page)
            flag = "" if cpage == ipage else "  <== SPLIT"
            if cpage != ipage:
                bad += 1
            print(f"img {idx:2d} p{ipage:3d} caption p{cpage:3d} {nxt[:26]!r}{flag}")
        print(f"\nsplit pairs: {bad}")
        return 1 if bad else 0
    finally:
        doc.Close(False)
        word.Quit()


if __name__ == "__main__":
    sys.exit(main())
