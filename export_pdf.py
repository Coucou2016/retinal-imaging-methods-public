#!/usr/bin/env python
"""Export the compiled Word manuscript to PDF through Word COM.

The Word document is the single source of truth for layout: the TOC field,
landscape table sections and embedded figures are already resolved there, so
printing that file keeps the PDF identical to what the reader opens in Word.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs" / "paper" / "manuscript_zh_v3.docx"
PDF = ROOT / "docs" / "paper" / "manuscript_zh_v3.pdf"
MIRROR = (
    ROOT
    / "docs"
    / "researchwrite"
    / "retinal-methods"
    / "exports"
    / "manuscript_zh_v3.pdf"
)

WD_EXPORT_FORMAT_PDF = 17
WD_DO_NOT_SAVE_CHANGES = 0


def main() -> int:
    try:
        import win32com.client as win32
    except Exception as exc:  # pragma: no cover
        print("pywin32 unavailable:", exc)
        return 1
    if not DOCX.is_file():
        print("missing docx:", DOCX)
        return 1

    word = None
    try:
        word = win32.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(str(DOCX), ReadOnly=False)
        doc.Fields.Update()
        for toc in doc.TablesOfContents:
            toc.Update()
        doc.Repaginate()
        doc.Save()
        doc.ExportAsFixedFormat(
            OutputFileName=str(PDF),
            ExportFormat=WD_EXPORT_FORMAT_PDF,
            OpenAfterExport=False,
            OptimizeFor=0,
            CreateBookmarks=1,
            DocStructureTags=True,
            BitmapMissingFonts=True,
            UseISO19005_1=False,
        )
        doc.Close(SaveChanges=WD_DO_NOT_SAVE_CHANGES)
    except Exception as exc:  # pragma: no cover
        print("pdf export failed:", exc)
        return 1
    finally:
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass

    if MIRROR.parent.is_dir():
        MIRROR.write_bytes(PDF.read_bytes())
        print(f"Mirrored {MIRROR}")
    DOCX_MIRROR = MIRROR.with_suffix(".docx")
    if DOCX_MIRROR.parent.is_dir():
        DOCX_MIRROR.write_bytes(DOCX.read_bytes())
        print(f"Mirrored {DOCX_MIRROR} (post TOC refresh)")
    print(f"Wrote {PDF} ({PDF.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
