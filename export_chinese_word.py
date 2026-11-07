#!/usr/bin/env python3
"""Build the Chinese Word manuscript from docs/paper/manuscript_zh_v2.md.

Features
--------
* Chinese journal typography (宋体 body, 黑体 headings, A4, 1.5 spacing)
* Native Word equations converted from LaTeX through MathML -> OMML
* Three-line (booktabs style) tables, landscape sections for wide tables
* Embedded figures with captions
* Live TOC field plus page-number footer; refreshed by Word COM afterwards
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

try:
    import latex_runs
except Exception:  # pragma: no cover
    latex_runs = None

try:
    from omml_math import to_omml
except Exception:  # pragma: no cover
    to_omml = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "paper" / "manuscript_zh_v2.md"
OUT = ROOT / "docs" / "paper" / "manuscript_zh_v3.docx"
MIRROR = ROOT / "docs" / "researchwrite" / "retinal-methods" / "exports" / "manuscript_zh_v3.docx"

CN_BODY = "宋体"
CN_HEAD = "黑体"
EN_FONT = "Times New Roman"
MONO = "Consolas"

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


# ---------------------------------------------------------------- font helpers
def set_font(run, size=12, bold=False, italic=False, cn=CN_BODY, en=EN_FONT):
    run.font.name = en
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def style_font(style, size, bold=False, cn=CN_BODY, en=EN_FONT, color=(0, 0, 0)):
    style.font.name = en
    style.font.size = Pt(size)
    style.font.bold = bold
    if color is not None:
        style.font.color.rgb = RGBColor(*color)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)


# --------------------------------------------------------------------- styles
def build_styles(doc):
    normal = doc.styles["Normal"]
    style_font(normal, 12)
    pf = normal.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(4)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for level, size, before in ((1, 14, 18), (2, 12, 12), (3, 12, 10)):
        st = doc.styles[f"Heading {level}"]
        style_font(st, size, bold=True, cn=CN_HEAD)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing = 1.4
        st.paragraph_format.keep_with_next = True
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        st.paragraph_format.first_line_indent = Cm(0)

    extra = {
        "PaperTitle": dict(size=20, bold=True, cn=CN_HEAD),
        "Subtitle": dict(size=11, bold=False, cn=CN_BODY),
        "TocTitle": dict(size=16, bold=True, cn=CN_HEAD),
        "Caption": dict(size=10, bold=False, cn=CN_BODY),
        "TableText": dict(size=9, bold=False, cn=CN_BODY),
        "CodeBlock": dict(size=9, bold=False, cn=CN_BODY, en=MONO),
        "Reference": dict(size=10, bold=False, cn=CN_BODY),
        "AbstractMeta": dict(size=11, bold=False, cn=CN_BODY),
    }
    existing = {s.name for s in doc.styles}
    for name, cfg in extra.items():
        if name in existing:
            st = doc.styles[name]
        else:
            st = doc.styles.add_style(name, 1)  # WD_STYLE_TYPE.PARAGRAPH
        style_font(st, cfg["size"], bold=cfg["bold"], cn=cfg["cn"], en=cfg.get("en", EN_FONT))
        st.base_style = doc.styles["Normal"]
        st.paragraph_format.line_spacing = 1.4

    cap = doc.styles["Caption"]
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(12)
    cap.paragraph_format.first_line_indent = Cm(0)
    cap.paragraph_format.line_spacing = 1.2

    tt = doc.styles["TableText"]
    tt.paragraph_format.space_before = Pt(2)
    tt.paragraph_format.space_after = Pt(2)
    tt.paragraph_format.first_line_indent = Cm(0)
    tt.paragraph_format.line_spacing = 1.0

    cb = doc.styles["CodeBlock"]
    cb.paragraph_format.left_indent = Cm(0.6)
    cb.paragraph_format.first_line_indent = Cm(0)
    cb.paragraph_format.space_after = Pt(2)
    cb.paragraph_format.line_spacing = 1.15

    ref = doc.styles["Reference"]
    ref.paragraph_format.left_indent = Cm(0.85)
    ref.paragraph_format.first_line_indent = Cm(-0.85)
    ref.paragraph_format.space_after = Pt(3)
    ref.paragraph_format.line_spacing = 1.25
    ref.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    toc = doc.styles["TocTitle"]
    toc.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    toc.paragraph_format.space_after = Pt(14)
    toc.paragraph_format.first_line_indent = Cm(0)


def page_setup(section):
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.6)
    section.right_margin = Cm(2.6)
    section.header_distance = Cm(1.4)
    section.footer_distance = Cm(1.4)


def landscape_section(doc):
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width = Cm(29.7)
    sec.page_height = Cm(21.0)
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)
    return sec


def portrait_section(doc):
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    sec.orientation = WD_ORIENT.PORTRAIT
    page_setup(sec)
    return sec


# --------------------------------------------------------------- field helpers
def add_field(paragraph, instr, placeholder="", size=10):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin.set(qn("w:dirty"), "true")
    run._r.append(begin)

    r2 = paragraph.add_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    r2._r.append(it)

    r3 = paragraph.add_run()
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    r3._r.append(sep)

    r4 = paragraph.add_run(placeholder)
    set_font(r4, size=size)

    r5 = paragraph.add_run()
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    r5._r.append(end)


def add_toc(doc):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_after = Pt(6)
    add_field(p, r'TOC \o "1-2" \h \z \u', "右键选择“更新域”生成目录。", size=11)
    for run in p.runs:
        set_font(run, size=11)


def add_footer_page_number(section):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    add_field(p, "PAGE", "1", size=10)


def enable_update_fields(doc):
    settings = doc.settings.element
    tag = settings.find(qn("w:updateFields"))
    if tag is None:
        tag = OxmlElement("w:updateFields")
        settings.append(tag)
    tag.set(qn("w:val"), "true")


# ------------------------------------------------------------- math rendering
INLINE_RE = re.compile(r"(\*\*.+?\*\*|\*[^*\n]+?\*|\\\(.+?\\\)|\[\d+(?:,\s*\d+)*\])")
CITE_RE = re.compile(r"^\[\d+(?:,\s*\d+)*\]$")


def add_math_runs(paragraph, latex, size=12):
    """Render inline math as styled text runs (italic vars, sub/superscripts)."""
    if latex_runs is None:
        run = paragraph.add_run(latex)
        set_font(run, size=size, italic=True)
        return
    for text, style in latex_runs.segments(latex):
        if not text:
            continue
        run = paragraph.add_run(text)
        set_font(run, size=size, italic=(style in ("i", "sub", "sup")), en=EN_FONT)
        if style == "sub":
            run.font.subscript = True
        elif style == "sup":
            run.font.superscript = True


def add_rich(paragraph, text, size=12, cn=CN_BODY, base_bold=False):
    for part in INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            run = paragraph.add_run(part[2:-2].replace("\\*", "*"))
            set_font(run, size=size, bold=True, cn=cn)
        elif CITE_RE.match(part):
            run = paragraph.add_run(part)
            set_font(run, size=size, bold=False, cn=cn)
            run.font.superscript = True
        elif part.startswith("\\(") and part.endswith("\\)"):
            add_math_runs(paragraph, part[2:-2], size=size)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run = paragraph.add_run(part[1:-1])
            set_font(run, size=size, italic=True, cn=cn)
        else:
            run = paragraph.add_run(part)
            set_font(run, size=size, bold=base_bold, cn=cn)


# --------------------------------------------------------------- table builder
def set_cell_margins(table, top=40, bottom=40, left=60, right=60):
    tblPr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for edge, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)


def three_line_borders(table):
    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        if edge in ("top", "bottom"):
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "12")
        else:
            el.set(qn("w:val"), "none")
            el.set(qn("w:sz"), "0")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tblPr.append(borders)


def header_bottom_rule(table):
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tcBorders")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "8")
        bottom.set(qn("w:space"), "0")
        bottom.set(qn("w:color"), "000000")
        borders.append(bottom)
        tcPr.append(borders)


def repeat_header(table):
    trPr = table.rows[0]._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def no_row_split(table):
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        el = OxmlElement("w:cantSplit")
        el.set(qn("w:val"), "true")
        trPr.append(el)


def fixed_layout(table, col_cm):
    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn("w:tblLayout")):
        tblPr.remove(old)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)

    total_cm = sum(col_cm)
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:w"), str(int(total_cm * 567)))
    tblW.set(qn("w:type"), "dxa")
    tblPr.append(tblW)

    tbl = table._tbl
    for old in tbl.findall(qn("w:tblGrid")):
        tbl.remove(old)
    grid = OxmlElement("w:tblGrid")
    for w in col_cm:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(int(w * 567)))
        grid.append(gc)
    tblPr.addnext(grid)


def add_table(doc, headers, rows, font_size=9, total_cm=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    three_line_borders(table)
    set_cell_margins(table)

    # width from estimated text extent so headers do not wrap needlessly
    if total_cm is None:
        total_cm = 15.6

    def text_cm(s, size_pt, bold=False):
        """Rough rendered width in cm: CJK glyphs are square, Latin about half."""
        k = 1.06 if bold else 1.0
        latin = 0.0
        for ch in s:
            if ord(ch) > 0x2E7F:
                latin += 1.0
            elif ch in "ilj.,:;'|! ":
                latin += 0.30
            elif ch in "mwMW%-−":
                latin += 0.72
            else:
                latin += 0.53
        return latin * size_pt * k / 72 * 2.54

    pad_cm = 0.34
    need = []
    for ci in range(len(headers)):
        texts = [headers[ci]] + [r[ci] if ci < len(r) else "" for r in rows]
        widest = max(text_cm(t, font_size, bold=(k == 0)) for k, t in enumerate(texts))
        need.append(max(widest + pad_cm, 0.95))

    if sum(need) <= total_cm:
        extra = total_cm - sum(need)
        col_cm = [w + extra * w / sum(need) for w in need]
    else:
        col_cm = [total_cm * w / sum(need) for w in need]

    for i, head in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.style = doc.styles["TableText"]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(head)
        set_font(run, size=font_size, bold=True, cn=CN_HEAD)

    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            p.style = doc.styles["TableText"]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if ci else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(val)
            set_font(run, size=font_size)

    for row in table.rows:
        for ci, cell in enumerate(row.cells):
            cell.width = Cm(col_cm[ci])

    fixed_layout(table, col_cm)
    header_bottom_rule(table)
    repeat_header(table)
    no_row_split(table)
    # keep the whole table on one page: Word otherwise leaves an orphaned
    # header row at the foot of a page and pushes the body over the break
    for row in table.rows[:-1]:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True
    return table


def set_table_width(table, total_cm):
    total = Cm(total_cm)
    table.autofit = False
    for row in table.rows:
        for cell in row.cells:
            cell.width = int(total / max(len(row.cells), 1))


# ---------------------------------------------------------------- figure build
def add_figure(doc, img_path: Path, caption: str, width_cm=14.6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_with_next = True
    if img_path.exists():
        p.add_run().add_picture(str(img_path), width=Cm(width_cm))
    else:
        run = p.add_run(f"[缺少图片 {img_path.name}]")
        set_font(run, size=10, italic=True)

    if caption:
        emit_caption(doc, caption, keep_next=False)


def emit_caption(doc, stripped: str, keep_next: bool = True):
    p = doc.add_paragraph(style="Caption")
    p.paragraph_format.keep_with_next = keep_next
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.space_before = Pt(12)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mm = re.match(r"^\*\*(表\s*\d+|图\s*\d+)\*\*\s*(.*)$", stripped)
    if mm:
        run = p.add_run(mm.group(1))
        set_font(run, size=10, bold=True, cn=CN_HEAD)
        rest = mm.group(2)
        if rest:
            if not rest.startswith(("（", "，", "。", "：", ":")):
                r0 = p.add_run(" ")
                set_font(r0, size=10)
            add_rich(p, rest, size=10)
    else:
        add_rich(p, stripped, size=10)
    return p


def add_equation(doc, latex: str, number: int):
    """Insert a numbered display equation.

    The equation is a 600 dpi transparent PNG rendered by
    ``render_equations.py``. Word's own OMML for these formulas was tried and
    rejected: the exported PDF collapsed the line box of the native math object
    and stacked consecutive equations on top of each other, while the image
    keeps a stable height. The formula sits on a centre tab and its number on a
    right tab, so both share one line at the standard geometry of a 21 cm page.
    """
    img = ROOT / "docs" / "paper_assets" / "equations" / f"eq{number}.png"
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(10)
    pf.space_after = Pt(10)
    pf.line_spacing = 1.0
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.keep_with_next = False
    pf.tab_stops.add_tab_stop(Cm(7.9), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Cm(15.8), WD_TAB_ALIGNMENT.RIGHT)

    p.add_run("\t")
    if img.exists():
        from PIL import Image

        with Image.open(img) as im:
            w_px, _h_px = im.size
        target_w_cm = min(w_px / 600 * 2.54, 12.6)
        p.add_run().add_picture(str(img), width=Cm(target_w_cm))
    else:
        add_math_runs(p, latex, size=12)

    num_run = p.add_run(f"\t（{number}）")
    set_font(num_run, size=11)


# ------------------------------------------------------------------ md parser
def split_row(line: str):
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def is_separator(line: str):
    return bool(re.match(r"^\|[\s:\-\|]+\|$", line.strip()))


def parse_table(lines, start):
    block = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        block.append(lines[i])
        i += 1
    rows = []
    for idx, raw in enumerate(block):
        if is_separator(raw):
            continue
        rows.append(split_row(raw))
    headers = rows[0]
    body = rows[1:]
    return headers, body, i


def resolve_image(target: str) -> Path:
    target = target.strip()
    if target.startswith("!"):
        target = target[1:]
    cand = (SRC.parent / target).resolve()
    if cand.exists():
        return cand
    name = Path(target).name
    alt = ROOT / "docs" / "paper_assets" / name
    if alt.exists():
        return alt
    return cand


def build():
    md = SRC.read_text(encoding="utf-8")
    lines = md.split("\n")

    doc = Document()
    build_styles(doc)
    sec0 = doc.sections[0]
    page_setup(sec0)
    add_footer_page_number(sec0)

    para_state = {"after_toc_inserted": False, "in_refs": False}
    eq_no = 0
    i = 0
    pending_caption = None

    first_heading_seen = False

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # ---- horizontal rule
        if stripped == "---":
            i += 1
            continue

        # ---- title
        if stripped.startswith("# ") and not stripped.startswith("## "):
            p = doc.add_paragraph(style="PaperTitle")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(24)
            p.paragraph_format.space_after = Pt(10)
            run = p.add_run(stripped[2:].strip())
            set_font(run, size=18, bold=True, cn=CN_HEAD)
            i += 1
            continue

        # ---- headings
        if stripped.startswith("#### "):
            doc.add_heading(stripped[5:].strip(), level=3)
            i += 1
            continue
        if stripped.startswith("### "):
            doc.add_heading(stripped[4:].strip(), level=2)
            i += 1
            continue
        if stripped.startswith("## "):
            title = stripped[3:].strip()
            if not para_state["after_toc_inserted"]:
                # title page block ends here -> TOC
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(18)
                p.paragraph_format.space_after = Pt(18)
                p.paragraph_format.first_line_indent = Cm(0)
                run = p.add_run("方法学论文")
                set_font(run, size=11, cn=CN_BODY)

                pbreak = doc.add_paragraph()
                pbreak.paragraph_format.first_line_indent = Cm(0)
                pbreak.add_run().add_break(WD_BREAK.PAGE)

                t = doc.add_paragraph(style="TocTitle")
                r = t.add_run("目  录")
                set_font(r, size=16, bold=True, cn=CN_HEAD)

                add_toc(doc)

                pbreak2 = doc.add_paragraph()
                pbreak2.paragraph_format.first_line_indent = Cm(0)
                pbreak2.add_run().add_break(WD_BREAK.PAGE)

                para_state["after_toc_inserted"] = True

            doc.add_heading(title, level=1)
            if title.startswith("参考文献"):
                para_state["in_refs"] = True
            i += 1
            continue

        # ---- code fence
        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip())
                i += 1
            i += 1
            for cl in code_lines:
                p = doc.add_paragraph(style="CodeBlock")
                run = p.add_run(cl if cl else " ")
                set_font(run, size=9, en=MONO)
            continue

        # ---- image
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)$", stripped)
        if m:
            img = resolve_image(m.group(1))
            # caption is the next non-empty line
            j = i + 1
            caption = ""
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith("**图"):
                caption = lines[j].strip()
                i = j + 1
            else:
                i += 1
            width = 15.2
            if img.name.startswith("fig5_fundus") or img.name.startswith("fig6b"):
                width = 13.0
            if img.name.startswith("fig1_architecture"):
                width = 15.4
            add_figure(doc, img, caption, width_cm=width)
            continue

        # ---- table
        if stripped.startswith("|"):
            headers, body, i = parse_table(lines, i)
            wide = len(headers) >= 10
            if wide:
                landscape_section(doc)
            if pending_caption:
                emit_caption(doc, pending_caption)
                pending_caption = None
            fs = 8 if len(headers) >= 8 else 9
            add_table(doc, headers, body, font_size=fs, total_cm=25.0 if wide else 15.6)
            sp = doc.add_paragraph()
            sp.paragraph_format.space_after = Pt(2)
            sp.paragraph_format.first_line_indent = Cm(0)
            if wide:
                portrait_section(doc)
            continue

        # ---- table caption lines (before the table)
        if re.match(r"^\*\*表\s*\d", stripped):
            pending_caption = stripped
            i += 1
            continue

        if re.match(r"^\*\*图\s*\d", stripped):
            emit_caption(doc, stripped)
            i += 1
            continue

        # ---- display equation block \[ ... \]
        if stripped.startswith("\\["):
            buf = []
            j = i
            text = stripped
            if "\\]" in text[2:]:
                inner = text[2 : text.index("\\]", 2)]
                i += 1
            else:
                buf.append(text[2:])
                j = i + 1
                while j < len(lines) and "\\]" not in lines[j]:
                    buf.append(lines[j])
                    j += 1
                tail = lines[j]
                buf.append(tail[: tail.index("\\]")])
                i = j + 1
                inner = " ".join(buf)
            tag = re.search(r"\\tag\{(\d+)\}", inner)
            latex = re.sub(r"\\tag\{\d+\}", "", inner).strip()
            eq_no = int(tag.group(1)) if tag else eq_no + 1
            add_equation(doc, latex, eq_no)
            continue

        # ---- reference / numbered list items
        if re.match(r"^\d+\.\s", stripped):
            if para_state["in_refs"]:
                p = doc.add_paragraph(style="Reference")
                add_rich(p, stripped, size=10)
            else:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.74)
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.space_after = Pt(6)
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                add_rich(p, stripped, size=12)
            i += 1
            continue

        # ---- ordinary paragraph
        if stripped in {"**方法学论文**", "**方法学论文。**"}:
            i += 1
            continue
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0.74)
        p.paragraph_format.space_after = Pt(6)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_rich(p, stripped, size=12)
        i += 1

    enable_update_fields(doc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    MIRROR.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(MIRROR))
    print(f"Wrote {OUT}")
    print(f"Mirrored {MIRROR}")
    print(f"Size MB {OUT.stat().st_size/1e6:.2f}")
    return OUT


def refresh_toc(path: Path):
    try:
        import win32com.client as win32
    except Exception as exc:
        print("skip TOC refresh:", exc)
        return False
    word = None
    try:
        word = win32.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(str(path), ReadOnly=False)
        doc.Fields.Update()
        for toc in doc.TablesOfContents:
            toc.Update()
        doc.Repaginate()
        doc.Save()
        doc.Close(SaveChanges=0)
        print("TOC refreshed")
        return True
    except Exception as exc:
        print("TOC refresh failed:", exc)
        return False
    finally:
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass


if __name__ == "__main__":
    out = build()
    if "--no-com" not in sys.argv:
        refresh_toc(out)
