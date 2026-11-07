#!/usr/bin/env python3
"""Render the Chinese manuscript to a self-contained HTML page with a live TOC.

Equations are typeset by MathJax from the LaTeX already present in the markdown,
figures are embedded as relative links to ``docs/paper_assets``, and the table of
contents is generated from the heading hierarchy.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "paper" / "manuscript_zh_v2.md"
OUT = ROOT / "docs" / "paper" / "manuscript_zh_v3.html"

CSS = """
:root { --ink:#1a1a1a; --muted:#5c5c5c; --rule:#d5d5d5; --accent:#7a4a2b; }
* { box-sizing: border-box; }
body {
  margin:0 auto; padding:48px 24px 96px; max-width: 860px;
  font-family: "Songti SC","SimSun","Source Han Serif SC",serif;
  font-size: 16px; line-height: 1.85; color: var(--ink);
  background:#fff;
}
h1 { font-family:"Heiti SC","SimHei",sans-serif; font-size:26px; line-height:1.45;
     margin:0 0 8px; text-align:center; letter-spacing:.5px; }
h1 + p em, h1 + p { text-align:center; color:var(--muted); font-size:15px; }
h2 { font-family:"Heiti SC","SimHei",sans-serif; font-size:20px; margin:44px 0 14px;
     padding-bottom:6px; border-bottom:1px solid var(--rule); }
h3 { font-family:"Heiti SC","SimHei",sans-serif; font-size:17px; margin:30px 0 10px; }
p { margin:0 0 12px; text-align:justify; }
strong { font-weight:700; }
img { display:block; margin:18px auto 6px; max-width:100%; height:auto; }
p > em:only-child { display:block; text-align:center; color:var(--muted); }
table { border-collapse:collapse; width:100%; margin:10px auto 20px; font-size:13.5px; }
th, td { padding:5px 8px; text-align:center; }
th { border-top:1.6px solid #000; border-bottom:0.9px solid #000; font-weight:700;
     font-family:"Heiti SC","SimHei",sans-serif; }
tr:last-child td { border-bottom:1.6px solid #000; }
td:first-child, th:first-child { text-align:left; }
figcaption, .cap { text-align:center; font-size:14px; color:var(--muted); margin:0 0 22px; }
.eq { text-align:center; margin:16px 0; overflow-x:auto; }
.tag { float:right; color:var(--muted); }
.refs { font-size:14px; }
.refs li { margin-bottom:8px; text-align:justify; }
pre { background:#f6f6f4; padding:12px 14px; overflow-x:auto; font-size:13px;
      line-height:1.6; border-left:3px solid var(--rule); }
code { font-family:"Consolas","SFMono-Regular",monospace; font-size:13.5px; }
#toc { border:1px solid var(--rule); padding:18px 24px; margin:28px 0 36px;
       background:#fbfaf7; }
#toc h2 { border:none; margin:0 0 10px; font-size:17px; text-align:center; }
#toc ol { list-style:none; padding-left:0; margin:0; font-size:14.5px; }
#toc ol ol { padding-left:18px; margin-top:2px; }
#toc li { margin:3px 0; }
#toc a { color:var(--ink); text-decoration:none; }
#toc a:hover { color:var(--accent); text-decoration:underline; }
@media print {
  body { max-width:none; padding:0; font-size:11.5pt; }
  #toc { page-break-after: always; }
  h2, h3 { page-break-after: avoid; }
  img, table { page-break-inside: avoid; }
}
"""

MATHJAX = """
<script>
window.MathJax = {
  tex: { inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']],
         tags: 'none' },
  svg: { fontCache: 'global' }
};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
"""


def slug(text: str, used: dict[str, int]) -> str:
    base = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text).strip("-") or "sec"
    used[base] = used.get(base, 0) + 1
    return base if used[base] == 1 else f"{base}-{used[base]}"


def inline(text: str) -> str:
    """Convert inline markdown to HTML, keeping LaTeX for MathJax."""
    hold: list[str] = []

    def stash(m: re.Match) -> str:
        hold.append(m.group(0))
        return f"\x00{len(hold) - 1}\x00"

    text = re.sub(r"\\\(.+?\\\)", stash, text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", r"<sup>[\1]</sup>", text)

    def restore(m: re.Match) -> str:
        return hold[int(m.group(1))]

    return re.sub(r"\x00(\d+)\x00", restore, text)


def figure_path(raw: str) -> str:
    p = (SRC.parent / raw).resolve()
    return Path("..") / "paper_assets" / p.name if p.exists() else Path(raw)


def main() -> int:
    lines = SRC.read_text(encoding="utf-8").split("\n")
    used: dict[str, int] = {}
    body: list[str] = []
    toc: list[tuple[int, str, str]] = []

    i = 0
    in_refs = False
    in_code = False
    code: list[str] = []
    table: list[str] = []

    def flush_table() -> None:
        nonlocal table
        if not table:
            return
        rows = []
        for ln in table:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            rows.append(cells)
        if rows:
            out = ["<table>"]
            out.append("<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr></thead>")
            out.append("<tbody>")
            for r in rows[1:]:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
            body.append("\n".join(out))
        table = []

    while i < len(lines):
        raw = lines[i]
        s = raw.strip()

        if in_code:
            if s.startswith("```"):
                body.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
                code, in_code = [], False
            else:
                code.append(raw)
            i += 1
            continue

        if not s:
            flush_table()
            i += 1
            continue

        if s.startswith("```"):
            flush_table()
            in_code = True
            i += 1
            continue

        if s == "---":
            flush_table()
            i += 1
            continue

        if s.startswith("|"):
            table.append(s)
            i += 1
            continue
        flush_table()

        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)$", s)
        if m:
            body.append(f'<img src="{figure_path(m.group(1))}" alt="">')
            i += 1
            continue

        if s.startswith("# "):
            body.append(f"<h1>{inline(s[2:].strip())}</h1>")
            i += 1
            continue

        for lvl in (3, 2):
            if s.startswith("#" * lvl + " "):
                title = s[lvl + 1:].strip()
                anchor = slug(title, used)
                if title.startswith("参考文献"):
                    in_refs = True
                if lvl == 2:
                    body.append(f'<h2 id="{anchor}">{inline(title)}</h2>')
                    toc.append((2, title, anchor))
                else:
                    body.append(f'<h3 id="{anchor}">{inline(title)}</h3>')
                    toc.append((3, title, anchor))
                i += 1
                break
        else:
            if s.startswith("**表") or s.startswith("**图"):
                cls = ' class="cap"'
                body.append(f"<p{cls}>{inline(s)}</p>")
                i += 1
                continue

            if s.startswith("\\["):
                buf = [s[2:]]
                j = i + 1
                while j < len(lines) and "\\]" not in lines[j]:
                    buf.append(lines[j])
                    j += 1
                if j < len(lines):
                    head, _, _tail = lines[j].partition("\\]")
                    buf.append(head)
                    i = j + 1
                else:
                    i = j
                tex = " ".join(buf).strip()
                tag = re.search(r"\\tag\{(\d+)\}", tex)
                tex = re.sub(r"\\tag\{\d+\}", "", tex).strip()
                label = f'<span class="tag">({tag.group(1)})</span>' if tag else ""
                body.append(f'<div class="eq">{label}$${tex}$$</div>')
                continue

            if re.match(r"^\d+\.\s", s) and in_refs:
                body.append(f'<p class="refs">{inline(s)}</p>')
                i += 1
                continue

            body.append(f"<p>{inline(s)}</p>")
            i += 1

    flush_table()

    # group level-2 sections with their level-3 children
    sections: list[tuple[str, str, list[tuple[str, str]]]] = []
    for lvl, title, anchor in toc:
        if lvl == 2:
            sections.append((title, anchor, []))
        elif sections:
            sections[-1][2].append((title, anchor))

    toc_lines = ['<nav id="toc">', "<h2>目 录</h2>", "<ol>"]
    for title, anchor, subs in sections:
        toc_lines.append(f'<li><a href="#{anchor}">{html.escape(title)}</a>')
        if subs:
            toc_lines.append("<ol>")
            for st, sa in subs:
                toc_lines.append(f'<li><a href="#{sa}">{html.escape(st)}</a></li>')
            toc_lines.append("</ol>")
        toc_lines.append("</li>")
    toc_lines.append("</ol>")
    toc_lines.append("</nav>")
    toc_html = "\n".join(toc_lines)

    # place the TOC right after the title block (h1 plus its subtitle paragraph)
    parts = list(body)
    insert_at = 0
    for k, blk in enumerate(parts):
        if blk.startswith("<h1"):
            insert_at = k + 1
            break
    if insert_at < len(parts) and parts[insert_at].startswith("<p>"):
        insert_at += 1
    parts.insert(insert_at, toc_html)

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>视网膜多病种筛查中的单调质量路由与共享多标签头</title>
<style>{CSS}</style>
{MATHJAX}
</head>
<body>
{chr(10).join(parts)}
</body>
</html>
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB), "
          f"{len(toc)} headings, {len(body)} blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
