"""Correct ordering audit: first mention must precede (or coincide with) caption.

Back-references later in the prose are legitimate and must not be flagged.
Equation mentions conventionally precede the displayed definition, so an
equation may be referenced before it is typeset.
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "docs" / "paper" / "manuscript_zh_v2.md"
text = SRC.read_text(encoding="utf-8")
lines = text.split("\n")

FIG_MENTION = re.compile(r"图\s*(\d+)")
TAB_MENTION = re.compile(r"表\s*(\d+)")
EQ_MENTION = re.compile(r"式\s*\((\d+)\)")


def scan():
    fig_mentions, tab_mentions, eq_mentions = [], [], []
    fig_caps, tab_caps, eq_defs = [], [], []
    in_refs = False
    for idx, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("## 参考文献"):
            in_refs = True
        if in_refs:
            continue
        if s.startswith("**图"):
            m = re.match(r"\*\*图\s*(\d+)\*\*", s)
            if m:
                fig_caps.append((int(m.group(1)), idx))
            continue
        if s.startswith("**表"):
            m = re.match(r"\*\*表\s*(\d+)\*\*", s)
            if m:
                tab_caps.append((int(m.group(1)), idx))
            continue
        if s.startswith("!["):
            continue
        for m in FIG_MENTION.finditer(s):
            fig_mentions.append((int(m.group(1)), idx))
        for m in TAB_MENTION.finditer(s):
            tab_mentions.append((int(m.group(1)), idx))
        for m in EQ_MENTION.finditer(s):
            eq_mentions.append((int(m.group(1)), idx))
        for m in re.finditer(r"\\tag\{(\d+)\}", s):
            eq_defs.append((int(m.group(1)), idx))
    return fig_mentions, tab_mentions, eq_mentions, fig_caps, tab_caps, eq_defs


def first_order(items):
    seen = []
    for num, _ in items:
        if num not in seen:
            seen.append(num)
    return seen


def first_line(items):
    out = {}
    for num, ln in items:
        out.setdefault(num, ln)
    return out


fig_m, tab_m, eq_m, fig_c, tab_c, eq_d = scan()
problems = []

for label, mentions, caps in (("图", fig_m, fig_c), ("表", tab_m, tab_c)):
    order = first_order(mentions)
    capnums = [n for n, _ in caps]
    expected = list(range(1, len(capnums) + 1))
    if capnums != expected:
        problems.append(f"{label} caption numbering is not 1..{len(capnums)}: {capnums}")
    if order != expected:
        problems.append(f"{label} first-mention order {order} != {expected}")
    fl = first_line(mentions)
    capmap = dict(caps)
    for num, cap_ln in caps:
        if num not in fl:
            problems.append(f"{label} {num} never cited in the prose")
        elif fl[num] > cap_ln:
            problems.append(
                f"{label} {num} first cited at line {fl[num]}, after its caption at line {cap_ln}"
            )

eqdef = dict(eq_d)
eqorder = first_order(eq_m)
if [n for n, _ in eq_d] != list(range(1, len(eq_d) + 1)):
    problems.append(f"equation numbering is not 1..N: {[n for n, _ in eq_d]}")
if eqorder != list(range(1, len(eqorder) + 1)):
    problems.append(f"equation first-mention order {eqorder} != ascending")
for num, ln in eq_m:
    if num not in eqdef:
        problems.append(f"equation ({num}) cited at line {ln} but never defined")

print(f"figures  : {len(fig_c)} captions, first mentions {first_order(fig_m)}")
print(f"tables   : {len(tab_c)} captions, first mentions {first_order(tab_m)}")
print(f"equations: {len(eq_d)} defined, mentions {eqorder}")
print()
if problems:
    for p in problems:
        print("PROBLEM:", p)
    sys.exit(1)
print("ORDER OK: every figure and table is cited before its caption, numbering ascends,"
      " and every cited equation is defined.")
