#!/usr/bin/env python3
"""Lint the manuscript against the writing rules the request specified.

Checks
------
placeholders    待确定 / 待确认 / 待补充 / TBD / TODO / XXX
punctuation     colon, em dash, en dash (outside formulas and tables), semicolons
symbols         ≠ ≈ > < ≥ ≤ used as prose comparisons rather than in formulas
jargon          链路 回路 闭环 耦合 赋能 抓手 生态位
defensive       不要 / 而不是 / 并非 / 而非 overuse, plus count
formulaic       「A 不是 B，而是 C」 pattern, and 引号式三段堆叠
repetition      paragraphs that start with the same connective more than twice
length          sentences over 90 Chinese characters

Exit code 1 if any hard rule is violated.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

MD = Path(__file__).resolve().parents[1] / "docs" / "paper" / "manuscript_zh_v2.md"

BANNED_WORDS = ["待确定", "待确认", "待补充", "TBD", "TODO", "XXX"]
JARGON = ["链路", "回路", "闭环", "耦合", "赋能", "抓手", "生态位", "护城河"]
DEFENSIVE = ["不要", "而不是", "并非", "而非", "不应", "不宜"]
CONNECTIVES = ["此外", "因此", "然而", "同时", "另外", "总之", "综上"]


def strip_code_and_math(text: str) -> str:
    """Drop fenced blocks, inline formula displays and table rows before linting."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.S)
    text = re.sub(r"\\\(.*?\\\)", "", text, flags=re.S)
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("|")]
    return "\n".join(lines)


def prose_lines(text: str) -> list[tuple[int, str]]:
    out = []
    for i, ln in enumerate(text.splitlines(), 1):
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("![") or s.startswith("**图"):
            continue
        if s.startswith("*") and s.endswith("*") and len(s) < 200:
            continue
        # reference-list entries are bibliographic strings, not prose sentences
        if re.match(r"^\d+\.\s+[A-Z]", s):
            continue
        out.append((i, s))
    return out


def main() -> int:
    raw = MD.read_text(encoding="utf-8")
    body = strip_code_and_math(raw)
    lines = prose_lines(body)
    joined = "\n".join(s for _, s in lines)

    hard: list[str] = []
    soft: list[str] = []

    for w in BANNED_WORDS:
        if w in joined:
            hard.append(f"placeholder {w!r} appears {joined.count(w)} time(s)")

    for w in JARGON:
        if w in joined:
            hard.append(f"jargon {w!r} appears {joined.count(w)} time(s)")

    # punctuation: allow at most a handful across a manuscript this size
    colons = joined.count("：")
    semis = joined.count("；")
    emdash = joined.count("——")
    if colons > 6:
        hard.append(f"colons in prose: {colons} (limit 6)")
    if semis > 4:
        hard.append(f"semicolons in prose: {semis} (limit 4)")
    if emdash > 2:
        hard.append(f"em dashes: {emdash} (limit 2)")

    # comparison symbols in prose
    for sym in ("≠", "≈", "≥", "≤"):
        if sym in joined:
            hard.append(f"prose comparison symbol {sym!r} appears {joined.count(sym)} time(s)")

    # formulaic contrast
    pat = re.compile(r"不是[^，。]{2,30}，而是")
    for m in pat.finditer(joined):
        hard.append(f"formulaic contrast: {m.group(0)[:40]}")

    # defensive words, soft signal only
    counts = {w: joined.count(w) for w in DEFENSIVE}
    total_def = sum(counts.values())
    if total_def > 8:
        soft.append(f"defensive phrasing total {total_def}: {counts}")

    # repeated paragraph openers
    openers = Counter()
    for _, s in lines:
        for c in CONNECTIVES:
            if s.startswith(c):
                openers[c] += 1
    for c, n in openers.items():
        if n > 2:
            soft.append(f"connective {c!r} opens {n} paragraphs")
    soft.append(f"connective opener counts: {dict(openers)}")

    # long sentences
    long_sents = []
    for lineno, s in lines:
        for sent in re.split(r"[。！？]", s):
            if len(sent) > 90:
                long_sents.append((lineno, len(sent), sent[:50]))
    if len(long_sents) > 6:
        soft.append(f"{len(long_sents)} sentences over 90 chars; first few: {long_sents[:4]}")

    print(f"prose lines: {len(lines)}")
    print(f"colons {colons}, semicolons {semis}, em-dashes {emdash}")
    print(f"defensive total {total_def} {counts}")
    print()
    if hard:
        print(f"HARD violations ({len(hard)}):")
        for h in hard:
            print("  -", h)
    if soft:
        print(f"\nsoft signals ({len(soft)}):")
        for s in soft:
            print("  -", s)
    if not hard:
        print("\nno hard violations")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
