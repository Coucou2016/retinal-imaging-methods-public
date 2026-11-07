#!/usr/bin/env python
"""Readability audit: sentence length, openers, repeated phrasing.

The style brief asks for short sentences and plain connectives. This script is
the mechanical check behind that request: it reports sentences longer than 90
Chinese characters, the most repeated sentence openers, and phrases that survive
from a first draft.
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "paper" / "manuscript_zh_v2.md"

SKIP_PREFIX = ("#", "|", "!", "\\[", "**表", "**图", "```", "---")
BANNED = ["链路", "回路", "闭环", "耦合", "赋能", "抓手", "生态位", "护城河", "壁垒"]


def prose_blocks(text: str) -> list[str]:
    out = []
    for raw in text.split("\n\n"):
        block = raw.strip()
        if not block:
            continue
        if block.startswith(SKIP_PREFIX):
            continue
        out.append(block)
    return out


def sentences(block: str) -> list[str]:
    body = re.sub(r"\\\(.+?\\\)", "算式", block)
    body = body.replace("**", "")
    parts = re.split(r"(?<=[。！？])", body)
    return [p.strip() for p in parts if len(p.strip()) > 1]


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    blocks = prose_blocks(text)
    sents: list[str] = []
    for b in blocks:
        sents.extend(sentences(b))

    long_sents = [s for s in sents if len(s) > 90]
    openers = collections.Counter(s[:4] for s in sents)

    print(f"paragraphs {len(blocks)}  sentences {len(sents)}")
    print(f"over 90 chars: {len(long_sents)}")
    for s in long_sents[:20]:
        print(f"  {len(s):3d}  {s[:80]}")

    print("\nmost repeated openers")
    for k, v in openers.most_common(10):
        print(f"  {k!r}: {v}")

    print("\nbanned vocabulary")
    hits = {w: text.count(w) for w in BANNED if text.count(w)}
    print(f"  {hits if hits else 'none'}")

    fillers = ["值得注意的是", "综上所述", "总而言之", "需要指出的是", "众所周知"]
    print("\nfiller phrases")
    fh = {w: text.count(w) for w in fillers if text.count(w)}
    print(f"  {fh if fh else 'none'}")

    quotes = re.findall(r"“[^”]{0,20}”", text)
    print(f"\nquoted phrases: {len(quotes)}")
    for q in quotes[:10]:
        print(f"  {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
