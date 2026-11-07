#!/usr/bin/env python3
"""Verify each manuscript reference resolves and that prose attributions match.

Three checks
------------
1. Resolution: every DOI must resolve at Crossref or DataCite (arXiv DOIs live at
   DataCite, so a Crossref-only check reports false failures).
2. Identity: the resolved title must appear in the reference string. Comparing
   against the whole line avoids fragile parsing of proceedings entries that
   begin with "In:".
3. Attribution: whenever the prose says "X 等人", X must be an author of one of
   the references cited in the same paragraph. Multi-word surnames such as
   "Van Calster" are matched in full.

Usage:  python scripts/verify_ref_authors.py
Exit 1 on any problem.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "paper" / "manuscript_zh_v2.md"
CACHE = ROOT / "results" / "real_rfmid" / "analysis_sel" / "stats" / "crossref_refs.json"

SURNAME = r"((?:Van|van|De|de|Von)\s+)?([A-Z][A-Za-z\-']{2,})"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def probe(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            payload = json.load(r)
    except Exception:  # noqa: BLE001
        return None
    if "message" in payload:
        return payload["message"]
    if "data" in payload:  # DataCite
        return payload["data"].get("attributes", {})
    return payload.get("attributes", payload)


def resolve(doi: str) -> tuple[str, dict] | tuple[None, None]:
    rec = probe(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    if rec and (rec.get("title") or rec.get("titles")):
        return "crossref", rec
    rec = probe(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}")
    if rec:
        return "datacite", rec
    return None, None


def title_of(rec: dict) -> str:
    t = rec.get("title") or rec.get("titles") or ""
    if isinstance(t, list):
        if t and isinstance(t[0], dict):
            return t[0].get("title") or ""
        return t[0] if t else ""
    return str(t)


def authors_of(rec: dict) -> list[str]:
    out = []
    for a in rec.get("author", []) or rec.get("creators", []) or []:
        if isinstance(a, dict):
            name = a.get("family") or a.get("name") or ""
            if name:
                out.append(norm(name))
    return out


def parse_refs(text: str) -> dict[int, dict]:
    refs: dict[int, dict] = {}
    block = text.split("## 参考文献", 1)[1]
    for line in block.splitlines():
        m = re.match(r"\s*(\d+)\.\s+(.*)", line)
        if not m:
            continue
        rest = m.group(2).strip()
        doi_m = re.search(r"doi:(10\.\S+)", rest)
        refs[int(m.group(1))] = {"raw": rest, "doi": doi_m.group(1) if doi_m else None}
    return refs


def cited_numbers(segment: str) -> list[int]:
    nums: list[int] = []
    for m in re.finditer(r"\[([0-9,\s\-]+)\]", segment):
        for part in m.group(1).split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                if a.strip().isdigit() and b.strip().isdigit():
                    nums.extend(range(int(a), int(b) + 1))
            elif part.isdigit():
                nums.append(int(part))
    return nums


def attribution_errors(text: str, records: dict[int, dict]) -> list[str]:
    problems: list[str] = []
    body = text.split("## 参考文献", 1)[0]
    for para in [p for p in body.split("\n\n") if p.strip()]:
        nums = cited_numbers(para)
        for m in re.finditer(SURNAME + r"\s*等[人者]", para):
            surface = (m.group(1) or "") + m.group(2)
            surname = norm(surface)
            if not nums:
                problems.append(f"attribution '{surface} 等' sits in a paragraph with no citation")
                continue
            hit = any(
                any(surname == f or surname in f.split() for f in authors_of(records[n]))
                for n in nums
                if n in records
            )
            if not hit:
                detail = ", ".join(
                    f"[{n}] {authors_of(records[n])[:3]}" for n in nums if n in records
                )
                problems.append(
                    f"attribution '{surface} 等' -> refs {nums}: surname absent; authors={detail}"
                )
    return problems


def main() -> int:
    text = MD.read_text(encoding="utf-8")
    refs = parse_refs(text)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    records: dict[int, dict] = {}
    problems: list[str] = []

    for num, ref in sorted(refs.items()):
        if not ref["doi"]:
            print(f"[{num}] no DOI (dataset / software): {ref['raw'][:60]}")
            continue
        registry, rec = resolve(ref["doi"])
        if rec is None:
            problems.append(f"ref {num}: DOI resolves at neither Crossref nor DataCite -> {ref['doi']}")
            continue
        records[num] = rec
        got = title_of(rec)
        key = {w for w in norm(got).split() if len(w) > 2}
        in_line = len(key & set(norm(ref["raw"]).split())) / max(1, len(key))
        print(f"[{num}] {'ok ' if in_line >= 0.8 else '?? '}{registry:8s} {got[:62]}")
        if in_line < 0.8:
            problems.append(
                f"ref {num}: resolved title not found in the reference line\n"
                f"    resolved: {got}\n    line    : {ref['raw'][:110]}"
            )

    CACHE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    problems.extend(attribution_errors(text, records))

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(" -", p)
        return 1
    print("\nAll references resolve and every named attribution matches a real author.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
