#!/usr/bin/env python3
"""Verify every reference DOI/title against Crossref and print a verdict table."""
from __future__ import annotations
# --- flat-snapshot bootstrap (auto-generated) -------------------------------
import os as _os
import sys as _sys

_FLAT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _FLAT_ROOT not in _sys.path:
    _sys.path.insert(0, _FLAT_ROOT)
# ---------------------------------------------------------------------------


import json
import sys
import urllib.parse
import urllib.request

REFS = [
    ("10.1038/s41591-026-04359-w", "AI framework for multidisease detection via retinal imaging"),
    ("10.1038/s41586-023-06555-x", "A foundation model for generalizable disease detection from retinal images"),
    ("10.1007/978-3-030-32239-7_6", "Evaluation of retinal image quality assessment networks in different color-spaces"),
    ("10.3390/data6020014", "Retinal Fundus Multi-Disease Image Dataset (RFMiD)"),
    ("10.1371/journal.pdig.0000454", "BRSET: a Brazilian multilabel ophthalmological dataset"),
    ("10.1177/0272989X06295361", "Decision curve analysis: a novel method for evaluating prediction models"),
    ("10.1038/s41598-023-38610-y", "Towards population-independent, multi-disease detection in fundus photographs"),
    ("10.1038/s41746-024-01109-5", "RETFound-enhanced community-based fundus disease screening"),
    ("10.3233/SHTI210537", "Multi-disease detection in retinal imaging based on ensembling heterogeneous deep learning models"),
    ("10.1167/tvst.13.4.8", "Enhancing Retinal Fundus Image Quality Assessment With Swin-Transformer-Based Learning Across Multiple Color-Spaces"),
    ("10.1177/0272989X8900900307", "Analyzing a portion of the ROC curve"),
]


def crossref(doi: str) -> dict | None:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ref-audit/1.0 (mailto:noreply@example.org)"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read().decode("utf-8"))["message"]
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Crossref lookup failed: {exc}")
        return None


def main() -> None:
    rows = []
    for doi, claimed in REFS:
        print(f"\n== {doi}")
        msg = crossref(doi)
        if not msg:
            rows.append((doi, claimed, "NOT_FOUND", "", ""))
            continue
        title = (msg.get("title") or [""])[0]
        journal = (msg.get("container-title") or [""])[0]
        year = ""
        for key in ("published-print", "published-online", "issued", "published"):
            if msg.get(key):
                year = str(msg[key]["date-parts"][0][0])
                break
        vol = msg.get("volume", "")
        page = msg.get("page", "")
        authors = msg.get("author") or []
        first = ""
        if authors:
            a = authors[0]
            first = f"{a.get('family','')} {a.get('given','')[:1]}".strip()
        print(f"  title  : {title}")
        print(f"  journal: {journal} {year};{vol}:{page}")
        print(f"  first  : {first}  (n={len(authors)})")
        verdict = "OK"
        cl = claimed.lower()
        tl = title.lower()
        if cl[:30] not in tl and tl[:30] not in cl:
            verdict = "TITLE_DIFFERS"
        rows.append((doi, claimed, verdict, title, f"{journal} {year};{vol}:{page}"))

    print("\n\n=== SUMMARY ===")
    for doi, claimed, verdict, title, cite in rows:
        print(f"{verdict:14s} {doi}")
        if verdict != "OK":
            print(f"    claimed: {claimed}")
            print(f"    actual : {title}")
            print(f"    cite   : {cite}")


if __name__ == "__main__":
    sys.exit(main())
