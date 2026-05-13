#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Check every references.bib against dblp.org.

Reports two kinds of problem:
  - missing required fields (title/author/year + venue)
  - no plausible dblp match for the title

Usage: uv run check_refs.py
"""
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", re.DOTALL)
FIELD_RE = re.compile(r"(\w+)\s*=\s*[\{\"]?(.+?)[\}\"]?\s*(?:,\s*\n|\n\s*\})", re.DOTALL)
REQUIRED_VENUE = {"article": "journal", "inproceedings": "booktitle", "book": "publisher"}


def parse_bib(text: str):
    for etype, key, body in ENTRY_RE.findall(text):
        fields = {k.lower(): re.sub(r"\s+", " ", v).strip() for k, v in FIELD_RE.findall(body + "\n}")}
        yield etype.lower(), key, fields


def dblp_match(title: str) -> tuple[float, str] | None:
    try:
        r = httpx.get("https://dblp.org/search/publ/api", params={"q": title, "format": "json", "h": 5}, timeout=15)
        r.raise_for_status()
        hits = r.json().get("result", {}).get("hits", {}).get("hit", [])
    except Exception as e:
        return (0.0, f"dblp error: {e}")
    best = (0.0, "")
    for h in hits:
        t = h.get("info", {}).get("title", "")
        score = SequenceMatcher(None, title.lower(), t.lower()).ratio()
        if score > best[0]:
            best = (score, t)
    return best if hits else None


def check(bib_path: Path) -> list[str]:
    problems = []
    for etype, key, f in parse_bib(bib_path.read_text()):
        missing = [x for x in ("title", "author", "year") if x not in f]
        venue_field = REQUIRED_VENUE.get(etype)
        if venue_field and venue_field not in f:
            missing.append(venue_field)
        if missing:
            problems.append(f"  [{key}] missing fields: {', '.join(missing)}")
        if "title" in f:
            m = dblp_match(f["title"])
            if m is None:
                problems.append(f"  [{key}] no dblp results for: {f['title']!r}")
            elif m[0] < 0.7:
                problems.append(f"  [{key}] weak dblp match ({m[0]:.2f}): {f['title']!r} vs {m[1]!r}")
    return problems


def main() -> int:
    bibs = sorted(ROOT.rglob("references.bib"))
    bibs = [b for b in bibs if "public/" not in str(b)]
    if not bibs:
        print("no references.bib files found")
        return 0
    rc = 0
    for b in bibs:
        print(f"checking {b.relative_to(ROOT)}")
        probs = check(b)
        if probs:
            rc = 1
            print("\n".join(probs))
        else:
            print("  ok")
    return rc


if __name__ == "__main__":
    sys.exit(main())
