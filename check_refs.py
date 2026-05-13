#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Check or auto-fix every references.bib against dblp.org.

Usage:
  uv run check_refs.py          # report problems
  uv run check_refs.py --fix    # rewrite matched entries from dblp metadata
"""

import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
THROTTLE_S = 1.2  # dblp asks for ~1 req/sec
RETRIES = 4
HITS = 15
MATCH_REPORT = 0.7  # below: flag as weak match
MATCH_FIX = 0.85  # above (and year ±1): safe to overwrite

ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", re.DOTALL)
FIELD_RE = re.compile(
    r"(\w+)\s*=\s*[\{\"]?(.+?)[\}\"]?\s*(?:,\s*\n|\n\s*\})", re.DOTALL
)
REQUIRED_VENUE = {
    "article": "journal",
    "inproceedings": "booktitle",
    "book": "publisher",
}

DBLP_TYPE_TO_BIB = {
    "Journal Articles": "article",
    "Conference and Workshop Papers": "inproceedings",
    "Books and Theses": "book",
    "Informal and Other Publications": "misc",
    "Reference Works": "misc",
    "Editorship": "misc",
    "Parts in Books or Collections": "incollection",
}


def parse_bib(text):
    for etype, key, body in ENTRY_RE.findall(text):
        fields = {
            k.lower(): re.sub(r"\s+", " ", v).strip()
            for k, v in FIELD_RE.findall(body + "\n}")
        }
        yield etype.lower(), key, fields


LATEX_ACCENTS = {
    "o": "ö",
    "a": "ä",
    "u": "ü",
    "O": "Ö",
    "A": "Ä",
    "U": "Ü",
}
LATEX_ACCENTS_ACUTE = {
    "e": "é",
    "a": "á",
    "i": "í",
    "o": "ó",
    "u": "ú",
    "E": "É",
    "n": "ń",
}
LATEX_ACCENTS_GRAVE = {"e": "è", "a": "à", "i": "ì", "o": "ò"}
LATEX_ACCENTS_CIRC = {"e": "ê", "a": "â", "i": "î", "o": "ô"}
LATEX_ACCENTS_TILDE = {"n": "ñ", "o": "õ", "a": "ã"}
LATEX_OTHER = {"L": "Ł", "l": "ł", "ss": "ß"}


def clean_title(t: str) -> str:
    # Both `{\"o}` and `\"{o}` and `\"o` forms occur.
    for letter, ch in LATEX_ACCENTS.items():
        t = re.sub(rf"\{{?\\\"\{{?{letter}\}}?\}}?", ch, t)
    for letter, ch in LATEX_ACCENTS_ACUTE.items():
        t = re.sub(rf"\{{?\\'\{{?{letter}\}}?\}}?", ch, t)
    for letter, ch in LATEX_ACCENTS_GRAVE.items():
        t = re.sub(rf"\{{?\\`\{{?{letter}\}}?\}}?", ch, t)
    for letter, ch in LATEX_ACCENTS_CIRC.items():
        t = re.sub(rf"\{{?\\\^\{{?{letter}\}}?\}}?", ch, t)
    for letter, ch in LATEX_ACCENTS_TILDE.items():
        t = re.sub(rf"\{{?\\~\{{?{letter}\}}?\}}?", ch, t)
    t = re.sub(r"\{?\\c\s*\{?c\}?\}?", "ç", t)
    for k, v in LATEX_OTHER.items():
        t = t.replace(rf"{{\{k}}}", v).replace(rf"\{k}{{}}", v)
    return re.sub(r"[{}\\]", "", t).strip()


def first_surname(authors_field: str) -> str:
    first = authors_field.split(" and ")[0].strip()
    if "," in first:
        return first.split(",")[0].strip()
    return first.rsplit(" ", 1)[-1].strip()


def dblp_query(title: str) -> list[dict]:
    for attempt in range(RETRIES):
        try:
            time.sleep(THROTTLE_S)
            r = httpx.get(
                "https://dblp.org/search/publ/api",
                params={"q": title, "format": "json", "h": HITS},
                timeout=30,
                headers={"User-Agent": "edadaltocg-bib-check/1.0"},
            )
            if r.status_code == 429:
                time.sleep(2 ** (attempt + 3))
                continue
            r.raise_for_status()
            hits = r.json().get("result", {}).get("hits", {}).get("hit", [])
            return [h["info"] for h in hits]
        except (httpx.HTTPError, httpx.TimeoutException):
            time.sleep(2 ** (attempt + 1))
    return []


def best_match(title: str, year: str, hits: list[dict]) -> tuple[float, dict | None]:
    best = (0.0, None)
    for h in hits:
        score = SequenceMatcher(None, title.lower(), h.get("title", "").lower()).ratio()
        if year and h.get("year") and abs(int(h["year"]) - int(year)) <= 1:
            score += 0.05
        if score > best[0]:
            best = (score, h)
    return best


def authors_from_dblp(info: dict) -> str:
    a = info.get("authors", {}).get("author", [])
    if isinstance(a, dict):
        a = [a]
    names = []
    for e in a:
        name = e.get("text") if isinstance(e, dict) else e
        if not name:
            continue
        name = re.sub(
            r"\s+\d{4}$", "", name
        )  # drop dblp disambiguator like "Tri Dao 0001"
        first, _, last = name.rpartition(" ")
        names.append(f"{last}, {first}" if first else name)
    return " and ".join(names)


def render_entry(key: str, info: dict) -> str:
    etype = DBLP_TYPE_TO_BIB.get(info.get("type", ""), "misc")
    venue = info.get("venue", "")
    if isinstance(venue, list):
        venue = " / ".join(venue)
    fields: dict[str, str] = {
        "title": info.get("title", "").rstrip("."),
        "author": authors_from_dblp(info),
        "year": str(info.get("year", "")),
    }
    if etype == "inproceedings" or etype == "incollection":
        fields["booktitle"] = venue
    elif etype == "article":
        fields["journal"] = venue
        if info.get("volume"):
            fields["volume"] = str(info["volume"])
        if info.get("number"):
            fields["number"] = str(info["number"])
        if info.get("pages"):
            fields["pages"] = str(info["pages"])
    elif etype == "book":
        if info.get("publisher"):
            fields["publisher"] = info["publisher"]
    if info.get("doi"):
        fields["doi"] = info["doi"]
    if info.get("ee") and "doi" not in fields:
        fields["url"] = info["ee"]
    body = ",\n".join(f"  {k:<13} = {{{v}}}" for k, v in fields.items() if v)
    return f"@{etype}{{{key},\n{body}\n}}"


def process(bib_path: Path, fix: bool) -> list[str]:
    text = bib_path.read_text()
    problems: list[str] = []
    replacements: list[tuple[re.Match, str]] = []
    for etype, key, f in parse_bib(text):
        missing = [x for x in ("title", "author", "year") if x not in f]
        venue_field = REQUIRED_VENUE.get(etype)
        if venue_field and venue_field not in f:
            missing.append(venue_field)
        if missing:
            problems.append(f"  [{key}] missing: {', '.join(missing)}")
        title = clean_title(f.get("title", ""))
        if not title:
            continue
        hits = dblp_query(title)
        score, hit = best_match(title, f.get("year", ""), hits)
        if (not hit or score < MATCH_REPORT) and "author" in f:
            surname = first_surname(f["author"])
            if surname:
                hits2 = dblp_query(f"{title} {surname}")
                s2, h2 = best_match(title, f.get("year", ""), hits2)
                if s2 > score:
                    score, hit = s2, h2
        if not hit:
            problems.append(f"  [{key}] no dblp results for {title!r}")
            continue
        if score < MATCH_REPORT:
            problems.append(f"  [{key}] weak match ({score:.2f}): {hit.get('title')!r}")
            continue
        if fix and score >= MATCH_FIX:
            existing_first = first_surname(f.get("author", "")).lower()
            dblp_authors = authors_from_dblp(hit)
            dblp_first = first_surname(dblp_authors).lower()
            existing_year = f.get("year", "").strip().strip("{}")
            dblp_year = str(hit.get("year", ""))
            year_ok = (
                not existing_year
                or not dblp_year
                or abs(int(existing_year) - int(dblp_year)) <= 1
            )
            if existing_first and existing_first != dblp_first:
                problems.append(
                    f"  [{key}] dblp first author {dblp_first!r} != existing {existing_first!r}; skipped"
                )
                continue
            if not year_ok:
                problems.append(
                    f"  [{key}] dblp year {dblp_year} too far from existing {existing_year}; skipped"
                )
                continue
            new = render_entry(key, hit)
            m = re.search(
                rf"@\w+\s*\{{\s*{re.escape(key)}\s*,.*?\n\}}", text, re.DOTALL
            )
            if m and m.group() != new:
                replacements.append((m, new))
                problems.append(f"  [{key}] updated from dblp ({score:.2f})")
    if fix and replacements:
        new_text = text
        for m, new in sorted(replacements, key=lambda r: r[0].start(), reverse=True):
            new_text = new_text[: m.start()] + new + new_text[m.end() :]
        bib_path.write_text(new_text)
    return problems


def main() -> int:
    fix = "--fix" in sys.argv
    bibs = sorted(p for p in ROOT.rglob("references.bib") if "public/" not in str(p))
    if not bibs:
        print("no references.bib files found")
        return 0
    rc = 0
    for b in bibs:
        print(f"checking {b.relative_to(ROOT)}{' (--fix)' if fix else ''}")
        probs = process(b, fix)
        if probs:
            rc = 1
            print("\n".join(probs))
        else:
            print("  ok")
    return rc


if __name__ == "__main__":
    sys.exit(main())
