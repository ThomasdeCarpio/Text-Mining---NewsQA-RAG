"""Remove surviving HTML page furniture from the benchmark corpus.

The staged corpus is the HTML-extraction output of the 2011 CNN pages, and some
of the page chrome came through as if it were prose: video teasers, the
"E-mail to a friend" widget, publisher footers, and block-boundary newline
runs. Those strings are chunked, embedded and retrieved exactly like article
text, so they are noise a retriever has to score around.

Evidence spans are character offsets into `context`, so nothing can be deleted
without moving them. This script records every deletion, remaps each span, and
then VERIFIES that the span still quotes the same text. A span that cannot be
remapped is flagged, never silently dropped.

Non-destructive: writes a parallel `cleaned/` tree and leaves the locked
benchmark under `final/` untouched.

Run:  python scripts/clean_corpus.py [--apply]
Without --apply it reports what would change and writes nothing.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "data" / "evaluation" / "newsqa_200_11064"
STAGING = DATASET / "staging"
FINAL = DATASET / "final"
CLEANED = DATASET / "cleaned"

# Each rule deletes its whole match.
#
# Deliberately NOT included: the "All About" pattern and the bullet separator.
# Both looked like widget markers in the residue scan, but reading the matches
# showed most are prose or genuine list markers. Removing them would delete
# article content, which is a worse error than leaving furniture behind.
RULES: list[tuple[str, str]] = [
    # "Watch Bush talk about the aftermath of 9/11 >>" - a video caption line.
    # The guillemet survived extraction as mojibake, so accept either form.
    (
        r"(?i)(?:^|(?<=[.!?\n]))[ \t]*(?:watch|see|learn|read|look)\b"
        r"[^.\n]{0,120}?[»�][ \t]*",
        "video / gallery teaser line",
    ),
    # Trailing share widget, plus whatever the page appended after it.
    (r"(?i)\s*E-mail to a friend\s*.*\Z", "e-mail-to-a-friend widget"),
    # Publisher footers.
    (
        r"(?i)\n[^\n]*?(?:copyright\s*)?[©�]?\s*\d{4}[^\n]*"
        r"all rights reserved[^\n]*",
        "copyright footer",
    ),
    (r"(?i)\n\s*subscribe to [^\n]{0,60}\Z", "subscribe footer"),
]

NEWLINE_RUN = re.compile(r"\n{3,}")


def deletions(text: str) -> list[tuple[int, int, str]]:
    """Spans to delete, as (start, end, rule), merged and in order."""
    found: list[tuple[int, int, str]] = []
    for pattern, name in RULES:
        for match in re.finditer(pattern, text, re.DOTALL):
            if match.end() > match.start():
                found.append((match.start(), match.end(), name))
    # Collapsing newline runs is expressed as a deletion of the surplus rather
    # than a re.sub, so that every offset shift lands in one map and spans
    # remap by arithmetic instead of by searching for the quote again.
    for match in NEWLINE_RUN.finditer(text):
        found.append((match.start() + 2, match.end(), "newline run"))
    # Sort real rules ahead of newline runs at the same offset so a merged
    # span keeps the name of the rule that actually motivated it.
    found.sort(key=lambda span: (span[0], span[2] == "newline run"))
    merged: list[tuple[int, int, str]] = []
    for start, end, name in found:
        if merged and start < merged[-1][1]:
            previous = merged[-1]
            keep = previous[2] if previous[2] != "newline run" else name
            merged[-1] = (previous[0], max(previous[1], end), keep)
        else:
            merged.append((start, end, name))
    return merged


def clean(text: str) -> tuple[str, list[int], list[int], list[tuple[int, int, str]]]:
    """Cleaned text plus the offset map (deletion starts, cumulative shift)."""
    cut = deletions(text)
    pieces: list[str] = []
    cuts: list[int] = []
    shift: list[int] = []
    at = removed = 0
    for start, end, _ in cut:
        pieces.append(text[at:start])
        removed += end - start
        cuts.append(start)
        shift.append(removed)
        at = end
    pieces.append(text[at:])
    return "".join(pieces), cuts, shift, cut


def remap(offset: int, cuts: list[int], shift: list[int]) -> int:
    """Move an original offset into the cleaned string."""
    index = bisect.bisect_right(cuts, offset) - 1
    return offset if index < 0 else offset - shift[index]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def selfcheck() -> None:
    """The offset map has to survive a deletion before and after a span."""
    text = ("Watch the mayor explain it »\n\n\n\nThe mayor resigned Tuesday.\n"
            "\n\n\nE-mail to a friend")
    quote = "resigned Tuesday"
    start = text.index(quote)
    body, cuts, shift, cut = clean(text)
    moved = remap(start, cuts, shift)
    assert body[moved:moved + len(quote)] == quote, body[moved:moved + 20]
    assert "Watch" not in body and "E-mail" not in body, body
    assert "\n\n\n" not in body, repr(body)
    names = {span[2] for span in cut}
    assert "video / gallery teaser line" in names, names
    print("selfcheck ok:", repr(body))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply", action="store_true",
        help="write the cleaned tree; otherwise report only",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the offset-remap check and exit")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0

    hits: dict[str, int] = {}
    chars_removed = touched = 0
    cleaned_articles: dict[str, list[dict]] = {}
    maps: dict[str, tuple[list[int], list[int]]] = {}

    for role in ("evaluation", "distractor"):
        rows = read_jsonl(STAGING / "corpus" / f"{role}_articles.jsonl")
        out_rows = []
        for row in rows:
            text = row["context"]
            body, cuts, shift, cut = clean(text)
            real = [span for span in cut if span[2] != "newline run"]
            if real:
                touched += 1
                chars_removed += sum(end - start for start, end, _ in real)
                for name in {span[2] for span in real}:
                    hits[name] = hits.get(name, 0) + 1
            new = dict(row)
            new["context"] = body
            out_rows.append(new)
            maps[row["article_id"]] = (cuts, shift)
        cleaned_articles[role] = out_rows

    total = sum(len(rows) for rows in cleaned_articles.values())
    print("=" * 72)
    print("CORPUS CLEANING - what the rules remove")
    print("=" * 72)
    for name, count in sorted(hits.items(), key=lambda kv: -kv[1]):
        print(f"  {name:34s} {count:6,d} articles  ({count / total:5.1%})")
    print(f"\n  articles changed  : {touched:,} of {total:,}  ({touched / total:.1%})")
    print(f"  characters removed: {chars_removed:,}")
    print("  newline runs of 3+ collapsed to one blank line, in every article")

    print()
    print("=" * 72)
    print("EVIDENCE SPANS - does the cleaning break the ground truth?")
    print("=" * 72)
    body_of = {
        row["article_id"]: row["context"]
        for rows in cleaned_articles.values() for row in rows
    }

    broken_total = 0
    cleaned_sets: dict[str, list[dict]] = {}
    for name in ("testset_original", "testset_reviewed_original",
                 "testset_resolved", "testset_clarified"):
        rows = read_jsonl(FINAL / f"{name}.jsonl")
        broken = exact = relocated = 0
        out_rows = []
        for row in rows:
            key = row.get("article_key")
            body = body_of.get(key)
            cuts, shift = maps.get(key, ([], []))
            spans: list[dict] = []
            ok = body is not None
            for span in row.get("evidence_spans") or []:
                quote = span["text"]
                start = remap(span["start"], cuts, shift)
                end = remap(span["end"], cuts, shift)
                if body[start:end] == quote:
                    exact += 1
                    spans.append({"start": start, "end": end, "text": quote})
                    continue
                # Should not happen now that every shift is in the map; keep
                # the search as a guard so a rule change cannot corrupt spans.
                found = body.find(quote, max(0, start - 400))
                if found < 0:
                    found = body.find(quote)
                if found < 0:
                    ok = False
                    break
                relocated += 1
                spans.append({"start": found, "end": found + len(quote),
                              "text": quote})
            new = dict(row)
            if ok:
                new["evidence_spans"] = spans
            else:
                new["evidence_spans_broken"] = True
                broken += 1
            out_rows.append(new)
        cleaned_sets[name] = out_rows
        broken_total += broken
        print(f"  {name:28s} {exact:5,d} exact  {relocated:5,d} relocated  "
              f"{broken:4,d} broken")

    print(f"\n  questions whose evidence no longer resolves: {broken_total:,}")
    if broken_total:
        print("  -> flagged with evidence_spans_broken, not deleted")

    if not args.apply:
        print(f"\n(dry run - nothing written. Pass --apply to write "
              f"{CLEANED.relative_to(PROJECT)})")
        return 0

    for role, rows in cleaned_articles.items():
        write_jsonl(CLEANED / "corpus" / f"{role}_articles.jsonl", rows)
    for name, rows in cleaned_sets.items():
        write_jsonl(CLEANED / f"{name}.jsonl", rows)
    (CLEANED / "cleaning_report.json").write_text(
        json.dumps({
            "rules": hits,
            "articles_changed": touched,
            "articles_total": total,
            "chars_removed": chars_removed,
            "broken_spans": broken_total,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {CLEANED.relative_to(PROJECT)}")
    print("The locked benchmark under final/ is unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
