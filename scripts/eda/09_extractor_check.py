"""Did the broken extractor change the VERDICT, or only the size of the gap?

08 recomputes everything with the project's NewsCleaner. This script answers the
narrower question that matters for trusting the old conclusions: take the
articles the broken extractor called TRUNCATED and the ones it called INTACT,
re-extract both sets with the correct cleaner, and see how many change class.

The broken extractor is reproduced here on purpose. It is not used for any
reported number - it exists so the two can be run on the same pages and
compared. What it did wrong:

    it selected  p.cnn_storypgraphtxt  first, which matches NOTHING in these
    archived pages, then fell through to "every <p> on the page", picking up
    CNN's sign-up form, weather widget and topic tags as if they were article
    text. Those are fixed-length blocks, so identical gaps repeated across
    hundreds of articles (1,736 chars x 675 articles, 25 chars x 610, ...).

Stratified sample rather than the full corpus, because the correct cleaner
costs ~41 ms/page: SAMPLE_PER_CLASS from each old class is enough to show
whether the verdicts moved.
"""

from __future__ import annotations

import json
import random
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import common as C

sys.path.insert(0, str(C.PROJECT / "backend"))

ARCHIVE = C.PROJECT / "data" / "cnn_downloads.tgz"
FURNITURE = 40
KEY = 60
SAMPLE_PER_CLASS = 250

PARA = re.compile(r"<p\b[^>]*>(.*?)</p>", re.S)
STORY = re.compile(r'<p class="cnn_storypgraphtxt">(.*?)</p>', re.S)
TAG = re.compile(r"<[^>]+>")


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def extract_broken(html: str) -> str:
    """The deleted extractor, reproduced so the bug can be measured."""
    parts = STORY.findall(html)          # never matches these pages
    if not parts:
        parts = PARA.findall(html)       # <- the bug: every <p>, boilerplate too
    return norm(" ".join(TAG.sub(" ", p) for p in parts))


def classify(ours: str, theirs: str) -> str:
    if len(theirs) <= len(ours):
        return "intact"
    if not theirs.startswith(ours):
        return "diverged"
    return "furniture" if len(theirs) - len(ours) <= FURNITURE else "truncated"


def pages():
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".html"):
                handle = tar.extractfile(member)
                if handle is not None:
                    yield member.name, handle.read().decode("utf-8", "replace")


def main() -> None:
    from newsqa_rag.ingestion.cleaner import NewsCleaner

    corpus = C.articles("evaluation") + C.articles("distractor")
    ours_of = {row["article_id"]: norm(row["context"]) for row in corpus}
    by_key = {}
    for row in corpus:
        by_key.setdefault(ours_of[row["article_id"]][:KEY], row["article_id"])

    print("PASS 1 - classify every page with the BROKEN extractor")
    started = time.perf_counter()
    old: dict[str, dict] = {}
    scanned = 0
    for name, html in pages():
        scanned += 1
        text = extract_broken(html)
        article_id = by_key.get(text[:KEY])
        if article_id is None:
            continue
        ours = ours_of[article_id]
        gap = len(text) - len(ours)
        record = {"page": name, "article_id": article_id, "gap": gap,
                  "verdict": classify(ours, text)}
        prior = old.get(article_id)
        if prior is None or gap > prior["gap"]:
            old[article_id] = record
    print(f"  scanned {scanned:,} pages in {time.perf_counter()-started:.0f}s")
    print(f"  articles classified by the broken extractor: {len(old):,}")

    counts = Counter(r["verdict"] for r in old.values())
    print("  its verdicts: " + ", ".join(f"{k}={v:,}" for k, v in counts.most_common()))

    tell_tale = Counter(r["gap"] for r in old.values() if r["gap"] > 0)
    print("\n  repeated identical gaps - the fingerprint of boilerplate:")
    for value, k in tell_tale.most_common(5):
        print(f"    gap={value:6,d} chars   {k:5,d} articles")

    # Stratified sample: the two classes whose verdicts we need to re-check.
    rng = random.Random(42)
    wanted: dict[str, str] = {}
    for verdict in ("truncated", "intact"):
        group = [r for r in old.values() if r["verdict"] == verdict]
        for r in rng.sample(group, min(SAMPLE_PER_CLASS, len(group))):
            wanted[r["page"]] = r["article_id"]
    print(f"\nPASS 2 - re-extract {len(wanted):,} of those pages with NewsCleaner")

    cleaner = NewsCleaner()
    started = time.perf_counter()
    moved = Counter()
    examples = {"truncated->intact": [], "intact->truncated": []}
    checked = 0
    for name, html in pages():
        if name not in wanted:
            continue
        article_id = wanted[name]
        try:
            text = norm(cleaner.clean_html_string(html)["text"])
        except Exception:
            continue
        if not text:
            continue
        checked += 1
        ours = ours_of[article_id]
        new_verdict = classify(ours, text)
        old_verdict = old[article_id]["verdict"]
        moved[(old_verdict, new_verdict)] += 1
        key = f"{old_verdict}->{new_verdict}"
        if key in examples and len(examples[key]) < 3:
            examples[key].append({
                "article_id": article_id,
                "ours": len(ours),
                "broken_said": old[article_id]["gap"],
                "cleaner_says": len(text) - len(ours),
                "tail": text[len(ours):len(ours) + 220] if text.startswith(ours) else "",
            })
    print(f"  cleaned {checked:,} pages in {time.perf_counter()-started:.0f}s")

    print("\n" + "=" * 74)
    print("DID THE VERDICT CHANGE?   broken extractor  ->  correct cleaner")
    print("=" * 74)
    olds = sorted({k[0] for k in moved})
    news = sorted({k[1] for k in moved})
    print(f"  {'was':12s}" + "".join(f"{n:>12s}" for n in news) + f"{'total':>10s}")
    for o in olds:
        row = [moved.get((o, n), 0) for n in news]
        print(f"  {o:12s}" + "".join(f"{v:>12,d}" for v in row) + f"{sum(row):>10,d}")
    agree = sum(v for (o, n), v in moved.items() if o == n)
    print(f"\n  verdict unchanged: {agree:,} of {checked:,}  ({agree/max(1,checked):.1%})")

    for key, rows in examples.items():
        if not rows:
            continue
        print(f"\n  {key} - what the correct cleaner sees:")
        for r in rows:
            print(f"    {r['article_id'][:22]}  benchmark={r['ours']:,} chars"
                  f"   broken claimed +{r['broken_said']:,}"
                  f"   cleaner says {r['cleaner_says']:+,}")
            if r["tail"]:
                print(f"      genuinely missing text: {r['tail'][:150]!r}")

    C.save("09_extractor_check", {
        "articles_classified_by_broken": len(old),
        "broken_verdicts": dict(counts),
        "repeated_gaps": dict(tell_tale.most_common(10)),
        "sampled": checked,
        "confusion": {f"{o}->{n}": v for (o, n), v in moved.items()},
        "verdict_unchanged": agree,
        "examples": examples,
    })
    print("\nSaved -> out/09_extractor_check.json")


if __name__ == "__main__":
    main()
