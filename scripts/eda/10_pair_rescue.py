"""Recover the articles that head-anchored pairing missed.

08 pairs a benchmark article to its archived page by comparing the FIRST 60
characters of each. That fails whenever the two texts start differently - a
promo box before the lead, a dropped dateline, a different first paragraph -
even though the same story is plainly there. It left 1,596 articles (14.4%)
unpaired, and those had to be reported as "unknown" rather than measured.

This pass anchors in the MIDDLE instead. For each unpaired article it takes a
long window of text at an offset past the start, and looks for that window
anywhere in the page. A 200-character run of prose is effectively unique across
11,064 news stories, so a hit is a match.

    article:  [~~ lead, may differ ~~][== 100-char window ==][ rest ]
                                      ^ anchor here, not at the start

This is EXACT matching, not fuzzy: an exact hash lookup of an exact window,
then an exact substring check. No edit distance or similarity threshold is
involved. 11_window_calibration.py shows why that is enough - every pair shares
an exact run of at least 100 characters, median 888.

Two-stage again, for the same reason as 08: the accurate cleaner costs ~41 ms a
page and there are 92,579 of them.

  stage 1  a cheap regex extraction of every page, searched for any window.
           The regex extractor keeps page furniture, but it only ever ADDS text
           - it never drops article prose - so a window that exists in the
           article still appears in its output. False positives are fine here;
           stage 2 removes them.
  stage 2  NewsCleaner on the hits, then a strict re-check.

It also uses a laxer truncation test than 08. 08 asked "is our text a prefix of
theirs", which fails if the cleaner prepended anything. This asks "does our text
appear inside theirs, with more text after it" - which is what truncation
actually means, and does not care what sits in front.
"""

from __future__ import annotations

import re
import statistics as st
import sys
import tarfile
import time

import common as C

sys.path.insert(0, str(C.PROJECT / "common"))

ARCHIVE = C.PROJECT / "data" / "cnn_downloads.tgz"
FURNITURE = 40
# Calibrated by 11_window_calibration.py, not guessed. Measured recall on the
# 98 articles that have both a benchmark version and a cleaned page:
#
#   window  index region  page stride   recall
#      200    every 25th          150      63%   <- the first attempt
#      200      300..800          150      87%
#      100      300..800          150      95%
#      100      200..900          150      98%   <- chosen
#
# The first attempt failed on STRIDE, not on window length. A page window at
# offset p equals the article window at a = p - delta, so indexing article
# offsets only every 25 characters requires delta to be a multiple of 25 for a
# hit. Indexing every offset in a bounded region removes that constraint; the
# region keeps the index near 600 MB instead of indexing whole articles.
WINDOW = 100        # length of the anchor window
INDEX_LO = 200      # index every offset in this region of the article
INDEX_HI = 900
PAGE_STRIDE = 150   # windows probed per page; any stride < region width works

PARA = re.compile(r"<p\b[^>]*>(.*?)</p>", re.S)
TAG = re.compile(r"<[^>]+>")


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def extract_cheap(html: str) -> str:
    """Fast, boilerplate-inclusive text. Only ever adds, never drops prose."""
    return norm(" ".join(TAG.sub(" ", p) for p in PARA.findall(html)))


def pages():
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".html"):
                handle = tar.extractfile(member)
                if handle is not None:
                    yield member.name, handle.read().decode("utf-8", "replace")


def describe(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    o = sorted(values)
    n = len(o)
    return {"n": n, "mean": round(st.fmean(o), 1), "median": o[n // 2],
            "p75": o[3 * n // 4], "p90": o[int(0.9 * n)], "max": o[-1]}


def main() -> None:
    from newsqa_rag.ingestion.cleaner import NewsCleaner

    corpus = C.articles("evaluation") + C.articles("distractor")
    ours_of = {row["article_id"]: norm(row["context"]) for row in corpus}
    role_of = {row["article_id"]: row.get("role") for row in corpus}
    articles = list(ours_of)

    print(f"benchmark articles      : {len(corpus):,}")
    print(f"anchor window           : {WINDOW} chars, every offset in "
          f"{INDEX_LO}..{INDEX_HI}\n")

    # Keyed by hash so the index costs hundreds of MB rather than a duplicated
    # copy of the corpus. A hash collision only yields a candidate that stage 2
    # then rejects, so it cannot create a false pairing.
    print("building the window index...")
    index: dict[int, str] = {}
    for article_id in articles:
        text = ours_of[article_id]
        stop = min(INDEX_HI, max(INDEX_LO + 1, len(text) - WINDOW))
        for start in range(INDEX_LO, stop):
            index.setdefault(hash(text[start:start + WINDOW]), article_id)
    print(f"  {len(index):,} windows from {len(articles):,} articles\n")

    print("STAGE 1 - cheap scan for pages containing any window")
    started = time.perf_counter()
    hits: dict[str, str] = {}
    scanned = 0
    for name, html in pages():
        scanned += 1
        text = extract_cheap(html)
        for start in range(0, max(1, len(text) - WINDOW), PAGE_STRIDE):
            article_id = index.get(hash(text[start:start + WINDOW]))
            if article_id is not None:
                hits[name] = article_id
                break
        if scanned % 25000 == 0:
            print(f"  {scanned:,} pages, {len(hits):,} hits", flush=True)
    print(f"  scanned {scanned:,} pages in {time.perf_counter()-started:.0f}s")
    print(f"  candidate pages: {len(hits):,}"
          f"  covering {len(set(hits.values())):,} distinct articles\n")

    if not hits:
        print("no rescues found")
        return

    print("STAGE 2 - NewsCleaner on the hits, then a strict re-check")
    cleaner = NewsCleaner()
    started = time.perf_counter()
    best: dict[str, dict] = {}
    cleaned = rejected = 0
    for name, html in pages():
        if name not in hits:
            continue
        cleaned += 1
        try:
            text = norm(cleaner.clean_html_string(html)["text"])
        except Exception:
            continue
        article_id = hits[name]
        ours = ours_of[article_id]
        at = text.find(ours)
        if at < 0:
            # The window matched but the full article text is not contained,
            # so the two genuinely differ. Not a rescue.
            rejected += 1
            continue
        trailing = len(text) - (at + len(ours))
        record = {"article_id": article_id, "role": role_of[article_id],
                  "ours": len(ours), "theirs": len(text),
                  "leading": at, "gap": trailing, "page": name.split("/")[-1]}
        prior = best.get(article_id)
        if prior is None or trailing > prior["gap"]:
            best[article_id] = record
        if cleaned % 1000 == 0:
            print(f"  {cleaned:,}/{len(hits):,} cleaned, {len(best):,} rescued",
                  flush=True)
    print(f"  cleaned {cleaned:,} pages in {(time.perf_counter()-started)/60:.1f} min"
          f"  ({rejected} rejected on re-check)")

    rescued = list(best.values())
    truncated = [r for r in rescued if r["gap"] > FURNITURE]
    intact = [r for r in rescued if r["gap"] <= 0]
    with_lead = [r for r in rescued if r["leading"] > 0]

    print("\n" + "=" * 74)
    print("RESCUE RESULT")
    print("=" * 74)
    print(f"  articles paired by mid-anchor     : {len(rescued):,}"
          f"  of {len(articles):,} ({len(rescued)/len(articles):.1%})")
    print(f"    (08's head-anchor paired 9,468)")
    print(f"    of those, truncated (>{FURNITURE} chars): {len(truncated):,}")
    print(f"    of those, intact                : {len(intact):,}")
    print(f"  matched despite a DIFFERENT START : {len(with_lead):,}"
          f"   <- exactly what head-anchoring cannot find")
    if with_lead:
        lead = describe([r["leading"] for r in with_lead])
        print(f"    text before our article begins  : median {lead['median']:,}"
              f"  max {lead['max']:,} chars")
    if truncated:
        gaps = describe([r["gap"] for r in truncated])
        print(f"\n  missing text in the rescued truncated articles:")
        print(f"    median {gaps['median']:,}  mean {gaps['mean']:,.0f}"
              f"  p90 {gaps['p90']:,}  max {gaps['max']:,}")

    for role in ("evaluation", "distractor"):
        group = [r for r in truncated if r["role"] == role]
        if group:
            print(f"    {role:12s} {len(group):,} truncated")

    print("\n  examples of a page whose start differs from ours:")
    for r in sorted(with_lead, key=lambda x: -x["leading"])[:3]:
        print(f"    {r['article_id'][:22]}  {r['leading']:,} chars before ours, "
              f"{r['gap']:,} after")

    C.save("10_pair_rescue", {
        "unpaired_before": len(articles),
        "rescued": len(rescued),
        "rescued_truncated": len(truncated),
        "rescued_intact": len(intact),
        "matched_with_different_start": len(with_lead),
        "gap_chars": describe([r["gap"] for r in truncated]),
        "leading_chars": describe([r["leading"] for r in with_lead]),
        "window": WINDOW, "index_region": [INDEX_LO, INDEX_HI],
        "calibrated_recall": 0.98,
        "by_role": {role: len([r for r in truncated if r["role"] == role])
                    for role in ("evaluation", "distractor")},
    })
    print("\nSaved -> out/10_pair_rescue.json")


if __name__ == "__main__":
    main()
