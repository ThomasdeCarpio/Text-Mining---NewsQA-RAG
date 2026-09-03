"""How much text is missing from each article, measured against the original page.

Answers: for every benchmark article we can pair with its archived CNN page,
HOW MUCH is missing, and what does the distribution of that gap look like?

WHY THIS SCRIPT USES THE PROJECT'S OWN CLEANER
----------------------------------------------
An earlier version of this analysis (and 01p_full_coverage.py, now deleted)
used a hand-rolled extractor with this fallback chain:

    p.cnn_storypgraphtxt  ->  #cnnContentContainer p  ->  every <p> on the page

The first selector matches NOTHING in these archived pages (0 of 1,500 sampled -
it is a 2011 CNN template class these pages do not use), so extraction always
fell through to "every <p>". That swept in CNN's sign-up form, weather widget
and topic tags, which are fixed-length template blocks. The result was 675
articles all reported as missing exactly 1,736 characters - not truncation, but
the same boilerplate counted every time.

So this script does not parse HTML itself. It calls NewsCleaner
(backend/newsqa_rag/ingestion/cleaner.py), the newspaper3k-based extractor the
project already uses to build data/processed/, which strips boilerplate properly.

Two passes, because the accurate cleaner costs ~41 ms/page and there are 92,579
pages in the archive:
  pass 1  a cheap regex scan finds which pages could match a benchmark article
  pass 2  NewsCleaner runs on only those pages and produces the authoritative
          text used for every number reported

Pass 1 only has to be a high-recall filter - pass 2 re-matches from scratch and
discards anything the cleaner does not confirm.

Three outcomes per pairing, and only one supports a truncation claim:

  intact      the cleaned page is no longer than ours - nothing was cut
  truncated   ours is an exact PREFIX of the cleaned page, so text is missing
              off the end and nothing else differs
  diverged    the cleaned page is longer but ours is not a prefix of it, so the
              two texts differ somewhere in the middle. Excluded from the
              truncation numbers rather than counted.

PERFORMANCE NOTE. Reading the 24,469 loose HTML files takes ~75 minutes here:
they are on an NVMe SSD, but every file open is scanned by real-time antivirus,
which drops throughput to 0.3 MB/s (178 ms/file cold vs 0.1 ms warm). Streaming
data/cnn_downloads.tgz is one file open instead of 24,469.
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
import tarfile
import time
from pathlib import Path

import common as C

sys.path.insert(0, str(C.PROJECT / "backend"))

ARCHIVE = C.PROJECT / "data" / "cnn_downloads.tgz"
FURNITURE = 40      # a gap this small is page chrome, not a sentence
KEY = 60            # characters of an article head used as its signature

PARA = re.compile(r"<p\b[^>]*>(.*?)</p>", re.S)
TAG = re.compile(r"<[^>]+>")


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def candidate_heads(html: str) -> list[str]:
    """Cheap signatures for pass 1 only - never used for any reported number.

    Returns the first KEY characters of the page text starting at each of the
    first several paragraphs, so a page whose article is preceded by navigation
    still produces the article's own head as one of its candidates.
    """
    parts = [norm(TAG.sub(" ", p)) for p in PARA.findall(html)]
    parts = [p for p in parts if p]
    heads = []
    for start in range(min(8, len(parts))):
        joined = " ".join(parts[start:start + 6])
        if len(joined) >= KEY:
            heads.append(joined[:KEY])
    return heads


def pages():
    """Yield (name, html) for every archived page."""
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".html"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            yield member.name, handle.read().decode("utf-8", "replace")


def describe(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n, "mean": round(st.fmean(ordered), 1), "median": ordered[n // 2],
        "p25": ordered[n // 4], "p75": ordered[3 * n // 4],
        "p90": ordered[int(0.90 * n)], "max": ordered[-1],
    }


def main() -> None:
    from newsqa_rag.ingestion.cleaner import NewsCleaner

    corpus = C.articles("evaluation") + C.articles("distractor")
    ours_of = {row["article_id"]: norm(row["context"]) for row in corpus}
    role_of = {row["article_id"]: row.get("role") for row in corpus}
    by_key: dict[str, str] = {}
    for row in corpus:
        by_key.setdefault(ours_of[row["article_id"]][:KEY], row["article_id"])

    print(f"benchmark articles : {len(corpus):,}")
    print(f"archive            : {ARCHIVE.name}\n")

    print("PASS 1 - cheap scan for pages that might match an article")
    started = time.perf_counter()
    candidates: dict[str, str] = {}     # tar member -> article_id (provisional)
    scanned = 0
    for name, html in pages():
        scanned += 1
        for head in candidate_heads(html):
            article_id = by_key.get(head)
            if article_id is not None:
                candidates[name] = article_id
                break
        if scanned % 20000 == 0:
            print(f"  {scanned:,} pages, {len(candidates):,} candidates", flush=True)
    print(f"  scanned {scanned:,} pages in {time.perf_counter()-started:.0f}s")
    print(f"  candidate pages: {len(candidates):,}"
          f"  covering {len(set(candidates.values())):,} distinct articles\n")

    print("PASS 2 - NewsCleaner on the candidates (authoritative text)")
    cleaner = NewsCleaner()
    started = time.perf_counter()
    best: dict[str, dict] = {}
    done = failed = rejected = 0
    for name, html in pages():
        if name not in candidates:
            continue
        done += 1
        try:
            text = norm(cleaner.clean_html_string(html)["text"])
        except Exception:
            failed += 1
            continue
        if not text:
            failed += 1
            continue
        # Re-match from scratch against the cleaner's own output. Pass 1 is a
        # filter, not evidence: anything it guessed wrong is dropped here.
        article_id = by_key.get(text[:KEY])
        if article_id is None:
            rejected += 1
            continue
        ours = ours_of[article_id]
        record = {
            "article_id": article_id, "role": role_of[article_id],
            "ours": len(ours), "theirs": len(text), "gap": len(text) - len(ours),
            "is_prefix": text.startswith(ours), "page": name.split("/")[-1],
        }
        prior = best.get(article_id)
        if prior is None or record["gap"] > prior["gap"]:
            best[article_id] = record
        if done % 2000 == 0:
            print(f"  {done:,}/{len(candidates):,} cleaned, "
                  f"{len(best):,} matched", flush=True)

    elapsed = time.perf_counter() - started
    matched = list(best.values())
    intact = [m for m in matched if m["gap"] <= 0]
    longer = [m for m in matched if m["gap"] > 0]
    truncated = [m for m in longer if m["is_prefix"]]
    diverged = [m for m in longer if not m["is_prefix"]]
    furniture = [m for m in truncated if m["gap"] <= FURNITURE]
    real = [m for m in truncated if m["gap"] > FURNITURE]

    total = len(corpus)
    print(f"\n  cleaned {done:,} pages in {elapsed/60:.1f} min"
          f"  ({failed} failed, {rejected} rejected on re-match)")

    print("\n" + "=" * 74)
    print("PAIRING OUTCOME")
    print("=" * 74)
    print(f"  matched to a benchmark article : {len(matched):,}"
          f"  ({len(matched)/total:.1%} of {total:,})")
    print(f"  unmatched, so unknown          : {total - len(matched):,}"
          f"  ({(total-len(matched))/total:.1%})")
    print(f"\n  of the {len(matched):,} matched articles:")
    for label, group in (("intact (page no longer than ours)", intact),
                         ("TRUNCATED (ours is an exact prefix)", truncated),
                         ("diverged (longer, but not a prefix)", diverged)):
        share = len(group) / len(matched) if matched else 0
        print(f"    {label:38s} {len(group):6,d}  ({share:5.1%})")
    print(f"      ...page furniture only (<={FURNITURE} chars): {len(furniture):,}")
    print(f"      ...real missing content                    : {len(real):,}")

    gaps = [m["gap"] for m in truncated]
    real_gaps = [m["gap"] for m in real]
    pct = [m["gap"] / m["theirs"] * 100 for m in truncated]
    real_pct = [m["gap"] / m["theirs"] * 100 for m in real]
    stats, pstats = describe(gaps), describe([int(p) for p in pct])
    rstats, rpstats = describe(real_gaps), describe([int(p) for p in real_pct])

    if real_gaps:
        print("\n" + "=" * 74)
        print("HOW MUCH TEXT IS MISSING?   (real content only, gap > 40 chars)")
        print("=" * 74)
        print(f"  {'':20s}{'mean':>10s}{'median':>9s}{'p75':>9s}{'p90':>9s}{'max':>9s}")
        print(f"  {'characters missing':20s}{rstats['mean']:>10,.0f}"
              f"{rstats['median']:>9,d}{rstats['p75']:>9,d}"
              f"{rstats['p90']:>9,d}{rstats['max']:>9,d}")
        print(f"  {'% of the article':20s}{rpstats['mean']:>9.1f}%"
              f"{rpstats['median']:>8d}%{rpstats['p75']:>8d}%"
              f"{rpstats['p90']:>8d}%{rpstats['max']:>8d}%")
        print(f"\n  total characters missing: {sum(real_gaps):,}")

        print("\n  how bad is the loss?")
        bands = [(41, 200, "a sentence      (41-200)"),
                 (200, 1000, "a few paragraphs(200-1k)"),
                 (1000, 3000, "a large section (1k-3k)"),
                 (3000, 10 ** 9, "most of the story (3k+)")]
        for lo, hi, label in bands:
            k = sum(1 for g in real_gaps if lo <= g < hi)
            print(f"    {label:26s} {k:6,d}  ({k/len(real_gaps):5.1%})")

    print("\n  by role:")
    for role in ("evaluation", "distractor"):
        group = [m for m in real if m["role"] == role]
        allrole = [m for m in matched if m["role"] == role]
        if not group or not allrole:
            continue
        gs = describe([m["gap"] for m in group])
        print(f"    {role:12s} {len(group):5,d} losing real content of "
              f"{len(allrole):6,d} matched  ({len(group)/len(allrole):5.1%})"
              f"   median {gs['median']:,} chars")

    C.save("08_truncation_gap", {
        "corpus": total, "matched": len(matched), "intact": len(intact),
        "truncated": len(truncated), "diverged": len(diverged),
        "furniture_only": len(furniture), "real_content": len(real),
        "gap_chars": stats, "gap_percent": pstats,
        "real_gap_chars": rstats, "real_gap_percent": rpstats,
        "total_chars_missing": sum(real_gaps),
        "pages_scanned": scanned, "pages_cleaned": done,
        "extractor": "newsqa_rag.ingestion.cleaner.NewsCleaner (newspaper3k)",
        "gaps_truncated": gaps,
        "gaps_percent": [round(p, 2) for p in pct],
        "gaps_real": real_gaps,
        "by_role": {role: [m["gap"] for m in real if m["role"] == role]
                    for role in ("evaluation", "distractor")},
    })
    # Sample kept so the extraction can be eyeballed rather than trusted.
    sample = sorted(real, key=lambda m: -m["gap"])[:40]
    Path(C.OUT / "08_samples.json").write_text(
        json.dumps(sample, indent=1), encoding="utf-8")
    print("\nSaved -> out/08_truncation_gap.json  and  out/08_samples.json")


if __name__ == "__main__":
    main()
