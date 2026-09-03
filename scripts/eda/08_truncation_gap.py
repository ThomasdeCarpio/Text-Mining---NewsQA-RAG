"""How much text is missing from each article, measured against the original page.

01p answered "how many articles could we restore". This answers the sharper
question: for every article we can pair with its archived CNN page, HOW MUCH is
missing, and what does the distribution of that gap look like?

Three outcomes per pairing, and only one of them supports a truncation claim:

  intact        the archived page is no longer than ours - nothing was cut
  truncated     ours is an exact PREFIX of the archived page, which is missing
                text and nothing else
  diverged      the archived page is longer but ours is not a prefix of it, so
                the two texts differ somewhere in the middle. That is an
                extraction difference, not evidence of truncation, and it is
                excluded from the truncation numbers rather than counted.

Saves the full per-article gap so the figure script can plot the distribution
instead of quoting a single average.
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import statistics as st
import time
from pathlib import Path

import common as C

HTML_DIR = C.PROJECT / "data" / "cnn_downloads" / "cnn" / "downloads"
WORKERS = 16
FURNITURE = 40   # a gap this small is page chrome, not a sentence


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def extract_one(path: str) -> tuple[str, str] | None:
    from bs4 import BeautifulSoup

    try:
        html = Path(path).read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        parts = [p.get_text(" ", strip=True) for p in soup.select("p.cnn_storypgraphtxt")]
        if not parts:
            parts = [p.get_text(" ", strip=True) for p in soup.select(
                "#cnnContentContainer p, .cnn_storyarea p, .storytext p")]
        if not parts:
            parts = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        return path, norm(" ".join(x for x in parts if x))
    except Exception:
        return None


def describe(values: list[int], name: str) -> dict:
    if not values:
        return {"name": name, "n": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "name": name, "n": n,
        "mean": round(st.fmean(ordered), 1),
        "median": ordered[n // 2],
        "p25": ordered[n // 4], "p75": ordered[3 * n // 4],
        "p90": ordered[int(0.90 * n)], "max": ordered[-1],
    }


def main() -> None:
    corpus = C.articles("evaluation") + C.articles("distractor")
    index: dict[str, dict] = {}
    for row in corpus:
        index.setdefault(norm(row["context"])[:200], row)

    files = sorted(glob.glob(str(HTML_DIR / "*.html")))
    print(f"benchmark articles : {len(corpus):,}")
    print(f"archived pages     : {len(files):,}")
    print(f"workers            : {WORKERS}\n", flush=True)

    started = time.perf_counter()
    best: dict[str, dict] = {}
    parsed = 0
    with mp.Pool(WORKERS) as pool:
        for result in pool.imap_unordered(extract_one, files, chunksize=32):
            if result is None:
                continue
            parsed += 1
            path, body = result
            row = index.get(body[:200])
            if row is None:
                continue
            ours = norm(row["context"])
            gap = len(body) - len(ours)
            record = {
                "article_id": row["article_id"],
                "role": row.get("role"),
                "ours": len(ours),
                "theirs": len(body),
                "gap": gap,
                "is_prefix": body.startswith(ours),
            }
            # Several archived copies can match one article; keep the fullest.
            prior = best.get(row["article_id"])
            if prior is None or gap > prior["gap"]:
                best[row["article_id"]] = record
            if parsed % 6000 == 0:
                print(f"  {parsed:,}/{len(files):,} parsed, "
                      f"{len(best):,} matched", flush=True)

    elapsed = time.perf_counter() - started
    matched = list(best.values())

    intact = [m for m in matched if m["gap"] <= 0]
    longer = [m for m in matched if m["gap"] > 0]
    truncated = [m for m in longer if m["is_prefix"]]
    diverged = [m for m in longer if not m["is_prefix"]]
    furniture = [m for m in truncated if m["gap"] <= FURNITURE]
    real = [m for m in truncated if m["gap"] > FURNITURE]

    print("\n" + "=" * 74)
    print("PAIRING OUTCOME")
    print("=" * 74)
    total = len(corpus)
    print(f"  parsed {parsed:,} pages in {elapsed/60:.1f} min")
    print(f"  matched to a benchmark article : {len(matched):,}"
          f"  ({len(matched)/total:.1%} of {total:,})")
    print(f"  unmatched, so unknown          : {total - len(matched):,}"
          f"  ({(total-len(matched))/total:.1%})")
    print()
    print(f"  of the {len(matched):,} matched articles:")
    for name, group in (("intact (page no longer than ours)", intact),
                        ("TRUNCATED (ours is an exact prefix)", truncated),
                        ("diverged (longer but not a prefix)", diverged)):
        print(f"    {name:38s} {len(group):6,d}  ({len(group)/len(matched):5.1%})")
    print(f"      ...of the truncated, page furniture only (<={FURNITURE} chars): "
          f"{len(furniture):,}")
    print(f"      ...of the truncated, real missing content        : {len(real):,}")

    print()
    print("=" * 74)
    print("HOW MUCH TEXT IS MISSING?   (the truncated articles only)")
    print("=" * 74)
    gaps = [m["gap"] for m in truncated]
    stats = describe(gaps, "characters missing")
    print(f"  {'':22s}{'mean':>9s}{'median':>9s}{'p75':>9s}{'p90':>9s}{'max':>9s}")
    print(f"  {'characters missing':22s}{stats['mean']:>9,.0f}{stats['median']:>9,d}"
          f"{stats['p75']:>9,d}{stats['p90']:>9,d}{stats['max']:>9,d}")
    pct = [m["gap"] / m["theirs"] * 100 for m in truncated]
    pstats = describe([int(p) for p in pct], "percent missing")
    print(f"  {'% of the article':22s}{pstats['mean']:>8.1f}%{pstats['median']:>8d}%"
          f"{pstats['p75']:>8d}%{pstats['p90']:>8d}%{pstats['max']:>8d}%")
    print(f"\n  total characters missing across the corpus: {sum(gaps):,}")

    print("\n  by role:")
    for role in ("evaluation", "distractor"):
        group = [m for m in truncated if m["role"] == role]
        allrole = [m for m in matched if m["role"] == role]
        if not group:
            continue
        gs = describe([m["gap"] for m in group], role)
        print(f"    {role:12s} {len(group):5,d} truncated of {len(allrole):6,d} matched"
              f"  ({len(group)/len(allrole):5.1%})"
              f"   median gap {gs['median']:,} chars")

    print()
    print("=" * 74)
    print("READING")
    print("=" * 74)
    rate = len(truncated) / len(matched)
    real_rate = len(real) / len(matched)
    print(f"""
  Of every article we can pair with its original page, {rate:.1%} are an exact
  prefix of it - cut off, with nothing else changed. Excluding gaps that are
  only page furniture, {real_rate:.1%} are missing real content, and the typical
  loss is {stats['median']:,} characters ({pstats['median']}% of the article).

  The {len(diverged):,} diverged pairs are NOT counted as truncated. Our HTML
  extraction is not the one the benchmark used, so a mid-text difference is
  more likely to be an extraction difference than a cut. Counting them would
  inflate the truncation rate.""")

    C.save("08_truncation_gap", {
        "corpus": total,
        "matched": len(matched),
        "intact": len(intact),
        "truncated": len(truncated),
        "diverged": len(diverged),
        "furniture_only": len(furniture),
        "real_content": len(real),
        "gap_chars": stats,
        "gap_percent": pstats,
        "total_chars_missing": sum(gaps),
        "minutes": round(elapsed / 60, 1),
        # The full distributions, so the figure plots data rather than a mean.
        "gaps_truncated": gaps,
        "gaps_percent": [round(p, 2) for p in pct],
        "ours_truncated": [m["ours"] for m in truncated],
        "by_role": {
            role: [m["gap"] for m in truncated if m["role"] == role]
            for role in ("evaluation", "distractor")
        },
    })
    print("\nSaved -> out/08_truncation_gap.json")


if __name__ == "__main__":
    main()
