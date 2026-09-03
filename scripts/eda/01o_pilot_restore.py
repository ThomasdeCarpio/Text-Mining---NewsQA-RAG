"""PILOT: can the archived CNN HTML restore the truncated benchmark text?

Runs on a small sample first so the full-corpus cost can be estimated before
committing to it. Reports match rate, recoverable characters, and timing.
"""

from __future__ import annotations

import glob
import random
import re
import sys
import time
from pathlib import Path

import common as C

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 300
HTML_DIR = C.PROJECT / "data" / "cnn_downloads" / "cnn" / "downloads"


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def extract(html: str) -> str:
    """Pull the story body from the 2011-era CNN template."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # 2011 CNN wrapped body paragraphs in .cnn_storypgraphtxt
    parts = [p.get_text(" ", strip=True) for p in soup.select("p.cnn_storypgraphtxt")]
    if not parts:
        parts = [p.get_text(" ", strip=True)
                 for p in soup.select("#cnnContentContainer p, .cnn_storyarea p, .storytext p")]
    if not parts:
        parts = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return " ".join(x for x in parts if x)


def main() -> None:
    corpus = C.articles("evaluation") + C.articles("distractor")
    # Index the benchmark by a normalised prefix so a truncated copy still matches.
    index = {}
    for row in corpus:
        index[norm(row["context"])[:200]] = row
    print(f"benchmark articles indexed by 200-char prefix: {len(index):,}")

    files = sorted(glob.glob(str(HTML_DIR / "*.html")))
    print(f"archived HTML files available               : {len(files):,}")
    sample = random.Random(42).sample(files, min(SAMPLE, len(files)))
    print(f"pilot sample                                : {len(sample):,}\n")

    started = time.perf_counter()
    matched = longer = shorter_or_equal = failed = 0
    recovered = []
    for path in sample:
        try:
            html = Path(path).read_text(encoding="utf-8", errors="replace")
            body = extract(html)
        except Exception:
            failed += 1
            continue
        key = norm(body)[:200]
        row = index.get(key)
        if not row:
            continue
        matched += 1
        nb, nh = norm(row["context"]), norm(body)
        if len(nh) > len(nb) + 40 and nh.startswith(nb):
            longer += 1
            recovered.append(len(nh) - len(nb))
        else:
            shorter_or_equal += 1

    elapsed = time.perf_counter() - started
    print("=" * 74)
    print("PILOT RESULT")
    print("=" * 74)
    print(f"  parsed                    : {len(sample) - failed:,}   failed: {failed}")
    print(f"  matched a benchmark article: {matched:,}  ({matched/len(sample):.1%} of sample)")
    print(f"    of those, HTML is LONGER (restorable): {longer:,}")
    print(f"    of those, same or shorter            : {shorter_or_equal:,}")
    if recovered:
        recovered.sort()
        print(f"  characters recoverable: median {recovered[len(recovered)//2]:,}  "
              f"max {recovered[-1]:,}  total {sum(recovered):,}")

    rate = elapsed / len(sample)
    print(f"\n  timing: {elapsed:.1f}s for {len(sample):,} files  ({rate*1000:.0f} ms/file)")
    print(f"  FULL RUN ESTIMATE over {len(glob.glob(str(HTML_DIR / '*.html'))):,} files: "
          f"{rate*len(files)/60:.1f} minutes")
    if matched:
        print(f"  projected benchmark articles matchable: "
              f"~{int(matched/len(sample)*len(files)):,} of {len(corpus):,}")

    C.save("01o_pilot_restore", {
        "sample": len(sample), "matched": matched, "longer": longer,
        "median_recoverable": recovered[len(recovered)//2] if recovered else 0,
        "seconds_per_file": rate,
    })


if __name__ == "__main__":
    main()
