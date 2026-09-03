"""Exact restoration coverage over all 24,469 archived HTML files.

Read-only measurement. Writes nothing into the project - only a report under
out/. Answers: how many benchmark articles can be matched to their original
HTML, and how much text is actually recoverable.
"""

from __future__ import annotations

import glob
import json
import multiprocessing as mp
import time
from pathlib import Path

import common as C

HTML_DIR = C.PROJECT / "data" / "cnn_downloads" / "cnn" / "downloads"
WORKERS = 16


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def extract_one(path: str) -> tuple[str, str] | None:
    """Return (path, normalised body text) for one archived page."""
    from bs4 import BeautifulSoup

    try:
        html = Path(path).read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        parts = [p.get_text(" ", strip=True) for p in soup.select("p.cnn_storypgraphtxt")]
        if not parts:
            parts = [p.get_text(" ", strip=True)
                     for p in soup.select("#cnnContentContainer p, .cnn_storyarea p, .storytext p")]
        if not parts:
            parts = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        return path, norm(" ".join(x for x in parts if x))
    except Exception:
        return None


def main() -> None:
    corpus = C.articles("evaluation") + C.articles("distractor")
    index: dict[str, dict] = {}
    for row in corpus:
        index.setdefault(norm(row["context"])[:200], row)
    print(f"benchmark articles          : {len(corpus):,}")
    print(f"distinct 200-char prefixes  : {len(index):,}")

    files = sorted(glob.glob(str(HTML_DIR / "*.html")))
    print(f"archived HTML files         : {len(files):,}")
    print(f"workers                     : {WORKERS}\n", flush=True)

    started = time.perf_counter()
    matched: dict[str, dict] = {}
    parsed = failed = 0

    with mp.Pool(WORKERS) as pool:
        for result in pool.imap_unordered(extract_one, files, chunksize=32):
            if result is None:
                failed += 1
                continue
            parsed += 1
            path, body = result
            row = index.get(body[:200])
            if row is None:
                continue
            aid = row["article_id"]
            nb = norm(row["context"])
            gain = len(body) - len(nb)
            is_prefix = body.startswith(nb)
            prior = matched.get(aid)
            # Keep the copy that recovers the most text.
            if prior is None or gain > prior["gain"]:
                matched[aid] = {
                    "article_id": aid,
                    "role": row.get("role"),
                    "html": Path(path).name,
                    "benchmark_chars": len(nb),
                    "html_chars": len(body),
                    "gain": gain,
                    "is_prefix": is_prefix,
                }
            if parsed % 4000 == 0:
                rate = (time.perf_counter() - started) / parsed
                print(f"  {parsed:,}/{len(files):,} parsed  {len(matched):,} matched  "
                      f"eta {rate*(len(files)-parsed)/60:.1f} min", flush=True)

    elapsed = time.perf_counter() - started
    restorable = [m for m in matched.values() if m["is_prefix"] and m["gain"] > 40]
    furniture = [m for m in matched.values() if m["is_prefix"] and 0 < m["gain"] <= 40]
    content = [m for m in restorable if m["gain"] >= 200]
    by_role: dict[str, int] = {}
    for m in content:
        by_role[m["role"]] = by_role.get(m["role"], 0) + 1

    print("\n" + "=" * 74)
    print("FULL COVERAGE RESULT")
    print("=" * 74)
    print(f"  parsed {parsed:,} files in {elapsed/60:.1f} min   (failed {failed})")
    print(f"  benchmark articles matched to their HTML : {len(matched):,}"
          f"  ({len(matched)/len(corpus):.1%} of {len(corpus):,})")
    print(f"  of those, HTML is a strict prefix-extension: {len(restorable):,}")
    print(f"    trivial gains (<=40 chars, page furniture): {len(furniture):,}")
    print(f"    REAL content recoverable (>=200 chars)    : {len(content):,}")
    print(f"      by role: {by_role}")
    if content:
        gains = sorted(m["gain"] for m in content)
        print(f"    characters recoverable: median {gains[len(gains)//2]:,}  "
              f"max {gains[-1]:,}  total {sum(gains):,}")

    C.save("01p_full_coverage", {
        "corpus": len(corpus),
        "html_files": len(files),
        "parsed": parsed,
        "matched": len(matched),
        "restorable": len(restorable),
        "furniture_only": len(furniture),
        "real_content": len(content),
        "content_by_role": by_role,
        "minutes": round(elapsed / 60, 1),
        "detail": sorted(content, key=lambda m: -m["gain"]),
    })
    print("\nSaved -> out/01p_full_coverage.json")


if __name__ == "__main__":
    main()
