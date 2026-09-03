"""Fetch the suspected-truncated articles from cnn.com and check whether the
text missing from the benchmark actually exists in the published article.

Third independent source: benchmark (HuggingFace) vs local crawl vs live page.
"""

from __future__ import annotations

import glob
import json
import re
import time
import ssl
import urllib.request

import common as C

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    with urllib.request.urlopen(request, timeout=30, context=ctx) as response:
        return response.read().decode("utf-8", errors="replace")


def extract(html: str) -> str:
    """Pull the article body out of a CNN page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # CNN's modern template wraps body paragraphs in .paragraph / [data-component-name=paragraph]
    parts = [
        p.get_text(" ", strip=True)
        for p in soup.select("p.paragraph, div.paragraph, p[data-component-name='paragraph'], .article__content p")
    ]
    if not parts:
        parts = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return " ".join(x for x in parts if x)


def main() -> None:
    corpus = C.articles("evaluation") + C.articles("distractor")
    bench = {norm(r["context"])[:300]: r for r in corpus}

    targets = []
    seen = set()
    for path in sorted(glob.glob(str(C.PROJECT / "data" / "processed" / "*_clean.json"))):
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        url = doc.get("metadata", {}).get("url", "")
        if not url or url == "Unknown URL":
            continue
        article = bench.get(norm(doc["text"])[:300])
        if not article or article["article_id"] in seen:
            continue
        b, c = norm(article["context"]), norm(doc["text"])
        if len(b) < len(c) - 40 and c.startswith(b):
            seen.add(article["article_id"])
            targets.append((article, doc, url))

    print(f"Fetching {len(targets)} suspected-truncated articles from cnn.com\n")
    results = []
    for index, (article, doc, url) in enumerate(targets, 1):
        print("=" * 78)
        print(f"{index}. {doc['metadata'].get('title')}")
        print(f"   {url}")
        try:
            live = extract(fetch(url.replace("http://", "https://")))
        except Exception as exc:
            print(f"   FETCH FAILED: {type(exc).__name__}: {exc}\n")
            continue

        nb, nc, nl = norm(article["context"]), norm(doc["text"]), norm(live)
        print(f"   benchmark {len(nb):,} | local crawl {len(nc):,} | LIVE PAGE {len(nl):,} chars")

        # The 120 characters the benchmark ends on, and the 120 that follow it locally.
        tail_of_bench = nb[-120:]
        missing_head = nc[len(nb):len(nb) + 120]

        in_live_bench_tail = tail_of_bench in nl
        in_live_missing = bool(missing_head) and missing_head[:80] in nl

        print(f"   benchmark's final 120 chars found on live page : {in_live_bench_tail}")
        print(f"   the MISSING text found on live page            : {in_live_missing}")
        if missing_head:
            print(f"     missing text tested: {missing_head[:100]!r}")
        verdict = (
            "TRUNCATION CONFIRMED by a third source"
            if in_live_bench_tail and in_live_missing
            else "inconclusive - live page differs from both local copies"
        )
        print(f"   -> {verdict}\n")
        results.append({
            "url": url,
            "article_id": article["article_id"],
            "benchmark_chars": len(nb),
            "local_crawl_chars": len(nc),
            "live_chars": len(nl),
            "bench_tail_on_live": in_live_bench_tail,
            "missing_text_on_live": in_live_missing,
            "verdict": verdict,
        })
        time.sleep(2)

    C.save("01k_live_verify", {"checked": len(results), "results": results})
    print(f"Saved -> out/01k_live_verify.json")


if __name__ == "__main__":
    main()
