"""Emit truncated articles with their source URL so a human can verify them
against the live page."""

from __future__ import annotations

import glob
import json

import common as C


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    corpus = C.articles("evaluation") + C.articles("distractor")
    bench = {norm(r["context"])[:300]: r for r in corpus}

    pairs = []
    for path in glob.glob(str(C.PROJECT / "data" / "processed" / "*_clean.json")):
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        head = norm(doc["text"])[:300]
        if head in bench:
            pairs.append((bench[head], doc, path))

    truncated = []
    for article, doc, path in pairs:
        b, c = norm(article["context"]), norm(doc["text"])
        if len(b) < len(c) - 40 and c.startswith(b):
            truncated.append((article, doc, path, len(c) - len(b)))
    truncated.sort(key=lambda x: -x[3])

    print(f"{len(truncated)} verified truncated articles (exact prefix, 40+ chars missing).")
    print("Showing the 5 with the most missing content.\n")

    for index, (article, doc, path, missing) in enumerate(truncated[:5], 1):
        meta = doc.get("metadata", {})
        print("=" * 78)
        print(f"{index}.  {meta.get('title') or '(no title)'}")
        print("=" * 78)
        print(f"  URL            : {meta.get('url')}")
        print(f"  published      : {meta.get('publish_date') or 'unknown'}")
        print(f"  article_id     : {article['article_id']}")
        print(f"  corpus role    : {article['_role'] if '_role' in article else article.get('role')}")
        print(f"  local crawl    : {path.split(chr(92))[-1]}")
        print(f"  benchmark len  : {len(article['context']):,} chars")
        print(f"  crawled len    : {len(doc['text']):,} chars")
        print(f"  MISSING        : {missing:,} chars")
        print()
        print("  benchmark version ENDS with:")
        print(f"    ...{' '.join(article['context'].split())[-140:]!r}")
        print()
        print("  the live article CONTINUES:")
        shared = len(norm(article["context"]))
        print(f"    {norm(doc['text'])[shared:shared + 240]!r} ...")
        print()

    C.save("01i_inspect_truncated", {
        "verified_truncated": len(truncated),
        "samples": [
            {
                "url": d.get("metadata", {}).get("url"),
                "article_id": a["article_id"],
                "benchmark_chars": len(a["context"]),
                "crawled_chars": len(d["text"]),
                "missing_chars": m,
            }
            for a, d, _, m in truncated[:5]
        ],
    })
    print("Saved -> out/01i_inspect_truncated.json")


if __name__ == "__main__":
    main()
