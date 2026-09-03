"""Write one side-by-side file per suspected-truncated article so they can be
opened and read directly, and print every on-disk location involved."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import common as C

OUTDIR = Path(__file__).resolve().parent / "inspect"


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    corpus = C.articles("evaluation") + C.articles("distractor")
    bench = {norm(r["context"])[:300]: r for r in corpus}

    found = []
    seen = set()
    for path in sorted(glob.glob(str(C.PROJECT / "data" / "processed" / "*_clean.json"))):
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        url = doc.get("metadata", {}).get("url", "")
        head = norm(doc["text"])[:300]
        article = bench.get(head)
        if not article:
            continue
        b, c = norm(article["context"]), norm(doc["text"])
        if not (len(b) < len(c) - 40 and c.startswith(b)):
            continue
        if article["article_id"] in seen:
            continue
        seen.add(article["article_id"])
        found.append((article, doc, path, url, len(c) - len(b)))

    found.sort(key=lambda x: -x[4])
    # Prefer the ones with a real URL, then the biggest cuts.
    found.sort(key=lambda x: (x[3] in ("", "Unknown URL"),))

    print(f"{len(found)} verified-truncated articles. Writing comparison files.\n")
    for index, (article, doc, path, url, missing) in enumerate(found[:6], 1):
        aid = article["article_id"]
        target = OUTDIR / f"{index:02d}_{aid}.txt"
        lines = [
            "=" * 78,
            f"TITLE        : {doc['metadata'].get('title')}",
            f"URL          : {url or 'Unknown URL'}",
            f"article_id   : {aid}",
            f"corpus role  : {article.get('role')}",
            f"benchmark len: {len(article['context']):,} chars",
            f"crawled len  : {len(doc['text']):,} chars",
            f"MISSING      : {missing:,} chars",
            "",
            "SOURCE FILES ON DISK",
            f"  crawled copy : {path}",
            f"  benchmark    : {C.STAGING / 'corpus' / (article.get('role', 'distractor') + '_articles.jsonl')}",
            f"                 (find the line where article_id == {aid})",
            "",
            "=" * 78,
            "BENCHMARK VERSION (what the corpus and every index actually contain)",
            "=" * 78,
            article["context"],
            "",
            "=" * 78,
            "CRAWLED VERSION (fetched from cnn.com by this repo's crawler)",
            "=" * 78,
            doc["text"],
            "",
            "=" * 78,
            "THE MISSING TAIL (present in the crawl, absent from the benchmark)",
            "=" * 78,
            doc["text"][len(article["context"]):],
        ]
        target.write_text("\n".join(lines), encoding="utf-8")
        print(f"  {index}. {doc['metadata'].get('title')[:58]}")
        print(f"     url     : {url or 'Unknown URL'}")
        print(f"     compare : {target}")
        print(f"     crawled : {path}")
        print(f"     corpus  : {C.STAGING / 'corpus' / (article.get('role') + '_articles.jsonl')}  [article_id {aid}]")
        print()

    print(f"All comparison files: {OUTDIR}")


if __name__ == "__main__":
    main()
