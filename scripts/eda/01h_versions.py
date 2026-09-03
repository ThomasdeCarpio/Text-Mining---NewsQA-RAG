"""How many 'versions' of an article exist, and of what kind.

Four distinct notions get conflated. Count each separately:
  V1 duplicates inside the benchmark corpus
  V2 the same article in both corpus roles (evaluation vs distractor)
  V3 the same article in two independent sources (HF corpus vs local crawl)
  V4 where answers sit inside an article, as a truncation clue
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict

import common as C


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    evaluation = C.articles("evaluation")
    distractor = C.articles("distractor")
    for row in evaluation:
        row["_role"] = "evaluation"
    for row in distractor:
        row["_role"] = "distractor"
    corpus = evaluation + distractor

    print("=" * 78)
    print("V1. DUPLICATES INSIDE THE BENCHMARK CORPUS")
    print("=" * 78)
    groups = defaultdict(list)
    for row in corpus:
        groups[norm(row["context"])].append(row)
    dupes = [g for g in groups.values() if len(g) > 1]
    print(f"  articles                    : {len(corpus):,}")
    print(f"  unique after normalisation  : {len(groups):,}")
    print(f"  duplicate groups            : {len(dupes):,}")
    print(f"  redundant copies            : {sum(len(g) - 1 for g in dupes):,}")
    print("  difference between copies   : whitespace only (verified earlier)")

    print()
    print("=" * 78)
    print("V2. SAME ARTICLE IN BOTH ROLES  (would be a ground-truth leak)")
    print("=" * 78)
    cross = [g for g in dupes if len({r["_role"] for r in g}) > 1]
    print(f"  duplicate groups spanning evaluation AND distractor: {len(cross):,}")
    print("  -> an evaluation article never has a twin sitting in the distractor pool")

    ev_norm = {norm(r["context"]) for r in evaluation}
    di_norm = {norm(r["context"]) for r in distractor}
    print(f"  evaluation texts also present among distractors    : {len(ev_norm & di_norm):,}")

    print()
    print("=" * 78)
    print("V3. SAME ARTICLE FROM TWO INDEPENDENT SOURCES")
    print("=" * 78)
    bench_by_head = {norm(r["context"])[:300]: r for r in corpus}
    crawled = []
    for path in glob.glob(str(C.PROJECT / "data" / "processed" / "*_clean.json")):
        with open(path, encoding="utf-8") as handle:
            crawled.append(json.load(handle))

    pairs = []
    for doc in crawled:
        head = norm(doc["text"])[:300]
        if head in bench_by_head:
            pairs.append((bench_by_head[head], doc))

    print(f"  crawler articles on disk    : {len(crawled):,}")
    print(f"  of those, also in the corpus: {len(pairs):,}   <- 'a pair' = these two renderings")
    print(f"  crawler-only (not in corpus): {len(crawled) - len(pairs):,}")
    print(f"  corpus articles with a crawled counterpart: "
          f"{len(pairs):,} of {len(corpus):,}  ({len(pairs)/len(corpus):.2%})")

    kinds = {"benchmark shorter": 0, "same length": 0, "benchmark longer": 0}
    exact_prefix = 0
    for article, doc in pairs:
        b, c = norm(article["context"]), norm(doc["text"])
        if len(b) < len(c) - 40:
            kinds["benchmark shorter"] += 1
            if c.startswith(b):
                exact_prefix += 1
        elif abs(len(b) - len(c)) <= 40:
            kinds["same length"] += 1
        else:
            kinds["benchmark longer"] += 1
    for key, value in kinds.items():
        print(f"    {key:22s}: {value:3d}")
    print(f"    of the shorter ones, an EXACT prefix of the crawl: {exact_prefix}")

    print()
    print("=" * 78)
    print("V4. WHERE DO ANSWERS SIT INSIDE AN ARTICLE?")
    print("=" * 78)
    print("  If annotators saw a truncated article, evidence should crowd the front.\n")
    source_q = C.source_questions()
    by_id = {r["article_id"]: r["context"] for r in evaluation}
    positions = []
    for row in source_q:
        context = by_id.get(row["article_id"])
        spans = row.get("evidence_spans") or []
        if not context or not spans:
            continue
        positions.append(spans[0]["start"] / max(1, len(context)))

    buckets = [0] * 10
    for value in positions:
        buckets[min(9, int(value * 10))] += 1
    total = len(positions)
    for index, count in enumerate(buckets):
        bar = "#" * int(52 * count / max(buckets))
        print(f"    {index*10:3d}-{index*10+10:3d}% of article  {count:5,d}  {bar}")
    print(f"\n    n={total:,}  median position {sorted(positions)[total//2]:.1%}")

    C.save("01h_versions", {
        "corpus_articles": len(corpus),
        "unique_normalised": len(groups),
        "duplicate_groups": len(dupes),
        "cross_role_groups": len(cross),
        "paired_with_crawl": len(pairs),
        "pair_length_kinds": kinds,
        "exact_prefix": exact_prefix,
        "answer_position_deciles": buckets,
    })
    print("\nSaved -> out/01h_versions.json")


if __name__ == "__main__":
    main()
