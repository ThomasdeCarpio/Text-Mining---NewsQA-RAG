"""A3 - are questions near-duplicates of each other?

Two questions that are lexically almost identical but point at different gold
chunks are a scoring hazard: no retriever can separate them, so both lose
points for a defect in the benchmark rather than in the system. Duplicates
pointing at the SAME chunk are merely redundant - they inflate the effective
weight of one article.

Blocking on shared rare terms keeps this at O(n * block) instead of O(n^2).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from itertools import combinations

import common as C

TOKEN = re.compile(r"[a-z0-9]+")
JACCARD = 0.70
STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "did", "do", "does", "that", "this",
    "it", "its", "his", "her", "their", "he", "she", "they", "s", "by", "with",
    "from", "as",
}


def tok(text: str) -> set[str]:
    return {w for w in TOKEN.findall((text or "").lower()) if w not in STOP}


def main() -> None:
    for label in ("original", "resolved"):
        rows = C.variant(label)
        print("=" * 76)
        print(f"A3. NEAR-DUPLICATE QUESTIONS - {label} wording")
        print("=" * 76)
        analyse(rows, label)
        print()


def analyse(rows: list[dict], label: str) -> None:
    n = len(rows)
    norm = {r["question_id"]: " ".join(TOKEN.findall(r["question"].lower())) for r in rows}
    gold = {r["question_id"]: frozenset(r.get("relevant_chunk_ids") or []) for r in rows}
    article = {r["question_id"]: r.get("article_key") for r in rows}
    text = {r["question_id"]: r["question"] for r in rows}

    # 1. exact duplicates after normalisation
    exact = defaultdict(list)
    for qid, key in norm.items():
        exact[key].append(qid)
    exact_groups = [g for g in exact.values() if len(g) > 1]
    exact_same_gold = sum(1 for g in exact_groups
                          if len({gold[q] for q in g}) == 1)
    exact_cross = sum(1 for g in exact_groups
                      if len({article[q] for q in g}) > 1)
    print(f"  questions                          : {n:,}")
    print(f"  exact duplicate groups (normalised): {len(exact_groups):,}")
    print(f"    ...all members share one gold set: {exact_same_gold:,}")
    print(f"    ...members DISAGREE on gold      : {len(exact_groups)-exact_same_gold:,}")
    print(f"    ...span DIFFERENT articles       : {exact_cross:,}"
          f"   <- unscoreable: identical text, different gold article")

    # 2. near duplicates, blocked on rare terms
    df = Counter()
    words = {qid: tok(text[qid]) for qid in norm}
    for w in words.values():
        df.update(w)
    idf = lambda w: math.log(n / (1 + df.get(w, 0)))

    blocks: dict[str, list[str]] = defaultdict(list)
    for qid, ws in words.items():
        rare = sorted(ws, key=lambda w: -idf(w))[:3]
        for w in rare:
            blocks[w].append(qid)

    seen: set[tuple[str, str]] = set()
    pairs = []
    for members in blocks.values():
        if len(members) > 300:      # a term that generic is not a useful block
            continue
        for a, b in combinations(sorted(members), 2):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            wa, wb = words[a], words[b]
            union = wa | wb
            if not union:
                continue
            j = len(wa & wb) / len(union)
            if j >= JACCARD:
                pairs.append((j, a, b))

    pairs.sort(key=lambda p: -p[0])
    conflict = [p for p in pairs if gold[p[1]] != gold[p[2]]]
    cross_article = [p for p in conflict if article[p[1]] != article[p[2]]]

    print(f"\n  pairs compared                     : {len(seen):,}  (rare-term blocking)")
    print(f"  near-duplicate pairs (jaccard>={JACCARD:.2f}): {len(pairs):,}")
    print(f"    same gold chunks - redundant     : {len(pairs)-len(conflict):,}")
    print(f"    DIFFERENT gold chunks            : {len(conflict):,}")
    print(f"      ...and different articles      : {len(cross_article):,}"
          f"   <- the real hazard")

    affected = {q for _, a, b in conflict for q in (a, b)}
    print(f"  questions touched by a conflicting near-duplicate: {len(affected):,}"
          f"  ({len(affected)/n:.1%})")

    if cross_article:
        print("\n  Worst cross-article collisions:")
        for j, a, b in cross_article[:5]:
            print(f"\n    jaccard {j:.2f}")
            print(f"      A: {text[a][:78]}")
            print(f"         -> article {article[a]}")
            print(f"      B: {text[b][:78]}")
            print(f"         -> article {article[b]}")

    C.save(f"06_near_duplicates_{label}", {
        "n": n,
        "exact_groups": len(exact_groups),
        "exact_gold_disagreement": len(exact_groups) - exact_same_gold,
        "exact_cross_article": exact_cross,
        "pairs_compared": len(seen),
        "near_dup_pairs": len(pairs),
        "conflicting_pairs": len(conflict),
        "cross_article_conflicts": len(cross_article),
        "questions_affected": len(affected),
        "affected_share": round(len(affected) / n, 4),
    })


if __name__ == "__main__":
    main()
