"""Explain the clarified variant's ID scheme, and diagnose where the 225
near-duplicate distractor articles came from and what removing them would cost.
"""

from __future__ import annotations

import difflib
from collections import Counter, defaultdict

import common as C


def main() -> None:
    print("=" * 78)
    print("A. WHAT IS testset_clarified AND ITS ::clarified ID SUFFIX?")
    print("=" * 78)

    resolved = {r["question_id"]: r for r in C.variant("resolved")}
    reviewed = {r["question_id"]: r for r in C.variant("reviewed_original")}
    clarified = C.variant("clarified")

    print(f"  resolved rows : {len(resolved):,}")
    print(f"  clarified rows: {len(clarified):,}")

    same_as_resolved = 0
    same_as_reviewed = 0
    missing_parent = 0
    for row in clarified:
        parent = row.get("source_question_id")
        if parent not in resolved:
            missing_parent += 1
            continue
        if row["question"] == resolved[parent]["question"]:
            same_as_resolved += 1
        if row["question"] == reviewed[parent]["question"]:
            same_as_reviewed += 1

    print(f"\n  clarified question text identical to its resolved parent : {same_as_resolved:,}")
    print(f"  clarified question text identical to reviewed_original   : {same_as_reviewed:,}")
    print(f"  clarified rows whose parent is missing from resolved      : {missing_parent:,}")

    labels = Counter(r.get("standalone_label") for r in C.variant("resolved"))
    print(f"\n  standalone_label across the 1,336 resolved rows: {dict(labels)}")

    in_clarified = {r.get("source_question_id") for r in clarified}
    untouched = [q for q in resolved if q not in in_clarified]
    print(f"  resolved rows NOT represented in clarified: {len(untouched):,}")
    if untouched:
        sample = resolved[untouched[0]]
        print(f"    example  q: {sample['question']}")
        print(f"             label: {sample.get('standalone_label')}")

    print(f"\n  variant tag on the rows: clarified={clarified[0].get('question_variant')!r} "
          f"resolved={next(iter(resolved.values())).get('question_variant')!r}")

    print()
    print("=" * 78)
    print("B. WHERE DO THE 225 NEAR-DUPLICATE ARTICLES COME FROM?")
    print("=" * 78)

    distractor = C.articles("distractor")
    evaluation = C.articles("evaluation")

    exact = {a["context"] for a in distractor}
    norm = {" ".join(a["context"].lower().split()) for a in distractor}
    print(f"  distractor articles          : {len(distractor):,}")
    print(f"  unique by exact text         : {len(exact):,}")
    print(f"  unique after lower+whitespace: {len(norm):,}")
    print(f"  redundant                    : {len(distractor) - len(norm):,}")

    # Are the stored hashes consistent with what we recompute?
    stored_norm = {a["normalized_context_sha256"] for a in distractor}
    print(f"  unique normalized_context_sha256 as stored: {len(stored_norm):,}"
          f"   <- pipeline already knew")

    groups: dict[str, list[dict]] = defaultdict(list)
    for a in distractor:
        groups[" ".join(a["context"].lower().split())].append(a)
    dupes = [g for g in groups.values() if len(g) > 1]

    print(f"\n  duplicate groups: {len(dupes):,}")
    print("\n  What actually differs inside a duplicate pair?")
    kinds = Counter()
    for group in dupes:
        a, b = group[0]["context"], group[1]["context"]
        if a == b:
            kinds["identical (should not happen)"] += 1
        elif a.lower() == b.lower():
            kinds["case only"] += 1
        elif " ".join(a.split()) == " ".join(b.split()):
            kinds["whitespace only"] += 1
        else:
            kinds["case + whitespace"] += 1
    for kind, count in kinds.most_common():
        print(f"    {kind:32s} {count:5,d}")

    print("\n  A concrete pair (first 2 differing fragments):")
    for group in dupes:
        a, b = group[0]["context"], group[1]["context"]
        if a != b:
            diff = [
                d for d in difflib.unified_diff(
                    a.splitlines(), b.splitlines(), lineterm="", n=0
                ) if d.startswith(("+", "-")) and not d.startswith(("+++", "---"))
            ]
            print(f"    ids: {group[0]['article_id']}  vs  {group[1]['article_id']}")
            for line in diff[:2]:
                print(f"      {line[:96]!r}")
            break

    print()
    print("  COST OF REMOVING THEM")
    chunk_rows = C.chunks()
    dup_ids = {a["article_id"] for g in dupes for a in g[1:]}
    doomed = [c for c in chunk_rows if c["metadata"]["article_id"] in dup_ids]
    gold = {cid for r in C.variant("resolved") for cid in (r.get("relevant_chunk_ids") or [])}
    print(f"    articles removed        : {len(dup_ids):,}")
    print(f"    chunks removed          : {len(doomed):,}  of {len(chunk_rows):,} "
          f"({len(doomed)/len(chunk_rows):.1%})")
    print(f"    gold chunks affected    : {len(gold & {c['id'] for c in doomed}):,}")
    print(f"    evaluation articles hit : "
          f"{len(dup_ids & {a['article_id'] for a in evaluation}):,}")

    C.save("01c_clarified_and_dupes", {
        "clarified_rows": len(clarified),
        "clarified_matches_resolved_text": same_as_resolved,
        "resolved_not_in_clarified": len(untouched),
        "duplicate_groups": len(dupes),
        "redundant_articles": len(dup_ids),
        "difference_kinds": dict(kinds),
        "chunks_removed_if_deduped": len(doomed),
        "gold_chunks_affected": len(gold & {c["id"] for c in doomed}),
    })
    print("\nSaved -> out/01c_clarified_and_dupes.json")


if __name__ == "__main__":
    main()
