"""Chase the two surprises from step 1: the clarified subset failure and the
225 normalised-duplicate article contexts."""

from __future__ import annotations

from collections import defaultdict

import common as C


def main() -> None:
    payload = {}

    print("=" * 78)
    print("A. WHY IS clarified NOT A SUBSET OF resolved?")
    print("=" * 78)
    variants = {name: C.variant(name) for name in C.VARIANTS}
    ids = {name: {r["question_id"] for r in rows} for name, rows in variants.items()}

    stray = ids["clarified"] - ids["resolved"]
    print(f"  clarified rows            : {len(ids['clarified']):,}")
    print(f"  resolved rows             : {len(ids['resolved']):,}")
    print(f"  clarified ids NOT in resolved: {len(stray):,}")

    if stray:
        by_id = {r["question_id"]: r for r in variants["clarified"]}
        src = {r["question_id"]: r for r in variants["resolved"]}
        print("\n  sample stray ids and whether their source_question_id is in resolved:")
        for qid in list(stray)[:5]:
            row = by_id[qid]
            parent = row.get("source_question_id")
            print(f"    {qid}  source={parent}  parent_in_resolved={parent in src}")
    payload["clarified_stray"] = len(stray)

    print()
    print("=" * 78)
    print("B. NEAR-DUPLICATE ARTICLES IN THE CORPUS")
    print("=" * 78)
    evaluation = C.articles("evaluation")
    distractor = C.articles("distractor")
    for row in evaluation:
        row["_role"] = "evaluation"
    for row in distractor:
        row["_role"] = "distractor"

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in evaluation + distractor:
        key = " ".join(row["context"].lower().split())
        groups[key].append(row)

    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    extra = sum(len(v) - 1 for v in dupes.values())
    print(f"  duplicate groups          : {len(dupes):,}")
    print(f"  redundant articles        : {extra:,}")

    mixed = [v for v in dupes.values() if len({r['_role'] for r in v}) > 1]
    print(f"  groups spanning BOTH roles: {len(mixed):,}   <-- distractor duplicates an evaluation article")

    sizes = {}
    for group in dupes.values():
        sizes[len(group)] = sizes.get(len(group), 0) + 1
    print(f"  group size distribution   : {dict(sorted(sizes.items()))}")

    if mixed:
        print("\n  Evaluation articles that have a duplicate sitting in the distractor pool:")
        for group in mixed[:6]:
            ev = [r for r in group if r["_role"] == "evaluation"]
            di = [r for r in group if r["_role"] == "distractor"]
            head = " ".join(group[0]["context"].split())[:72]
            print(f"    eval={len(ev)} distractor={len(di)}  |  {head}...")

    affected_q = 0
    if mixed:
        dup_eval_ids = {r["article_id"] for g in mixed for r in g if r["_role"] == "evaluation"}
        affected_q = sum(
            1 for r in variants["resolved"] if r.get("article_key") in dup_eval_ids
        )
        print(f"\n  resolved questions whose gold article has a corpus twin: {affected_q:,}")
    payload["duplicate_groups"] = len(dupes)
    payload["redundant_articles"] = extra
    payload["cross_role_groups"] = len(mixed)
    payload["questions_with_twin_article"] = affected_q

    C.save("01b_followup", payload)
    print("\nSaved -> out/01b_followup.json")


if __name__ == "__main__":
    main()
