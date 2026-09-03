"""A2 - what was actually wrong with the original questions?

The reason codes are the review's own diagnosis of why a question could not
stand alone. They are the evidence for the resolution decision: if the defects
are real defect classes rather than reviewer taste, resolving them is a
correction, not a convenience.

Also measures whether each defect class actually gained retrieval signal, by
counting rare terms added between original and resolved wording.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import common as C

RARE_IDF = 6.0
TOKEN = re.compile(r"[a-z0-9]+")
STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "did", "do", "does", "what", "who",
    "when", "where", "which", "how", "why", "whom", "whose", "that", "this",
    "it", "its", "his", "her", "their", "he", "she", "they", "s", "by", "with",
    "from", "as", "many", "much", "long", "old",
}


def tok(text: str) -> list[str]:
    return TOKEN.findall((text or "").lower())


def bar(count: int, total: int, width: int = 40) -> str:
    return "#" * max(0, int(width * count / total))


def main() -> None:
    annotations = C.annotations()
    n = len(annotations)

    print("=" * 76)
    print("A2. REASON CODES - the review's diagnosis of the original questions")
    print("=" * 76)
    print(f"  annotated questions: {n:,}\n")

    codes = Counter()
    per_question = Counter()
    for row in annotations:
        rc = row.get("reason_codes") or []
        codes.update(rc)
        per_question[len(rc)] += 1

    for name, count in codes.most_common():
        print(f"    {name:28s} {count:5,d}  {count/n:6.1%}  {bar(count, n)}")
    print(f"\n  codes per question: "
          + ", ".join(f"{k}:{v:,}" for k, v in sorted(per_question.items())))

    print()
    print("=" * 76)
    print("LABELS - proposed by the model vs decided by the human")
    print("=" * 76)
    proposed = Counter(r.get("proposed_label") for r in annotations)
    final = Counter(r.get("final_label") for r in annotations)
    decision = Counter(r.get("review_decision") for r in annotations)
    print("  proposed:")
    for k, v in proposed.most_common():
        print(f"    {str(k):28s} {v:5,d}  {v/n:6.1%}")
    print("  final:")
    for k, v in final.most_common():
        print(f"    {str(k):28s} {v:5,d}  {v/n:6.1%}")
    print("  review decision:")
    for k, v in decision.most_common():
        print(f"    {str(k):28s} {v:5,d}  {v/n:6.1%}")

    overturned = sum(1 for r in annotations if r.get("review_decision") != "approve")
    print(f"\n  human overturned or amended the model proposal: {overturned:,}"
          f"  ({overturned/n:.1%})")

    excluded = [r for r in annotations if r.get("excluded")]
    modified = [r for r in annotations if r.get("answer_modified")]
    print(f"  questions excluded outright                   : {len(excluded):,}")
    print(f"  answers modified during review                : {len(modified):,}"
          f"  ({len(modified)/n:.1%})")

    print()
    print("=" * 76)
    print("DID EACH DEFECT CLASS ACTUALLY GAIN RETRIEVAL SIGNAL?")
    print("=" * 76)
    print("  Rare terms added to the question, by reason code.\n")

    chunks = C.chunks()
    n_chunks = len(chunks)
    df = Counter()
    for chunk in chunks:
        df.update(set(tok(chunk["text"])))
    idf = lambda w: math.log(n_chunks / (1 + df.get(w, 0)))

    before = {r["question_id"]: r["question"] for r in C.variant("reviewed_original")}
    after = {r["question_id"]: r["question"] for r in C.variant("resolved")}
    codes_of = {r["question_id"]: (r.get("reason_codes") or []) for r in annotations}

    gained: dict[str, list[int]] = defaultdict(list)
    words_gained: dict[str, list[int]] = defaultdict(list)
    for qid, old in before.items():
        new = after.get(qid)
        if new is None:
            continue
        o = {w for w in tok(old) if w not in STOP}
        v = {w for w in tok(new) if w not in STOP}
        added = v - o
        rare_added = sum(1 for w in added if idf(w) >= RARE_IDF)
        for code in codes_of.get(qid) or ["(none)"]:
            gained[code].append(rare_added)
            words_gained[code].append(len(added))

    rows = sorted(gained.items(), key=lambda kv: -len(kv[1]))
    print(f"    {'reason code':28s} {'n':>6s} {'+words':>8s} {'+rare':>8s} {'%>=1 rare':>10s}")
    print(f"    {'-'*28} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
    payload_gain = {}
    for code, values in rows:
        m = len(values)
        rare_mean = sum(values) / m
        word_mean = sum(words_gained[code]) / m
        any_rare = sum(1 for x in values if x >= 1) / m
        print(f"    {code:28s} {m:6,d} {word_mean:8.2f} {rare_mean:8.2f} {any_rare:9.1%}")
        payload_gain[code] = {"n": m, "words_added": round(word_mean, 2),
                              "rare_added": round(rare_mean, 2),
                              "share_with_rare": round(any_rare, 4)}

    print()
    print("=" * 76)
    print("READING")
    print("=" * 76)
    top = codes.most_common(1)[0]
    print(f"""
  The dominant defect is '{top[0]}' at {top[1]/n:.0%} of annotated questions.
  Every code here names a property of the QUESTION TEXT that can be checked
  against the article - a missing subject, an unresolved pronoun, a dangling
  reference - not a judgement about difficulty. That is what makes resolution
  defensible: the review repaired a stated defect class, and the repair is
  auditable question by question.

  The rare-term column says how much retrieval signal each repair added. Codes
  that add rare terms are the ones that move the retrieval scores; codes that
  add none repaired readability without changing what a retriever can match.""")

    C.save("05_reason_codes", {
        "n": n,
        "reason_codes": dict(codes),
        "codes_per_question": dict(per_question),
        "proposed_label": {str(k): v for k, v in proposed.items()},
        "final_label": {str(k): v for k, v in final.items()},
        "review_decision": {str(k): v for k, v in decision.items()},
        "overturned": overturned,
        "excluded": len(excluded),
        "answer_modified": len(modified),
        "signal_gain_by_code": payload_gain,
    })
    print("\nSaved -> out/05_reason_codes.json")


if __name__ == "__main__":
    main()
