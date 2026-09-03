"""C2 - how hard is the retrieval task really, and can a distractor answer?

Two things at once:
  1. COMPETITION - how many non-gold chunks share the question's rare terms.
     Many competitors means first-stage retrieval cannot separate them alone,
     which is what justifies a reranker.
  2. UNLABELLED ANSWERS - how many non-gold chunks actually contain the gold
     answer string. This is the empirical test of whether open-corpus
     ambiguity is real, and it bounds the benchmark's false-negative rate.
"""

from __future__ import annotations

import math
import re
import statistics as st
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
    return TOKEN.findall(text.lower())


def norm_answer(text: str) -> str:
    return " ".join(TOKEN.findall((text or "").lower()))


def main() -> None:
    chunks = C.chunks()
    n = len(chunks)
    print(f"indexing {n:,} chunks...", flush=True)

    words_of: dict[str, set[str]] = {}
    df: Counter = Counter()
    for chunk in chunks:
        words = set(tok(chunk["text"]))
        words_of[chunk["id"]] = words
        df.update(words)

    idf = lambda w: math.log(n / (1 + df.get(w, 0)))

    # inverted index over rare terms only
    postings: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        for word in words_of[chunk["id"]]:
            if idf(word) >= RARE_IDF:
                postings[word].add(chunk["id"])

    # normalised text per chunk, for answer-substring search
    text_of = {c["id"]: norm_answer(c["text"]) for c in chunks}
    role_of = {c["id"]: c["metadata"].get("corpus_role") for c in chunks}
    article_of = {c["id"]: c["metadata"].get("canonical_article_id") for c in chunks}

    resolved = C.variant("resolved")

    print()
    print("=" * 74)
    print("1. COMPETITION - how many chunks share the question's rare terms?")
    print("=" * 74)
    competitors, no_rare = [], 0
    for row in resolved:
        gold = set(row.get("relevant_chunk_ids") or [])
        rare = [w for w in set(tok(row["question"]))
                if w not in STOP and idf(w) >= RARE_IDF and w in postings]
        if not rare:
            no_rare += 1
            continue
        # chunks containing ANY of the question's rare terms
        candidates: set[str] = set()
        for word in rare:
            candidates |= postings[word]
        competitors.append(len(candidates - gold))

    comp = sorted(competitors)
    m = len(comp)
    print(f"  questions with at least one rare term : {m:,} of {len(resolved):,}")
    print(f"  questions with NO rare term           : {no_rare:,}  "
          f"(lexical retrieval has no sharp handle on these)")
    print(f"\n  non-gold chunks sharing a rare term:")
    for label, value in (("median", comp[m // 2]), ("p75", comp[int(.75 * m)]),
                         ("p90", comp[int(.90 * m)]), ("max", comp[-1])):
        print(f"    {label:7s} {value:6,d}")
    tight = sum(1 for c in comp if c <= 10)
    print(f"  questions where rare terms alone narrow to <=10 competitors: "
          f"{tight:,}  ({tight/m:.1%})")

    print()
    print("=" * 74)
    print("2. UNLABELLED ANSWERS - can a NON-GOLD chunk answer the question?")
    print("=" * 74)
    print("  Searching every chunk that shares a rare term for the gold answer string.\n")
    hits_any, hits_distractor, checked = 0, 0, 0
    strength: Counter = Counter()
    examples = []
    words_of_chunk = words_of.__getitem__
    for row in resolved:
        gold = set(row.get("relevant_chunk_ids") or [])
        answers = [norm_answer(a) for a in (row.get("accepted_answers") or [row.get("ground_truth")])]
        answers = [a for a in answers if a and len(a) >= 8]  # skip trivially short strings
        if not answers or not gold:
            continue
        rare = [w for w in set(tok(row["question"]))
                if w not in STOP and idf(w) >= RARE_IDF and w in postings]
        if not rare:
            continue
        checked += 1
        candidates: set[str] = set()
        for word in rare:
            candidates |= postings[word]
        candidates -= gold
        found_here = []
        for cid in candidates:
            body = text_of[cid]
            if any(a in body for a in answers):
                found_here.append(cid)
        if found_here:
            hits_any += 1
            if any(role_of[c] == "distractor" for c in found_here):
                hits_distractor += 1
                # Grade the match: string presence alone is weak evidence. How
                # many of the question's own rare terms does that distractor
                # also carry? 1 is likely coincidence, 2+ means same event.
                best = max(len([w for w in rare if w in words_of_chunk(c)])
                           for c in found_here if role_of[c] == "distractor")
                strength[min(best, 3)] += 1
                if len(examples) < 5:
                    other = next(c for c in found_here if role_of[c] == "distractor")
                    examples.append((row, other))

    print(f"  questions checked                          : {checked:,}")
    print(f"  a non-gold chunk contains the gold answer  : {hits_any:,}  "
          f"({hits_any/max(1,checked):.1%})")
    print(f"  ...and that chunk is a DISTRACTOR article  : {hits_distractor:,}  "
          f"({hits_distractor/max(1,checked):.1%})")
    print("""
  A distractor containing the gold answer is an UNLABELLED correct answer:
  retrieval returning it is scored wrong. This is the benchmark's
  false-negative rate, measured rather than assumed.

  But a raw string match overstates it - "German authorities" turns up in
  unrelated articles. Grading by how many of the question's rare terms the
  answer-bearing distractor also carries separates coincidence from a genuine
  second copy of the same event:""")
    none = checked - hits_distractor
    labels = {0: "0 terms - coincidental string match",
              1: "1 term  - likely coincidental",
              2: "2 terms - probably the same event",
              3: "3+ terms- almost certainly the same event"}
    print(f"\n{'no answer-bearing distractor':44s} {none:5,d}  {none/checked:6.1%}")
    for k in sorted(labels):
        v = strength.get(k, 0)
        print(f"    {labels[k]:44s} {v:5,d}  {v/checked:6.1%}")
    strong = sum(v for k, v in strength.items() if k >= 2)
    print(f"\nSTRONG unlabelled-answer candidates (>=2 shared rare terms): "
          f"{strong:,}  ({strong/checked:.1%})")
    print(f"  UPPER BOUND (any string match)                            : "
          f"{hits_distractor:,}  ({hits_distractor/checked:.1%})")

    if examples:
        print("\n  Examples:")
        for row, cid in examples:
            print(f"\n    Q: {row['question'][:82]}")
            print(f"    A: {str(row.get('ground_truth'))[:60]}")
            print(f"    gold chunk     : {list(row['relevant_chunk_ids'])[:1]}")
            print(f"    distractor with the same answer: {cid}  "
                  f"(article {article_of[cid]})")

    print()
    print("=" * 74)
    print("READING")
    print("=" * 74)
    print(f"""
  Competition: the median question has {comp[m//2]:,} non-gold chunks sharing one of
  its rare terms, and only {tight/m:.0%} are narrowed to 10 or fewer. First-stage
  lexical retrieval gets close but cannot finish the job on its own - which is
  the empirical case for a reranker.

  Ambiguity: {sum(v for k,v in strength.items() if k>=2)/max(1,checked):.1%} of questions have a distractor that plausibly answers
  them - same event, different article - and up to {hits_distractor/max(1,checked):.1%} on a raw string
  match. That range is the closed-world assumption's error rate.""")

    C.save("04_distractor_collision", {
        "questions_with_rare": m,
        "questions_without_rare": no_rare,
        "competitors_median": comp[m // 2],
        "competitors_p90": comp[int(.90 * m)],
        "narrowed_to_10": tight / m,
        "checked": checked,
        "nongold_has_answer": hits_any,
        "distractor_has_answer": hits_distractor,
        "match_strength": dict(strength),
        "strong_candidates": sum(v for k, v in strength.items() if k >= 2),
    })
    print("\nSaved -> out/04_distractor_collision.json")


if __name__ == "__main__":
    main()
