"""C1 / C3 - do questions share rare terms with their gold chunk?

C1: measure question-to-gold lexical overlap against a random-chunk baseline.
    A large gap predicts that lexical retrieval should beat dense retrieval on
    this corpus, independently of any tournament result.
C3: run the same measurement on original vs resolved wording, to test whether
    resolution inflated the lexical signal by injecting proper nouns.
"""

from __future__ import annotations

import math
import random
import re
import statistics as st
from collections import Counter

import common as C

RARE_IDF = 6.0  # a word in fewer than ~50 of 19,263 chunks
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


def main() -> None:
    chunks = C.chunks()
    by_id = {c["id"]: c for c in chunks}
    n_chunks = len(chunks)

    print("Building document frequencies over the corpus...", flush=True)
    df: Counter = Counter()
    words_of: dict[str, set[str]] = {}
    for chunk in chunks:
        words = set(tok(chunk["text"]))
        words_of[chunk["id"]] = words
        df.update(words)

    def idf(word: str) -> float:
        return math.log(n_chunks / (1 + df.get(word, 0)))

    rng = random.Random(42)
    chunk_ids = list(by_id)

    def measure(rows: list[dict], label: str) -> dict:
        share_all, share_rare, rare_counts, gold_idf = [], [], [], []
        base_all, base_rare = [], []
        per_form: dict[str, list[int]] = {}

        for row in rows:
            gold_ids = [g for g in (row.get("relevant_chunk_ids") or []) if g in words_of]
            if not gold_ids:
                continue
            gold_words: set[str] = set()
            for g in gold_ids:
                gold_words |= words_of[g]

            q = [w for w in set(tok(row["question"])) if w not in STOP]
            if not q:
                continue
            rare_q = [w for w in q if idf(w) >= RARE_IDF]

            hit = [w for w in q if w in gold_words]
            rare_hit = [w for w in rare_q if w in gold_words]
            share_all.append(len(hit) / len(q))
            rare_counts.append(len(rare_hit))
            if rare_q:
                share_rare.append(len(rare_hit) / len(rare_q))
            gold_idf.append(sum(idf(w) for w in hit))

            # baseline: the same question against a random chunk
            other = words_of[rng.choice(chunk_ids)]
            b_hit = [w for w in q if w in other]
            base_all.append(len(b_hit) / len(q))
            base_rare.append(len([w for w in rare_q if w in other]))

            first = row["question"].strip().split()[0].lower() if row["question"].strip() else "?"
            per_form.setdefault(first, []).append(len(rare_hit))

        n = len(share_all)
        print(f"\n  {label}  (n={n:,})")
        print(f"    question words also in gold chunk : {st.fmean(share_all):6.1%}")
        print(f"    same, against a RANDOM chunk      : {st.fmean(base_all):6.1%}   <- baseline")
        print(f"    lift                              : {st.fmean(share_all)/max(1e-9,st.fmean(base_all)):6.1f}x")
        print()
        print(f"    RARE words shared with gold chunk : mean {st.fmean(rare_counts):.2f}  "
              f"median {int(st.median(rare_counts))}")
        print(f"    RARE words shared with random     : mean {st.fmean(base_rare):.3f}   <- baseline")
        dist = Counter(rare_counts)
        total = sum(dist.values())
        for k in sorted(dist)[:6]:
            print(f"      {k} rare term(s): {dist[k]:5,d}  ({dist[k]/total:5.1%})")
        at_least_one = sum(v for k, v in dist.items() if k >= 1)
        print(f"    questions with >=1 rare locator   : {at_least_one:,}  ({at_least_one/total:.1%})")
        return {
            "n": n,
            "overlap_gold": st.fmean(share_all),
            "overlap_random": st.fmean(base_all),
            "rare_gold_mean": st.fmean(rare_counts),
            "rare_random_mean": st.fmean(base_rare),
            "at_least_one_rare": at_least_one / total,
            "per_form": {k: st.fmean(v) for k, v in per_form.items() if len(v) >= 25},
        }

    print("=" * 74)
    print("C1. DOES THE QUESTION SHARE RARE TERMS WITH ITS OWN GOLD CHUNK?")
    print("=" * 74)
    original = measure(C.variant("reviewed_original"), "ORIGINAL wording")
    resolved = measure(C.variant("resolved"), "RESOLVED wording")

    print()
    print("=" * 74)
    print("C3. DID RESOLUTION INFLATE THE LEXICAL SIGNAL?")
    print("=" * 74)
    d_rare = resolved["rare_gold_mean"] - original["rare_gold_mean"]
    print(f"  rare locators per question, original : {original['rare_gold_mean']:.2f}")
    print(f"  rare locators per question, resolved : {resolved['rare_gold_mean']:.2f}")
    print(f"  change                               : {d_rare:+.2f}  "
          f"({d_rare/max(1e-9,original['rare_gold_mean']):+.0%})")
    print(f"  questions with >=1 rare locator      : "
          f"{original['at_least_one_rare']:.1%} -> {resolved['at_least_one_rare']:.1%}")

    print()
    print("  rare locators by question form (resolved):")
    for form, value in sorted(resolved["per_form"].items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {form:12s} {value:.2f}")

    print()
    print("=" * 74)
    print("READING")
    print("=" * 74)
    lift = resolved["rare_gold_mean"] / max(1e-9, resolved["rare_random_mean"])
    print(f"""
  A question shares {resolved['rare_gold_mean']:.2f} rare terms with its own gold chunk on
  average, versus {resolved['rare_random_mean']:.3f} with a random chunk - a {lift:.0f}x separation.
  {resolved['at_least_one_rare']:.0%} of questions carry at least one rare locator.

  That is the mechanism a lexical retriever exploits directly, and it predicts
  sparse retrieval winning on this corpus without appealing to any tournament
  result. The C3 delta says how much of that signal resolution added.""")

    C.save("03_lexical_overlap", {"original": original, "resolved": resolved,
                                  "rare_delta": d_rare})
    print("\nSaved -> out/03_lexical_overlap.json")


if __name__ == "__main__":
    main()
