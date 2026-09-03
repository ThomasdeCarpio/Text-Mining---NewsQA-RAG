"""Test the claim that the source corpus is truncated.

Distinguishes three hypotheses:
  H1 article text is cut off mid-content (truncation)
  H2 long articles were dropped from the dataset (filtering)
  H3 neither - this is just how long CNN articles are (natural)
"""

from __future__ import annotations

import re
from collections import Counter

import common as C

SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s*$")


def ends_cleanly(text: str) -> bool:
    return bool(SENTENCE_END.search(text.rstrip()))


def main() -> None:
    articles = C.articles("evaluation") + C.articles("distractor")
    lengths = [len(a["context"]) for a in articles]
    ordered = sorted(lengths)
    top = ordered[-1]

    print("=" * 78)
    print("1. SHAPE OF THE UPPER TAIL")
    print("=" * 78)
    print("  A hard cap piles articles up at one value. A natural distribution tapers.\n")
    bins = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 3500), (3500, 4000),
            (4000, 4200), (4200, 4300), (4300, 4400), (4400, 4500), (4500, 4600)]
    for lo, hi in bins:
        count = sum(lo <= v < hi for v in lengths)
        bar = "#" * int(60 * count / len(lengths))
        print(f"  {lo:5,}-{hi:5,} chars  {count:6,d}  {bar}")

    print(f"\n  max observed: {top:,} chars")
    tail = Counter(v for v in lengths if v > top - 40)
    print(f"  exact counts within 40 chars of the max: {dict(sorted(tail.items()))}")
    print("  (a truncation cap would show a large spike on a single value)")

    print()
    print("=" * 78)
    print("2. DO THE LONGEST ARTICLES END MID-SENTENCE?")
    print("=" * 78)
    print("  Truncated text usually stops without terminal punctuation.\n")
    groups = {
        "shortest 25%": [a for a in articles if len(a["context"]) <= ordered[len(ordered) // 4]],
        "middle": [a for a in articles
                   if ordered[len(ordered) // 4] < len(a["context"]) < ordered[3 * len(ordered) // 4]],
        "longest 25%": [a for a in articles if len(a["context"]) >= ordered[3 * len(ordered) // 4]],
        "within 200 of max": [a for a in articles if len(a["context"]) >= top - 200],
    }
    for name, rows in groups.items():
        clean = sum(ends_cleanly(a["context"]) for a in rows)
        print(f"  {name:20s} n={len(rows):6,d}   ends with . ! or ? : {clean / len(rows):6.1%}")

    print("\n  Tails of the five longest articles:")
    for a in sorted(articles, key=lambda x: -len(x["context"]))[:5]:
        tail_text = " ".join(a["context"].split())[-70:]
        print(f"    {len(a['context']):5,d} chars  ...{tail_text!r}")

    print()
    print("=" * 78)
    print("3. IS THERE A CAP IN TOKENS OR WORDS INSTEAD OF CHARACTERS?")
    print("=" * 78)
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    longest = sorted(articles, key=lambda x: -len(x["context"]))[:400]
    tok = [len(enc.encode(a["context"])) for a in longest]
    wrd = [len(a["context"].split()) for a in longest]
    print(f"  among the 400 longest articles:")
    print(f"    tokens  max {max(tok):,}  distinct values in top 10: "
          f"{sorted(set(sorted(tok)[-10:]))}")
    print(f"    words   max {max(wrd):,}  distinct values in top 10: "
          f"{sorted(set(sorted(wrd)[-10:]))}")
    print("  (a token or word cap would repeat one value many times)")

    C.save("01d_truncation", {
        "max_chars": top,
        "tail_counts": {str(k): v for k, v in sorted(tail.items())},
        "ends_cleanly": {k: round(sum(ends_cleanly(a["context"]) for a in v) / len(v), 4)
                         for k, v in groups.items()},
    })
    print("\nSaved -> out/01d_truncation.json")


if __name__ == "__main__":
    main()
