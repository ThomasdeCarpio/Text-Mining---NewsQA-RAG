"""Show, character by character, what 'the benchmark text is a prefix of the
crawled text' actually means - and how much tolerance my test allowed."""

from __future__ import annotations

import glob
import json

import common as C


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    bench = {norm(a["context"])[:300]: a
             for a in C.articles("evaluation") + C.articles("distractor")}
    pairs = []
    for path in glob.glob(str(C.PROJECT / "data" / "processed" / "*_clean.json")):
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        key = norm(doc["text"])[:300]
        if key in bench:
            pairs.append((bench[key], doc))

    # Pick a pair with a large, unambiguous cut.
    best = None
    for article, doc in pairs:
        b, c = norm(article["context"]), norm(doc["text"])
        if len(c) - len(b) > 1500 and c.startswith(b[:-40]):
            best = (article, doc, b, c)
            break

    article, doc, b, c = best

    print("=" * 78)
    print("WHAT 'PREFIX' MEANS, ON ONE REAL PAIR")
    print("=" * 78)
    print(f"  benchmark version : {len(b):,} characters")
    print(f"  crawled version   : {len(c):,} characters")
    print(f"  difference        : {len(c) - len(b):,} characters the benchmark does not have")

    print("\n  Do they START the same? Compare character by character:")
    shared = 0
    for x, y in zip(b, c):
        if x != y:
            break
        shared += 1
    print(f"    identical for the first {shared:,} characters")
    print(f"    that is {shared / len(b):.1%} of the benchmark version")

    print("\n  THE OVERLAP (last 100 chars the two versions share):")
    print(f"    ...{b[shared-100:shared]!r}")

    print("\n  WHERE THEY PART:")
    print(f"    benchmark  stops here and has nothing more")
    print(f"    crawled    continues: {c[shared:shared+180]!r}")

    print()
    print("=" * 78)
    print("WHY THIS MEANS 'TRUNCATED' AND NOT 'A DIFFERENT ARTICLE'")
    print("=" * 78)
    print("""
  Two versions of a text can differ in three ways:

    1. Different articles      -> they diverge almost immediately
    2. Different edits/edition -> they diverge somewhere in the MIDDLE,
                                  then both continue to their own ends
    3. One is CUT              -> they are identical from the start until
                                  one simply stops, and the other carries on

  This pair matches case 3. The benchmark text is identical to the crawl
  from character 0 up to where it ends, then ends mid-sentence. Nothing was
  reworded, reordered or replaced - content was removed from the tail.
    """)

    print("=" * 78)
    print("HONEST NOTE ON MY TEST")
    print("=" * 78)
    print("""
  My check was  crawled.startswith(benchmark[:-40])  - the crawled text must
  begin with the benchmark text MINUS its final 40 characters.

  That 40-character tolerance is deliberate: the benchmark's last word is
  often cut mid-token, so a zero-tolerance prefix test would reject genuine
  truncations on the final fragment alone.

  So 'strict prefix' overstated it. Accurate wording: the benchmark text is a
  prefix of the crawled text up to its final 40 characters. Below is how the
  count changes as the tolerance shrinks.
    """)
    for tolerance in (0, 5, 10, 20, 40, 80):
        hits = sum(
            1 for a, d in pairs
            if (lambda x, y: len(x) < len(y) and y.startswith(x[:-tolerance] if tolerance else x))(
                norm(a["context"]), norm(d["text"])
            )
        )
        print(f"    tolerance {tolerance:3d} chars -> {hits:3d} of {len(pairs)} pairs count as truncated")


if __name__ == "__main__":
    main()
