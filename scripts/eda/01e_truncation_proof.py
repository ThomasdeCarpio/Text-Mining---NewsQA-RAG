"""Settle the truncation question by direct comparison, not heuristics.

93 articles appear in both the benchmark corpus (from HuggingFace NewsQA) and
the crawler output (fetched from cnn.com). If the benchmark text is a prefix of
the crawled text and stops short, that is truncation observed rather than inferred.
"""

from __future__ import annotations

import glob
import json
import re

import common as C


def norm(text: str) -> str:
    return " ".join(text.lower().split())


SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s*$")


def main() -> None:
    bench = {}
    for article in C.articles("evaluation") + C.articles("distractor"):
        bench[norm(article["context"])[:300]] = article

    crawled = []
    for path in glob.glob(str(C.PROJECT / "data" / "processed" / "*_clean.json")):
        with open(path, encoding="utf-8") as handle:
            crawled.append(json.load(handle))

    pairs = []
    for doc in crawled:
        key = norm(doc["text"])[:300]
        if key in bench:
            pairs.append((bench[key], doc))

    print("=" * 78)
    print(f"PAIRED ARTICLES: {len(pairs)}  (same article, two independent sources)")
    print("=" * 78)

    shorter = prefix_match = 0
    deltas = []
    for article, doc in pairs:
        b, c = norm(article["context"]), norm(doc["text"])
        if len(b) < len(c):
            shorter += 1
            deltas.append(len(c) - len(b))
            if c.startswith(b[:-40]):
                prefix_match += 1

    print(f"  benchmark version SHORTER than crawled : {shorter} / {len(pairs)}")
    print(f"  benchmark text is a PREFIX of crawled  : {prefix_match} / {len(pairs)}")
    if deltas:
        deltas.sort()
        print(f"  characters missing: median {deltas[len(deltas)//2]:,}  "
              f"max {deltas[-1]:,}  total {sum(deltas):,}")

    print()
    print("=" * 78)
    print("WHAT THE CUT LOOKS LIKE  (3 examples)")
    print("=" * 78)
    shown = 0
    for article, doc in pairs:
        b, c = article["context"], doc["text"]
        if len(norm(b)) >= len(norm(c)):
            continue
        nb, nc = norm(b), norm(c)
        if not nc.startswith(nb[:-40]):
            continue
        print(f"\n  benchmark {len(b):,} chars   crawled {len(c):,} chars   "
              f"missing {len(nc)-len(nb):,}")
        print(f"    benchmark ENDS : ...{nb[-90:]!r}")
        cut = nc[len(nb) - 40:len(nb) + 90]
        print(f"    crawled CONTINUES: ...{cut!r}")
        shown += 1
        if shown == 3:
            break

    print()
    print("=" * 78)
    print("HOW GOOD IS THE 'ends with . ! ?' HEURISTIC?")
    print("=" * 78)
    print("  Tested against the paired ground truth above.\n")
    tp = fp = tn = fn = 0
    for article, doc in pairs:
        nb, nc = norm(article["context"]), norm(doc["text"])
        truly_cut = len(nb) < len(nc) - 40 and nc.startswith(nb[:-40])
        heuristic_says_cut = not SENTENCE_END.search(article["context"].rstrip())
        if truly_cut and heuristic_says_cut:
            tp += 1
        elif truly_cut and not heuristic_says_cut:
            fn += 1
        elif not truly_cut and heuristic_says_cut:
            fp += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    print(f"    truly cut  + heuristic flags it   (correct) : {tp:4d}")
    print(f"    truly cut  + heuristic misses it  (MISS)    : {fn:4d}")
    print(f"    not cut    + heuristic flags it   (FALSE +) : {fp:4d}")
    print(f"    not cut    + heuristic clears it  (correct) : {tn:4d}")
    if tp + fp:
        print(f"\n    precision {tp/(tp+fp):.1%}   ", end="")
    if tp + fn:
        print(f"recall {tp/(tp+fn):.1%}   ", end="")
    print(f"accuracy {(tp+tn)/total:.1%}" if total else "")

    C.save("01e_truncation_proof", {
        "paired": len(pairs),
        "benchmark_shorter": shorter,
        "prefix_match": prefix_match,
        "heuristic": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    })
    print("\nSaved -> out/01e_truncation_proof.json")


if __name__ == "__main__":
    main()
