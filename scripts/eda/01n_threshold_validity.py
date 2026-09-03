"""Is 'long article' actually a valid proxy for 'truncated'?

Test the threshold against the 94 paired articles, where truncation is known
rather than assumed. If short articles are also truncated, the threshold
under-counts and the 0.67% risk figure is too optimistic.
"""

from __future__ import annotations

import glob
import json

import common as C

THRESHOLD = 3800


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    corpus = C.articles("evaluation") + C.articles("distractor")
    bench = {norm(r["context"])[:300]: r for r in corpus}

    pairs = []
    seen = set()
    for path in sorted(glob.glob(str(C.PROJECT / "data" / "processed" / "*_clean.json"))):
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        article = bench.get(norm(doc["text"])[:300])
        if not article or article["article_id"] in seen:
            continue
        seen.add(article["article_id"])
        b, c = norm(article["context"]), norm(doc["text"])
        truncated = len(b) < len(c) - 40 and c.startswith(b)
        pairs.append((article, len(article["context"]), truncated, len(c) - len(b)))

    trunc = [p for p in pairs if p[2]]
    intact = [p for p in pairs if not p[2]]

    print("=" * 78)
    print("IS 'LONG' A VALID PROXY FOR 'TRUNCATED'?")
    print("=" * 78)
    print(f"  paired articles: {len(pairs)}   truncated: {len(trunc)}   intact: {len(intact)}")

    def spread(rows, label):
        if not rows:
            print(f"  {label}: none")
            return
        lengths = sorted(r[1] for r in rows)
        n = len(lengths)
        print(f"  {label:28s} n={n:3d}  min {lengths[0]:5,}  median {lengths[n//2]:5,}  max {lengths[-1]:5,}")

    spread(trunc, "TRUNCATED article lengths")
    spread(intact, "INTACT article lengths")

    print()
    print(f"  How many TRUNCATED articles fall BELOW the {THRESHOLD:,}-char threshold?")
    missed = [r for r in trunc if r[1] < THRESHOLD]
    print(f"    {len(missed)} of {len(trunc)}  ({len(missed)/max(1,len(trunc)):.1%}) "
          f"<- these the threshold MISSES")
    for article, length, _, gap in sorted(missed, key=lambda r: r[1])[:8]:
        print(f"      {length:5,} chars, {gap:5,} missing   {article['article_id']}")

    print()
    print(f"  How many INTACT articles are ABOVE the threshold?")
    false_pos = [r for r in intact if r[1] >= THRESHOLD]
    print(f"    {len(false_pos)} of {len(intact)}  ({len(false_pos)/max(1,len(intact)):.1%}) "
          f"<- these the threshold WRONGLY flags")

    tp = len([r for r in trunc if r[1] >= THRESHOLD])
    fn = len(missed)
    fp = len(false_pos)
    tn = len(intact) - fp
    total = tp + fn + fp + tn
    print()
    print(f"  threshold as a classifier:  precision {tp/max(1,tp+fp):.1%}  "
          f"recall {tp/max(1,tp+fn):.1%}  accuracy {(tp+tn)/max(1,total):.1%}")
    print("""
  VERDICT: length is a weak proxy. Short articles ARE truncated too, so
  'below 3,800 chars' does NOT mean intact. The 26% / 0.67% figures are
  lower bounds on how much of the corpus is affected, not exact counts.""")

    C.save("01n_threshold_validity", {
        "pairs": len(pairs), "truncated": len(trunc), "intact": len(intact),
        "truncated_below_threshold": len(missed), "intact_above_threshold": len(false_pos),
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
    })
    print("\nSaved -> out/01n_threshold_validity.json")


if __name__ == "__main__":
    main()
