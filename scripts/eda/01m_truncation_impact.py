"""Does the truncation actually damage THIS benchmark?

Truncation removes the tail of an article. Answers sit somewhere inside it.
If answers live far from the cut, the defect is cosmetic for our task. If they
crowd the boundary, it is real. Measure it instead of arguing about it.
"""

from __future__ import annotations

import common as C

# Articles longer than this are in the zone where truncation demonstrably happens
# (verified pairs ranged 3,841-4,313 chars).
AT_RISK_CHARS = 3800


def main() -> None:
    evaluation = C.articles("evaluation")
    by_id = {a["article_id"]: a for a in evaluation}
    resolved = C.variant("resolved")

    print("=" * 78)
    print("1. HOW MUCH OF THE QUESTION-BEARING SET IS PLAUSIBLY TRUNCATED?")
    print("=" * 78)
    at_risk = {a["article_id"] for a in evaluation if len(a["context"]) >= AT_RISK_CHARS}
    print(f"  evaluation articles                    : {len(evaluation):,}")
    print(f"  at or above {AT_RISK_CHARS:,} chars (truncation zone): {len(at_risk):,}"
          f"  ({len(at_risk)/len(evaluation):.1%})")

    q_at_risk = [r for r in resolved if r.get("article_key") in at_risk]
    print(f"  resolved questions on those articles   : {len(q_at_risk):,}"
          f"  of {len(resolved):,}  ({len(q_at_risk)/len(resolved):.1%})")

    print()
    print("=" * 78)
    print("2. WHERE DOES THE EVIDENCE SIT, RELATIVE TO THE CUT?")
    print("=" * 78)
    print("  Truncation removes the END. Evidence near the end is at risk;")
    print("  evidence near the front is untouched by definition.\n")

    positions = []
    for row in resolved:
        article = by_id.get(row.get("article_key"))
        spans = row.get("evidence_spans") or []
        if not article or not spans:
            continue
        end = max(s["end"] for s in spans)
        positions.append((end / len(article["context"]), row, article))

    positions.sort(key=lambda item: item[0])
    values = [p for p, _, _ in positions]
    n = len(values)
    print(f"  questions measured: {n:,}")
    print(f"  evidence END position within the article:")
    for label, value in (("median", values[n // 2]), ("p75", values[int(.75 * n)]),
                         ("p90", values[int(.90 * n)]), ("p95", values[int(.95 * n)]),
                         ("max", values[-1])):
        print(f"    {label:7s} {value:6.1%}")

    for threshold in (0.80, 0.90, 0.95):
        count = sum(1 for v in values if v >= threshold)
        print(f"\n  evidence ending in the last {100-threshold*100:.0f}% of its article: "
              f"{count:,}  ({count/n:.1%})")

    print()
    print("=" * 78)
    print("3. THE ACTUAL RISK GROUP: long article AND evidence near the end")
    print("=" * 78)
    risky = [
        (pos, row, art) for pos, row, art in positions
        if pos >= 0.90 and len(art["context"]) >= AT_RISK_CHARS
    ]
    print(f"  questions on a truncation-zone article whose evidence sits")
    print(f"  in the final 10% of the retained text: {len(risky):,}  ({len(risky)/n:.2%})")
    print("\n  These are the only questions where a longer article could plausibly")
    print("  have contained a competing or better answer.\n")
    for pos, row, art in risky[:5]:
        print(f"    [{pos:.0%} into a {len(art['context']):,}-char article]")
        print(f"      Q: {row['question'][:88]}")
        print(f"      A: {str(row.get('ground_truth'))[:70]}")

    print()
    print("=" * 78)
    print("4. WHAT WOULD RE-CRAWLING COST AND BREAK?")
    print("=" * 78)
    no_url = len(evaluation)  # the HF corpus carries no URLs at all
    print(f"  evaluation articles carrying a source URL in the corpus : 0 of {no_url}")
    print("    (HF metadata is only {publisher, title-as-first-150-chars})")
    print("  live-fetch outcome on the 4 URLs we did have:")
    print("    1 confirmed  |  1 partial  |  1 page re-rendered  |  1 dead (404)")
    print("  CNN rate-limited the session after 4 requests.")
    print("""
  And the methodological cost, which matters more than the effort:
  evidence spans are character offsets into the TRUNCATED text. Appending a
  recovered tail keeps those offsets valid, but it injects text that no
  annotator ever saw. Any answer-bearing sentence in the restored tail becomes
  an UNLABELLED correct answer - so retrieval returning it would be scored
  WRONG. Restoring text without re-annotating actively corrupts the ground
  truth rather than improving it.""")

    C.save("01m_truncation_impact", {
        "evaluation_articles": len(evaluation),
        "at_risk_articles": len(at_risk),
        "questions_on_at_risk": len(q_at_risk),
        "evidence_position_median": values[n // 2],
        "evidence_last_10pct": sum(1 for v in values if v >= 0.90),
        "risk_group": len(risky),
        "n": n,
    })
    print("\nSaved -> out/01m_truncation_impact.json")


if __name__ == "__main__":
    main()
