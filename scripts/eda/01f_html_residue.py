"""How much HTML page furniture survived into the benchmark text, and how often
an article ENDS in furniture rather than prose."""

from __future__ import annotations

import re

import common as C

SIGNATURES = [
    (r"»", "link marker  >>  (see video / watch teaser)"),
    (r"•", "bullet separator (nav / related topics)"),
    (r"(?i)all about", '"All About" related-topics block'),
    (r"(?i)e-mail to a friend", "e-mail-to-a-friend widget"),
    (r"(?i)watch\b[^.]{0,40}»", "watch-video teaser"),
    (r"\(CNN\)\s*--", "(CNN) -- dateline anywhere"),
    (r"^\(CNN\)\s*--", "(CNN) -- dateline at the very start"),
    (r"(?i)click here", "click-here link text"),
    (r"\n{4,}", "4+ blank lines (block boundary residue)"),
    (r"(?i)copyright|all rights reserved", "copyright footer"),
]

TAIL_SIGNATURES = [
    (r"»\s*\Z", "ends on a link marker"),
    (r"•[^•]{0,60}\Z", "ends inside a bullet list"),
    (r"(?i)(all about|e-mail to a friend)[^.]{0,60}\Z", "ends in a widget label"),
    (r"[a-z,]\s*\Z", "ends on a lowercase word or comma (mid-sentence)"),
    (r"[.!?][\"')\]]?\s*\Z", "ends on sentence punctuation"),
]


def main() -> None:
    articles = C.articles("evaluation") + C.articles("distractor")
    texts = [a["context"] for a in articles]
    n = len(texts)

    print("=" * 78)
    print("HTML-EXTRACTION RESIDUE STILL PRESENT IN THE BENCHMARK TEXT")
    print("=" * 78)
    for pattern, name in SIGNATURES:
        count = sum(bool(re.search(pattern, t)) for t in texts)
        flag = "!!" if count / n > 0.10 else " -" if count else "  "
        print(f"  {flag} {name:46s} {count:6,d}  ({count / n:5.1%})")

    print()
    print("=" * 78)
    print("WHAT DOES THE LAST CHARACTER OF AN ARTICLE LOOK LIKE?")
    print("=" * 78)
    tails = [re.sub(r"\s+", " ", t).strip() for t in texts]
    for pattern, name in TAIL_SIGNATURES:
        count = sum(bool(re.search(pattern, t)) for t in tails)
        print(f"     {name:50s} {count:6,d}  ({count / n:5.1%})")

    print()
    print("  20 random article endings, read them yourself:")
    import random

    for t in random.Random(42).sample(tails, 20):
        print(f"    ...{t[-72:]!r}")

    C.save("01f_html_residue", {
        "signatures": {
            name: sum(bool(re.search(p, t)) for t in texts) for p, name in SIGNATURES
        },
        "tails": {
            name: sum(bool(re.search(p, t)) for t in tails) for p, name in TAIL_SIGNATURES
        },
    })
    print("\nSaved -> out/01f_html_residue.json")


if __name__ == "__main__":
    main()
