"""A4 - is the benchmark dominated by a few question forms?

If one form dominates, a single headline retrieval score describes that form
rather than the system, and results need a per-form breakdown to be honest.

Classifies on two axes that matter for retrieval:
  form        the wh-word / interrogative shape
  answer type what kind of thing the gold answer is
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import common as C

WH = [
    (r"^(who|whom|whose)\b", "who"),
    (r"^what\b", "what"),
    (r"^(where)\b", "where"),
    (r"^(when)\b", "when"),
    (r"^(why)\b", "why"),
    (r"^how many\b", "how many"),
    (r"^how much\b", "how much"),
    (r"^how long\b", "how long"),
    (r"^how old\b", "how old"),
    (r"^how\b", "how (other)"),
    (r"^(which)\b", "which"),
    (r"^(is|are|was|were|did|does|do|has|have|had|can|will|would)\b", "yes/no"),
]

MONTHS = r"(january|february|march|april|may|june|july|august|september|october|november|december)"


def form_of(question: str) -> str:
    q = question.strip().lower()
    q = re.sub(r"^[\"'(\[]+", "", q)
    for pattern, label in WH:
        if re.match(pattern, q):
            return label
    return "other"


def answer_type(answer: str) -> str:
    a = (answer or "").strip()
    if not a:
        return "empty"
    low = a.lower()
    if re.fullmatch(r"(yes|no)\.?", low):
        return "yes/no"
    if re.search(r"\d", a) and re.search(r"(percent|%|\$|million|billion|thousand)", low):
        return "quantity"
    if re.search(rf"\b{MONTHS}\b|\b(19|20)\d{{2}}\b|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", low):
        return "date/time"
    if re.fullmatch(r"[\d,.]+( \w+)?", a):
        return "number"
    words = a.split()
    caps = sum(1 for w in words if w[:1].isupper())
    if caps and caps >= len(words) * 0.6:
        return "proper noun"
    if len(words) <= 3:
        return "short phrase"
    return "long phrase"


def bar(count: int, total: int, width: int = 44) -> str:
    return "#" * max(0, int(width * count / total))


def report(rows: list[dict], label: str) -> dict:
    total = len(rows)
    forms = Counter(form_of(r["question"]) for r in rows)
    print(f"\n  {label}  (n={total:,})")
    for name, count in forms.most_common():
        print(f"    {name:14s} {count:5,d}  {count/total:6.1%}  {bar(count, total)}")
    top2 = sum(c for _, c in forms.most_common(2))
    top3 = sum(c for _, c in forms.most_common(3))
    print(f"    -> top 2 forms cover {top2/total:.1%},  top 3 cover {top3/total:.1%}")
    return {"counts": dict(forms), "top2_share": top2 / total, "top3_share": top3 / total}


def main() -> None:
    original = C.variant("original")
    resolved = C.variant("resolved")

    print("=" * 76)
    print("A4. QUESTION FORM DISTRIBUTION")
    print("=" * 76)
    payload = {
        "original": report(original, "original wording"),
        "resolved": report(resolved, "resolved wording"),
    }

    print()
    print("=" * 76)
    print("DOES RESOLUTION CHANGE THE FORM?")
    print("=" * 76)
    by_id = {r["question_id"]: r for r in original}
    changed = Counter()
    same = 0
    for row in resolved:
        before = by_id.get(row["question_id"])
        if not before:
            continue
        a, b = form_of(before["question"]), form_of(row["question"])
        if a == b:
            same += 1
        else:
            changed[f"{a} -> {b}"] += 1
    total_pairs = same + sum(changed.values())
    print(f"  paired questions: {total_pairs:,}")
    print(f"  form unchanged  : {same:,}  ({same/total_pairs:.1%})")
    for move, count in changed.most_common(6):
        print(f"    {move:28s} {count:4,d}")

    print()
    print("=" * 76)
    print("ANSWER TYPE, AND HOW IT PAIRS WITH FORM")
    print("=" * 76)
    types = Counter(answer_type(str(r.get("ground_truth"))) for r in resolved)
    n = len(resolved)
    for name, count in types.most_common():
        print(f"    {name:14s} {count:5,d}  {count/n:6.1%}  {bar(count, n)}")
    payload["answer_types"] = dict(types)

    print("\n  form x answer-type (share within each form):")
    grid = defaultdict(Counter)
    for row in resolved:
        grid[form_of(row["question"])][answer_type(str(row.get("ground_truth")))] += 1
    for form, counts in sorted(grid.items(), key=lambda kv: -sum(kv[1].values()))[:6]:
        total_form = sum(counts.values())
        top = ", ".join(f"{k} {v/total_form:.0%}" for k, v in counts.most_common(3))
        print(f"    {form:14s} n={total_form:4,d}   {top}")

    print()
    print("=" * 76)
    print("WHY THIS MATTERS FOR THE SCORES")
    print("=" * 76)
    dominant = Counter(form_of(r["question"]) for r in resolved).most_common(1)[0]
    print(f"""
  The largest form is '{dominant[0]}' at {dominant[1]/n:.1%} of the benchmark.
  A single headline MRR@5 is therefore weighted heavily toward that form. Two
  consequences for the report:

    1. Publish retrieval metrics BROKEN DOWN BY FORM, not only in aggregate.
       If one form scores far above the rest, the headline number flatters the
       system on questions it happens to see most often.
    2. When comparing retrievers, check whether the winner wins on every form
       or only on the dominant one. A retriever that wins only on '{dominant[0]}'
       has not been shown to be better in general.""")

    C.save("02_question_forms", payload)
    print("\nSaved -> out/02_question_forms.json")


if __name__ == "__main__":
    main()
