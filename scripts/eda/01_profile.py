"""EDA step 1 - inventory, basic profiling, and a cleanliness audit of the source data.

Answers two things: what is in the dataset, and whether the raw NewsQA text and
question set were clean when they arrived.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import common as C

_enc = None


def tokens(text: str) -> int:
    """Token count under the same encoder the chunker uses."""
    global _enc
    if _enc is None:
        import tiktoken

        _enc = tiktoken.get_encoding("cl100k_base")
    return len(_enc.encode(text))


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show(summary: dict, label: str) -> None:
    print(
        f"  {label:32s} n={summary['n']:6d}  min {summary['min']:6.0f} | "
        f"median {summary['median']:7.1f} | p90 {summary['p90']:6.0f} | "
        f"max {summary['max']:6.0f} | mean {summary['mean']:7.1f}"
    )


# ---------------------------------------------------------------------------
# Cleanliness probes. Each counts the records exhibiting one artifact.
# ---------------------------------------------------------------------------

BOILERPLATE = [
    (r"^\(CNN\)\s*--", "opens with (CNN) -- dateline"),
    (r"^\([A-Za-z.]+\.com\)\s*--", "opens with (site.com) -- dateline"),
    (r"E-mail to a friend", "e-mail-to-a-friend footer"),
    (r"CNN\.com", "CNN.com self-reference"),
    (r"All Rights Reserved|Copyright \d{4}", "copyright line"),
    (r"»", "watch-teaser guillemet"),
]

MOJIBAKE = re.compile(r"â€|Ã[©¨¤¼]")
HTML_LEFTOVER = re.compile(r"&[a-z]{2,6};|&#\d+;|<[a-z/][^>]{0,40}>", re.I)


def audit_text(texts: list[str], label: str) -> dict:
    """Probe a body of text for artifacts a cleaning stage is meant to remove."""
    findings = {
        "leading/trailing whitespace": sum(t != t.strip() for t in texts),
        "3+ consecutive newlines": sum(bool(re.search(r"\n{3,}", t)) for t in texts),
        "double space inside a line": sum(bool(re.search(r"[^\n] {2,}[^\n]", t)) for t in texts),
        "tab character": sum("\t" in t for t in texts),
        "mojibake": sum(bool(MOJIBAKE.search(t)) for t in texts),
        "html entity or tag": sum(bool(HTML_LEFTOVER.search(t)) for t in texts),
        "non-ascii character": sum(any(ord(ch) > 127 for ch in t) for t in texts),
        "control character": sum(
            any(unicodedata.category(ch) == "Cc" and ch not in "\n\t\r" for ch in t)
            for t in texts
        ),
        "empty or whitespace only": sum(not t.strip() for t in texts),
    }
    for pattern, name in BOILERPLATE:
        findings[name] = sum(bool(re.search(pattern, t)) for t in texts)
    return {"label": label, "n": len(texts), "findings": findings}


def print_audit(audit: dict) -> None:
    print(f"\n  {audit['label']}  (n={audit['n']:,})")
    for issue, count in audit["findings"].items():
        share = count / audit["n"] if audit["n"] else 0.0
        flag = "  " if count == 0 else ("!!" if share > 0.10 else " -")
        print(f"   {flag} {issue:32s} {count:6,d}  ({share:6.1%})")


def main() -> None:
    payload: dict = {}

    section("1. INVENTORY")
    evaluation = C.articles("evaluation")
    distractor = C.articles("distractor")
    source_q = C.source_questions()
    chunk_rows = C.chunks()
    variants = {name: C.variant(name) for name in C.VARIANTS}
    notes = C.annotations()

    inventory = [
        {"artifact": "evaluation articles", "rows": f"{len(evaluation):,}"},
        {"artifact": "distractor articles", "rows": f"{len(distractor):,}"},
        {"artifact": "source questions (pre-review)", "rows": f"{len(source_q):,}"},
        {"artifact": "chunks @512/64", "rows": f"{len(chunk_rows):,}"},
        {"artifact": "review annotations", "rows": f"{len(notes):,}"},
    ] + [
        {"artifact": f"testset - {name}", "rows": f"{len(rows):,}"}
        for name, rows in variants.items()
    ]
    print(C.table(inventory, ["artifact", "rows"], [32, 10]))
    payload["inventory"] = {row["artifact"]: row["rows"] for row in inventory}

    section("2. CORPUS PROFILE")
    corpus_stats = {}
    for label, rows in (("evaluation", evaluation), ("distractor", distractor)):
        texts = [r["context"] for r in rows]
        measures = (
            ("chars", len),
            ("words", lambda t: len(t.split())),
            ("tokens", tokens),
        )
        for unit, fn in measures:
            summary = C.describe([fn(t) for t in texts], f"{label} {unit}")
            show(summary, f"{label} - {unit}")
            corpus_stats[f"{label}_{unit}"] = summary
        print()
    payload["corpus"] = corpus_stats

    section("3. CHUNK PROFILE")
    per_article = Counter(c["metadata"]["article_id"] for c in chunk_rows)
    show(C.describe(list(per_article.values())), "chunks per article")
    show(C.describe([tokens(c["text"]) for c in chunk_rows]), "chunk size (tokens)")
    dist = Counter(per_article.values())
    print("\n    chunks/article distribution:")
    for k in sorted(dist):
        print(f"      {k} chunk(s): {dist[k]:6,d} articles ({dist[k] / len(per_article):6.1%})")
    payload["chunks"] = {
        "per_article": C.describe(list(per_article.values())),
        "distribution": {str(k): v for k, v in sorted(dist.items())},
    }

    section("4. QUESTION PROFILE BY VARIANT")
    qstats = {}
    for name, rows in variants.items():
        qlen = C.describe([len(r["question"].split()) for r in rows])
        alen = C.describe([len(str(r.get("ground_truth") or "").split()) for r in rows])
        show(qlen, f"{name} - question words")
        show(alen, f"{name} - answer words")
        qstats[name] = {"question_words": qlen, "answer_words": alen}
    payload["questions"] = qstats

    section("5. WAS THE SOURCE DATA CLEAN?")
    print("  Probing raw article text and pre-review questions for artifacts a")
    print("  cleaning stage is meant to remove. '!!' marks issues above 10%.")

    audits = [
        audit_text([r["context"] for r in evaluation], "Article text - evaluation (raw context)"),
        audit_text([r["context"] for r in distractor], "Article text - distractor (raw context)"),
        audit_text([r["question"] for r in source_q], "Question text - pre-review"),
        audit_text([str(r.get("ground_truth") or "") for r in source_q], "Answer text - pre-review"),
    ]
    for audit in audits:
        print_audit(audit)
    payload["cleanliness"] = {a["label"]: a["findings"] for a in audits}

    section("6. QUESTION WELL-FORMEDNESS (pre-review)")
    q_texts = [r["question"] for r in source_q]
    defects = {
        "no question mark": sum(not q.strip().endswith("?") for q in q_texts),
        "starts lowercase": sum(bool(q[:1].islower()) for q in q_texts),
        "under 4 words": sum(len(q.split()) < 4 for q in q_texts),
        "no capitalised token after first": sum(
            not any(t[:1].isupper() for t in q.split()[1:]) for q in q_texts
        ),
        "contains a bare pronoun": sum(
            bool(re.search(r"\b(he|she|they|it|his|her|their|its)\b", q, re.I))
            for q in q_texts
        ),
    }
    for issue, count in defects.items():
        print(f"    {issue:38s} {count:5,d}  ({count / len(q_texts):6.1%})")
    payload["question_defects"] = defects

    section("7. INTEGRITY CROSSCHECKS")
    checks = []

    by_article = {a["article_id"]: a["context"] for a in evaluation}
    span_ok = span_bad = span_absent = 0
    for row in source_q:
        context = by_article.get(row["article_id"])
        spans = row.get("evidence_spans") or []
        if context is None or not spans:
            span_absent += 1
            continue
        for span in spans:
            if context[span["start"]:span["end"]] == span["text"]:
                span_ok += 1
            else:
                span_bad += 1
    checks.append({
        "check": "evidence span offsets match context",
        "result": f"{span_ok:,} ok / {span_bad:,} mismatched / {span_absent:,} absent",
    })

    chunk_ids = {c["id"] for c in chunk_rows}
    orphan = sum(
        1
        for r in variants["resolved"]
        for cid in (r.get("relevant_chunk_ids") or [])
        if cid not in chunk_ids
    )
    checks.append({"check": "gold chunk ids present in chunks.jsonl", "result": f"{orphan:,} orphaned"})

    no_gold = sum(1 for r in variants["resolved"] if not r.get("relevant_chunk_ids"))
    checks.append({"check": "resolved rows with no gold chunk", "result": f"{no_gold:,}"})

    ids = {name: {r["question_id"] for r in rows} for name, rows in variants.items()}
    checks.append({
        "check": "reviewed_original vs resolved id sets",
        "result": "identical" if ids["reviewed_original"] == ids["resolved"] else "DIFFER",
    })
    checks.append({
        "check": "clarified is a subset of resolved",
        "result": "yes" if ids["clarified"] <= ids["resolved"] else "NO",
    })

    contexts = [a["context"] for a in evaluation + distractor]
    normalized = [" ".join(c.lower().split()) for c in contexts]
    checks.append({
        "check": "duplicate article contexts (exact)",
        "result": f"{len(contexts) - len(set(contexts)):,}",
    })
    checks.append({
        "check": "duplicate article contexts (normalised)",
        "result": f"{len(normalized) - len(set(normalized)):,}",
    })

    print(C.table(checks, ["check", "result"], [40, 38]))
    payload["integrity"] = {c["check"]: c["result"] for c in checks}

    path = C.save("01_profile", payload)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
