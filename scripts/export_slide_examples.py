#!/usr/bin/env python3
"""Freeze the worked examples the slides show, as a committed artifact.

The per-question scores under docs/reports/phase2/scores/ and phase2c/scores/
carry the answers and every metric, but not the question text or the evidence passage:
those live in the evaluation dataset, which is too large to commit. This script joins
the two once and writes the cases the deck quotes, so the slides keep building
from committed artifacts alone.

Rerun only when changing which examples the deck shows; it needs the local
dataset under data/evaluation/.

    python scripts/export_slide_examples.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "data/evaluation/newsqa_200_11064_restored/final/testset_resolved.jsonl"
CHUNKS = ROOT / "data/evaluation/newsqa_200_11064/final/chunks.jsonl"
OUT = ROOT / "docs/reports/phase2/examples/slide_examples.json"

CASES = {
    "prompt_effect": "94b8ba1c855e4789a90cd5692ee3e710",
    "abstains": "9206f18617d749b09fc00c5f949e4a54",
    "answers_anyway": "ac0b5576a66945a299d5ff45ce412c7a",
    "reranker": "04aba1b206a74b468046427c8ba664a9",
    "depth_abstain": "c20046f2bae84d69a5bc57eb097ed474",
    "depth_distractor": "1416c66b2eb543aea04972d5780b31d6",
    "heldout": "2aa93a690e904c89b259f98500106b69",
    "p2c_c0_vs_c3": "27b67e1a4f4d40b5b5d242ef0b789b9a",
    "reranker_ferry": "ef29092974ce40b980f963ac746247ec",
}

EVIDENCE_CHARS = 240


def read_jsonl(path: Path, key: str = "question_id") -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row[key]] = row
    return rows


def tidy(text: str) -> str:
    """Chunk text keeps the article's paragraph padding; slides need one line."""
    return re.sub(r"\s+", " ", text).strip()


def snippet(text: str, target: str | None = None, max_chars: int = EVIDENCE_CHARS) -> str:
    text = tidy(text)
    if len(text) <= max_chars:
        return text
    if target:
        t_clean = target.lower().strip("., ")
        pos = text.lower().find(t_clean)
        if pos != -1 and pos >= max_chars // 2:
            start = max(0, pos - max_chars // 3)
            start = text.find(" ", start) + 1 if start > 0 else 0
            end = min(len(text), start + max_chars)
            end = text.rfind(" ", start, end) if end < len(text) else len(text)
            prefix = "... " if start > 0 else ""
            suffix = " ..." if end < len(text) else ""
            return prefix + text[start:end].strip() + suffix
    cut = text[:max_chars]
    return cut[:cut.rfind(" ")] + " ..."


def extract_run_entry(row: dict) -> dict:
    return {
        "answer": tidy(row["evaluated_answer"]),
        "citation_indices": row["citation_indices"],
        "answer_correctness": row.get("ragas", {}).get("answer_correctness", 0.0),
        "exact_match": row.get("qa", {}).get("exact_match", 0.0),
        "token_f1": row.get("qa", {}).get("f1", 0.0),
        "citation_f1": row.get("citations", {}).get("citation_f1", 0.0),
        "faithfulness": row.get("ragas", {}).get("faithfulness", 0.0),
        "gold_in_top5": row.get("retrieval", {}).get("hit_rate@5") == 1.0,
        "gold_rank_one": row.get("retrieval", {}).get("hit_rate@1") == 1.0,
    }


def main() -> None:
    for path in (TESTSET, CHUNKS):
        if not path.exists():
            raise SystemExit(f"Missing {path.relative_to(ROOT)}; the local dataset is required.")

    questions = read_jsonl(TESTSET)

    p0_d5 = read_jsonl(ROOT / "docs/reports/phase2/scores/p0_d5_development.jsonl")
    p2_d3 = read_jsonl(ROOT / "docs/reports/phase2/scores/p2_d3_development.jsonl")
    p2_d5 = read_jsonl(ROOT / "docs/reports/phase2/scores/p2_d5_development.jsonl")
    p2_ho = read_jsonl(ROOT / "docs/reports/phase2/scores/p2_d5_heldout.jsonl")
    c0_d5 = read_jsonl(ROOT / "docs/reports/phase2c/scores/c0_d5_development.jsonl")
    c3_d5 = read_jsonl(ROOT / "docs/reports/phase2c/scores/c3_d5_development.jsonl")

    wanted = {cid for qid in CASES.values() for cid in questions[qid]["relevant_chunk_ids"]} | {"bc41ea4a51f0_chunk_0"}
    chunks = {row["id"]: row["text"] for row in
              (json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip())
              if row["id"] in wanted}

    cases = {}

    # 1. prompt_effect, abstains, answers_anyway (backward compatible)
    for label in ("prompt_effect", "abstains", "answers_anyway"):
        qid = CASES[label]
        q = questions[qid]
        case = {
            "question_id": qid,
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "evidence": snippet(chunks[q["relevant_chunk_ids"][0]]),
            "runs": {
                "p0": extract_run_entry(p0_d5[qid]),
                "p2": extract_run_entry(p2_d5[qid]),
            },
        }
        cases[label] = case

    # 2. reranker
    qid_rerank = CASES["reranker"]
    q_rerank = questions[qid_rerank]
    r_rerank = p2_d5[qid_rerank]
    cases["reranker"] = {
        "question_id": qid_rerank,
        "question": q_rerank["question"],
        "ground_truth": q_rerank["ground_truth"],
        "evidence": snippet(chunks[q_rerank["relevant_chunk_ids"][0]], q_rerank["ground_truth"]),
        "initial_rank": "> 10",
        "initial_ndcg5": r_rerank["retrieval_initial"]["ndcg@5"],
        "initial_hit1": r_rerank["retrieval_initial"]["hit_rate@1"],
        "reranked_rank": 1,
        "reranked_ndcg5": r_rerank["retrieval"]["ndcg@5"],
        "reranked_hit1": r_rerank["retrieval"]["hit_rate@1"],
        "runs": {
            "p2": extract_run_entry(r_rerank),
        },
    }

    # 3. depth_abstain (Kim Il Sung - missing at rank 4)
    qid_da = CASES["depth_abstain"]
    q_da = questions[qid_da]
    cases["depth_abstain"] = {
        "question_id": qid_da,
        "question": q_da["question"],
        "ground_truth": q_da["ground_truth"],
        "evidence": snippet(chunks[q_da["relevant_chunk_ids"][0]], q_da["ground_truth"]),
        "gold_rank": 4,
        "runs": {
            "d3": extract_run_entry(p2_d3[qid_da]),
            "d5": extract_run_entry(p2_d5[qid_da]),
        },
    }

    # 4. depth_distractor (Listeria - distractor at rank 1, gold at rank 5)
    qid_dd = CASES["depth_distractor"]
    q_dd = questions[qid_dd]
    cases["depth_distractor"] = {
        "question_id": qid_dd,
        "question": q_dd["question"],
        "ground_truth": q_dd["ground_truth"],
        "evidence": snippet(chunks[q_dd["relevant_chunk_ids"][0]], q_dd["ground_truth"]),
        "gold_rank": 5,
        "runs": {
            "d3": extract_run_entry(p2_d3[qid_dd]),
            "d5": extract_run_entry(p2_d5[qid_dd]),
        },
    }

    # 5. heldout (Omar bin Laden)
    qid_ho = CASES["heldout"]
    q_ho = questions[qid_ho]
    cases["heldout"] = {
        "question_id": qid_ho,
        "question": q_ho["question"],
        "ground_truth": q_ho["ground_truth"],
        "evidence": snippet(chunks[q_ho["relevant_chunk_ids"][0]], q_ho["ground_truth"]),
        "runs": {
            "p2": extract_run_entry(p2_ho[qid_ho]),
        },
    }

    # 6. p2c_c0_vs_c3 (Thaksin Shinawatra)
    qid_p2c = CASES["p2c_c0_vs_c3"]
    q_p2c = questions[qid_p2c]
    cases["p2c_c0_vs_c3"] = {
        "question_id": qid_p2c,
        "question": q_p2c["question"],
        "ground_truth": q_p2c["ground_truth"],
        "evidence": snippet(chunks[q_p2c["relevant_chunk_ids"][0]], q_p2c["ground_truth"]),
        "runs": {
            "c0": extract_run_entry(c0_d5[qid_p2c]),
            "c3": extract_run_entry(c3_d5[qid_p2c]),
        },
    }

    # 7. reranker_ferry (Ferry in Bangladesh: Philippines distractor at #1, Bangladesh gold at #6 -> #1)
    qid_rf = CASES["reranker_ferry"]
    q_rf = questions[qid_rf]
    r_rf = p2_d5[qid_rf]
    cases["reranker_ferry"] = {
        "question_id": qid_rf,
        "question": q_rf["question"],
        "ground_truth": q_rf["ground_truth"],
        "gold_snippet": snippet(chunks[q_rf["relevant_chunk_ids"][0]], q_rf["ground_truth"], max_chars=180),
        "distractor_snippet": snippet(chunks["bc41ea4a51f0_chunk_0"], max_chars=180),
        "initial_distractor_rank": 1,
        "initial_gold_rank": 6,
        "reranked_gold_rank": 1,
        "reranked_distractor_rank": 2,
        "answer": tidy(r_rf["evaluated_answer"]),
        "citation_indices": r_rf["citation_indices"],
        "answer_correctness": r_rf["ragas"]["answer_correctness"],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema_version": 2,
        "note": ("Verbatim answers from committed per-question scores, joined with "
                 "question and evidence text from the evaluation dataset. Evidence is the "
                 "gold chunk, truncated for display."),
        "sources": {
            "scores": [
                "p0_d5_development.jsonl",
                "p2_d3_development.jsonl",
                "p2_d5_development.jsonl",
                "p2_d5_heldout.jsonl",
                "c0_d5_development.jsonl",
                "c3_d5_development.jsonl",
            ],
            "testset": TESTSET.relative_to(ROOT).as_posix(),
            "testset_sha256": hashlib.sha256(TESTSET.read_bytes()).hexdigest(),
            "chunks": CHUNKS.relative_to(ROOT).as_posix(),
            "chunks_sha256": hashlib.sha256(CHUNKS.read_bytes()).hexdigest(),
        },
        "cases": cases,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} examples to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
