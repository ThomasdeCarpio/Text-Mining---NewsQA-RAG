#!/usr/bin/env python3
"""Create a reproducible human-review sample of low-correctness RAG answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


STRATUM_TARGETS = {
    "severe_gold_in_top5": 10,
    "moderate_gold_in_top5": 14,
    "severe_gold_not_in_top5": 5,
    "moderate_gold_not_in_top5": 1,
}
ERROR_CATEGORIES = [
    "correct_but_verbose",
    "partially_correct",
    "wrong_entity_or_value",
    "answer_type_mismatch",
    "unsupported_extra_information",
    "retrieval_failure",
    "citation_error",
    "gold_or_evaluation_issue",
    "judge_disagreement",
    "other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--testset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correctness-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keyed(rows: list[dict]) -> dict[str, dict]:
    return {row["question_id"]: row for row in rows}


def stable_key(seed: int, question_id: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def stratum(correctness: float, gold_in_top5: bool) -> str:
    severity = "severe" if correctness < 0.25 else "moderate"
    retrieval = "gold_in_top5" if gold_in_top5 else "gold_not_in_top5"
    return f"{severity}_{retrieval}"


def select(candidates: list[dict], seed: int) -> list[dict]:
    by_stratum: dict[str, list[dict]] = {}
    for row in candidates:
        by_stratum.setdefault(row["sampling_stratum"], []).append(row)
    selected: list[dict] = []
    used_articles: set[str] = set()
    selection_order = sorted(
        STRATUM_TARGETS,
        key=lambda name: len({row["article_key"] for row in by_stratum.get(name, [])}),
    )
    for name in selection_order:
        target = STRATUM_TARGETS[name]
        pool = sorted(by_stratum.get(name, []), key=lambda row: stable_key(seed, row["question_id"]))
        unique = [row for row in pool if row["article_key"] not in used_articles]
        chosen = unique[:target]
        if len(chosen) < target:
            chosen_ids = {row["question_id"] for row in chosen}
            remaining = [row for row in pool if row["question_id"] not in chosen_ids]
            chosen.extend(remaining[: target - len(chosen)])
        if len(chosen) != target:
            raise ValueError(f"Stratum {name} has {len(chosen)} records; expected {target}")
        selected.extend(chosen)
        used_articles.update(row["article_key"] for row in chosen)
    return sorted(selected, key=lambda row: (row["answer_correctness"], row["question_id"]))


def compact(text: str, limit: int = 700) -> str:
    value = " ".join((text or "").split())
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def question_type(question: str) -> str:
    first = question.strip().lower().split(maxsplit=1)[0].strip(".,?!:;")
    if first in {"who", "what", "when", "where", "why", "how", "which"}:
        return first
    if first in {"is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "can", "could", "will", "would"}:
        return "yes_no"
    return "other"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = keyed(load_jsonl(run_dir / "predictions.jsonl"))
    judges = keyed(load_jsonl(run_dir / "judge_results.jsonl"))
    scores = keyed(load_jsonl(run_dir / "deterministic_scores.jsonl"))
    testset = keyed(load_jsonl(Path(args.testset)))

    candidates = []
    for question_id, judge in judges.items():
        if judge.get("status") != "success":
            continue
        correctness = float(judge.get("scores", {}).get("answer_correctness", 1.0))
        if correctness >= args.correctness_threshold:
            continue
        prediction = predictions[question_id]
        result = prediction["result"]
        score = scores[question_id]
        test_row = testset[question_id]
        relevant_ids = set(prediction.get("relevant_chunk_ids", []))
        ranked_chunks = result.get("reranked_chunks", [])
        cited_ids = set(score.get("citation_indices", []))
        review_contexts = [
            {
                "rank": rank,
                "chunk_id": chunk["id"],
                "is_gold": chunk["id"] in relevant_ids,
                "is_cited": rank in cited_ids,
                "text": chunk.get("text", ""),
            }
            for rank, chunk in enumerate(ranked_chunks, start=1)
        ]
        gold_in_top5 = bool(score["retrieval"].get("hit_rate@5"))
        candidates.append({
            "question_id": question_id,
            "article_key": prediction["article_key"],
            "sampling_stratum": stratum(correctness, gold_in_top5),
            "question": prediction["question"],
            "question_type": question_type(prediction["question"]),
            "ground_truth": prediction["ground_truth"],
            "accepted_answers": prediction.get("accepted_answers", []),
            "gold_evidence": test_row.get("evidence", ""),
            "raw_answer": result.get("answer", ""),
            "evaluated_answer": score.get("evaluated_answer", ""),
            "answer_correctness": correctness,
            "answer_relevancy": judge["scores"].get("answer_relevancy"),
            "faithfulness": judge["scores"].get("faithfulness"),
            "token_f1": score["qa"].get("f1"),
            "exact_match": score["qa"].get("exact_match"),
            "gold_in_top5": gold_in_top5,
            "gold_rank": next((item["rank"] for item in review_contexts if item["is_gold"]), None),
            "citation_f1": score["citations"].get("citation_f1"),
            "relevant_chunk_ids": sorted(relevant_ids),
            "generation_contexts": review_contexts,
            "human_review": {
                "primary_error": "pending",
                "secondary_errors": [],
                "model_answer_semantically_correct": None,
                "judge_score_reasonable": None,
                "reviewer_id": "",
                "notes": "",
            },
        })

    sample = select(candidates, args.seed)
    json_path = output_dir / "low_correctness_sample_30.json"
    json_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "selection": "purposive_stratified_low_correctness_audit",
        "seed": args.seed,
        "correctness_threshold_exclusive": args.correctness_threshold,
        "candidate_questions": len(candidates),
        "candidate_articles": len({row["article_key"] for row in candidates}),
        "sample_questions": len(sample),
        "sample_articles": len({row["article_key"] for row in sample}),
        "stratum_targets": STRATUM_TARGETS,
        "sample_strata": dict(Counter(row["sampling_stratum"] for row in sample)),
        "allowed_error_categories": ERROR_CATEGORIES,
        "source_files": {
            "predictions": str(run_dir / "predictions.jsonl"),
            "judge_results": str(run_dir / "judge_results.jsonl"),
            "deterministic_scores": str(run_dir / "deterministic_scores.jsonl"),
            "testset": str(Path(args.testset)),
        },
    }
    (output_dir / "low_correctness_sample_30_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Phase 2 Baseline: Low-Correctness Review Sample\n",
        f"- Selection: `{manifest['selection']}`",
        f"- Threshold: Answer Correctness < `{args.correctness_threshold}`",
        f"- Sample: `{len(sample)}` questions from `{manifest['sample_articles']}` articles",
        "- Purpose: diagnose answer-generation errors; this is not an unbiased performance estimate.",
        "- Fill `human_review` in the JSON file; do not change model outputs or judge scores.\n",
        "## Allowed Categories\n",
        ", ".join(f"`{value}`" for value in ERROR_CATEGORIES) + "\n",
    ]
    for index, row in enumerate(sample, start=1):
        lines.extend([
            f"## {index}. `{row['question_id']}`",
            f"- Article: `{row['article_key']}`",
            f"- Stratum: `{row['sampling_stratum']}`",
            f"- Answer Correctness: `{row['answer_correctness']:.4f}`; token F1: `{row['token_f1']:.4f}`; faithfulness: `{row['faithfulness']:.4f}`",
            f"- Gold in top 5: `{row['gold_in_top5']}`; gold rank: `{row['gold_rank']}`; citation F1: `{row['citation_f1']:.4f}`",
            f"- Question: {row['question']}",
            f"- Question type: `{row['question_type']}`",
            f"- Accepted answers: {json.dumps(row['accepted_answers'], ensure_ascii=False)}",
            f"- Gold evidence: {compact(row['gold_evidence'])}",
            f"- Generated answer: {compact(row['raw_answer'])}",
            "- Review: `pending`\n",
            "### Generation Contexts",
        ])
        for context in row["generation_contexts"]:
            flags = ", ".join(name for name, enabled in (("gold", context["is_gold"]), ("cited", context["is_cited"])) if enabled) or "neither"
            lines.append(f"- Rank {context['rank']} `{context['chunk_id']}` ({flags}): {compact(context['text'])}")
        lines.append("")
    (output_dir / "low_correctness_sample_30_readable.md").write_text("\n".join(lines), encoding="utf-8")
    print(json_path)
    print(output_dir / "low_correctness_sample_30_readable.md")
    print(output_dir / "low_correctness_sample_30_manifest.json")


if __name__ == "__main__":
    main()
