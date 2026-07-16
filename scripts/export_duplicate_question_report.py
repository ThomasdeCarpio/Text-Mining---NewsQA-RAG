#!/usr/bin/env python3
"""Export a readable report of semantic duplicate question clusters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.question_dedup import (
    derive_deduplicated_artifacts,
    validate_cluster_decisions,
)
from src.evaluation.testset import load_testset


DEFAULT_BASE_ROOT = PROJECT_ROOT / "data/evaluation/newsqa_200_11064/final"
DEFAULT_DECISIONS = PROJECT_ROOT / "evaluation/question_dedup/newsqa_200_11064.semantic_clusters.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/newsqa_200_11064/staging/dedup/duplicate_questions_readable.md"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def escaped(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_report(clusters: list[dict]) -> str:
    duplicate_clusters = [row for row in clusters if row["member_count"] > 1]
    duplicate_clusters.sort(key=lambda row: (row["article_id"], row["cluster_id"]))
    duplicate_count = sum(row["member_count"] - 1 for row in duplicate_clusters)
    member_count = sum(row["member_count"] for row in duplicate_clusters)

    lines = [
        "# Semantic Duplicate Questions",
        "",
        "## Summary",
        "",
        f"- Multi-question clusters: {len(duplicate_clusters)}",
        f"- Questions in those clusters: {member_count}",
        f"- Representative questions retained for scoring: {len(duplicate_clusters)}",
        f"- Questions marked as duplicates and removed from scoring: {duplicate_count}",
        "",
        "## Decision criteria",
        "",
        "A question was marked as a duplicate only when all applicable conditions held:",
        "",
        "1. It belongs to the same NewsQA article as its representative.",
        "2. After human-approved resolution, it asks for the same answer-bearing fact.",
        "3. Its reviewed answer and evidence are compatible with the same semantic target.",
        "4. Wording differences are superficial or answer-preserving paraphrases.",
        "5. The decision was made without using RAG retrieval or generation performance.",
        "",
        "The same topic, entity, answer string, or evidence span alone was not sufficient. "
        "Questions with different predicates, WH targets, scopes, dates, counts, locations, "
        "or relations remain separate. Questions were never merged across articles.",
        "",
        "The representative is the lexicographically smallest stable source question ID. "
        "Reviewed accepted answers, evidence spans, and relevant chunk IDs from equivalent "
        "members are retained on that representative.",
        "",
        "## Duplicate clusters",
        "",
    ]

    for index, cluster in enumerate(duplicate_clusters, start=1):
        representative_id = cluster["representative_question_id"]
        representative = next(
            row for row in cluster["member_questions"] if row["question_id"] == representative_id
        )
        lines.extend(
            [
                f"### {index}. `{cluster['article_id']}` / `{cluster['cluster_id']}`",
                "",
                f"Representative: `{representative_id}`",
                "",
                f"> {representative['resolved_question']}",
                "",
                "| Status | Question ID | Original question | Resolved question | Reviewed answer |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for member in cluster["member_questions"]:
            status = "retained" if member["question_id"] == representative_id else "duplicate"
            lines.append(
                "| {status} | `{question_id}` | {original} | {resolved} | {answer} |".format(
                    status=status,
                    question_id=member["question_id"],
                    original=escaped(member["original_question"]),
                    resolved=escaped(member["resolved_question"]),
                    answer=escaped(member["ground_truth"]),
                )
            )
        lines.extend(["", f"Rationale: {cluster['rationale']}", ""])

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clusters",
        type=Path,
        default=None,
    )
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument(
        "--reviewed-original",
        type=Path,
        default=DEFAULT_BASE_ROOT / "testset_reviewed_original.jsonl",
    )
    parser.add_argument(
        "--resolved",
        type=Path,
        default=DEFAULT_BASE_ROOT / "testset_resolved.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()
    if args.clusters:
        clusters = load_jsonl(args.clusters)
    else:
        decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
        reviewed = load_testset(args.reviewed_original)
        resolved = load_testset(args.resolved)
        partition = validate_cluster_decisions(decisions, resolved)
        clusters = derive_deduplicated_artifacts(reviewed, resolved, partition)[4]
    report = build_report(clusters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
