#!/usr/bin/env python3
"""Create the authoritative article-grouped human-review JSON.

The generated JSON preserves immutable source values beside editable reviewed
question, answer, evidence, and decision fields. Dataset finalization consumes
this hierarchical file directly.

Examples:
    python scripts/format_review_queue.py
    python scripts/format_review_queue.py --output /tmp/review_queue.json
    python scripts/format_review_queue.py --format jsonl --output /tmp/review_articles.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "newsqa_200_1000"
    / "staging"
    / "review"
    / "review_queue.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name("review_queue_readable.json")
DEFAULT_ARTICLES = DEFAULT_INPUT.parents[1] / "corpus" / "evaluation_articles.jsonl"
REVIEW_FIELDS = (
    "review_decision",
    "final_clarified_question",
    "review_supporting_quotes",
    "reviewer_id",
    "review_notes",
)


class FormatError(RuntimeError):
    """Raised when a review artifact cannot be safely reformatted."""


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FormatError(f"Review queue does not exist: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise FormatError(f"Invalid JSON on line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise FormatError(f"Line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def _json_list(value: object, field: str, question_id: str) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise FormatError(f"Invalid {field} JSON for {question_id}") from error
        if isinstance(parsed, list):
            return parsed
    raise FormatError(f"{field} must be a JSON list for {question_id}")


def _load_review_csv(path: Path, question_ids: set[str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    reviews: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            question_id = (row.get("question_id") or "").strip()
            if not question_id:
                continue
            if question_id not in question_ids:
                raise FormatError(f"Review CSV contains unknown question ID: {question_id}")
            if question_id in reviews:
                raise FormatError(f"Review CSV contains duplicate question ID: {question_id}")
            reviews[question_id] = {field: row.get(field, "") for field in REVIEW_FIELDS}
    return reviews


def _relative_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def build_readable_queue(
    rows: Iterable[dict],
    csv_reviews: dict[str, dict],
    source_questions: dict[str, dict] | None = None,
) -> dict:
    grouped: dict[str, dict] = {}
    seen_questions: set[str] = set()
    label_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    warning_questions = 0

    for row in rows:
        question_id = str(row.get("question_id") or "").strip()
        article_id = str(row.get("article_id") or "").strip()
        if not question_id or not article_id:
            raise FormatError("Every review row must contain question_id and article_id")
        if question_id in seen_questions:
            raise FormatError(f"Duplicate question ID in review queue: {question_id}")
        seen_questions.add(question_id)

        review_values = {field: row.get(field, "") for field in REVIEW_FIELDS}
        review_values.update(csv_reviews.get(question_id, {}))
        reasons = _json_list(row.get("reason_codes"), "reason_codes", question_id)
        warnings = _json_list(
            row.get("validation_warnings"), "validation_warnings", question_id
        )
        supporting_quotes = _json_list(
            row.get("supporting_context_quotes"),
            "supporting_context_quotes",
            question_id,
        )
        review_quotes = _json_list(
            review_values.get("review_supporting_quotes"),
            "review_supporting_quotes",
            question_id,
        )
        label = str(row.get("llm_label") or "unknown")
        decision = str(review_values.get("review_decision") or "pending")
        label_counts[label] += 1
        decision_counts[decision] += 1
        warning_questions += bool(warnings)

        article = grouped.setdefault(
            article_id,
            {
                "article_id": article_id,
                "source_title": row.get("source_title", ""),
                "questions": [],
            },
        )
        if article["source_title"] != row.get("source_title", ""):
            raise FormatError(f"Conflicting source titles for article {article_id}")
        article["questions"].append(
            {
                "question_id": question_id,
                "comparison": {
                    "original_question": row.get("original_question", ""),
                    "candidate_clarified_question": row.get(
                        "candidate_clarified_question", ""
                    ),
                    "reviewed_candidate_clarified_question": (
                        review_values.get("final_clarified_question", "")
                        or (
                            row.get("candidate_clarified_question", "")
                            if decision == "approve"
                            else ""
                        )
                    ),
                    "final_clarified_question": review_values.get(
                        "final_clarified_question", ""
                    ),
                },
                "answer_and_evidence": {
                    "source_expected_answer": row.get("ground_truth", ""),
                    "expected_answer": row.get("ground_truth", ""),
                    "accepted_answers": [row.get("ground_truth", "")],
                    "source_evidence_text": row.get("evidence_text", ""),
                    "evidence_text": row.get("evidence_text", ""),
                    "source_evidence_spans": (
                        source_questions.get(question_id, {}).get("evidence_spans", [])
                        if source_questions
                        else []
                    ),
                    "evidence_spans": (
                        source_questions.get(question_id, {}).get("evidence_spans", [])
                        if source_questions
                        else []
                    ),
                    "answer_modified": False,
                    "answer_review_notes": "",
                    "supporting_context_quotes": supporting_quotes,
                },
                "llm_assessment": {
                    "label": label,
                    "reason_codes": reasons,
                    "confidence": row.get("llm_confidence"),
                    "rationale": row.get("llm_rationale", ""),
                    "validation_warnings": warnings,
                },
                "human_review": {
                    "decision": decision,
                    "reviewer_id": review_values.get("reviewer_id", ""),
                    "supporting_quotes": review_quotes,
                    "notes": review_values.get("review_notes", ""),
                },
            }
        )

    articles = sorted(grouped.values(), key=lambda item: item["article_id"])
    for article in articles:
        article["questions"].sort(key=lambda item: item["question_id"])
        article["question_count"] = len(article["questions"])

    return {
        "schema_version": "2.0",
        "summary": {
            "articles": len(articles),
            "questions": len(seen_questions),
            "questions_with_validation_warnings": warning_questions,
            "llm_labels": dict(sorted(label_counts.items())),
            "review_decisions": dict(sorted(decision_counts.items())),
        },
        "articles": articles,
    }


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, articles: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for article in articles:
            handle.write(json.dumps(article, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Flat review_queue.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Formatted output path")
    parser.add_argument(
        "--articles",
        default=str(DEFAULT_ARTICLES),
        help="Evaluation articles JSONL used to preserve source evidence spans",
    )
    parser.add_argument(
        "--review-csv",
        default=None,
        help="Optional review CSV to merge; defaults to review_queue.csv beside the input",
    )
    parser.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="json",
        help="Pretty hierarchical JSON or one compact article object per JSONL line",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing authoritative review JSON and discard direct edits",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    articles_path = Path(args.articles).resolve()
    review_csv = (
        Path(args.review_csv).resolve()
        if args.review_csv
        else input_path.with_name("review_queue.csv")
    )
    try:
        if output_path.exists() and not args.overwrite:
            raise FormatError(
                f"Authoritative review file already exists: {output_path}; "
                "pass --overwrite only to intentionally reset human edits"
            )
        rows = _load_jsonl(input_path)
        article_rows = _load_jsonl(articles_path)
        source_questions = {
            question["question_id"]: question
            for article in article_rows
            for question in article.get("questions", [])
        }
        question_ids = {str(row.get("question_id") or "").strip() for row in rows}
        csv_reviews = _load_review_csv(review_csv, question_ids)
        document = build_readable_queue(rows, csv_reviews, source_questions)
        document["source"] = {
            "review_queue": _relative_display(input_path),
            "review_csv": _relative_display(review_csv) if review_csv.exists() else None,
        }
        document["approval_note"] = (
            "Authoritative review artifact: edit reviewed fields in this file; "
            "source_* fields must remain unchanged."
        )
        if args.format == "json":
            _write_json(output_path, document)
        else:
            _write_jsonl(output_path, document["articles"])
    except FormatError as error:
        print(f"ERROR: {error}")
        return 2

    summary = document["summary"]
    print(
        f"Wrote {summary['questions']} questions grouped into "
        f"{summary['articles']} articles: {output_path}"
    )
    if review_csv.exists():
        print(f"Merged review decisions from: {review_csv}")
    else:
        print("No review CSV found; review fields came from the JSONL queue.")
    print("This JSON is now the authoritative input to review-status and finalize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
