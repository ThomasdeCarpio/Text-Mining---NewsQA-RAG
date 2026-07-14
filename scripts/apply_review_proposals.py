#!/usr/bin/env python3
"""Validate and atomically apply one Codex proposal batch to the review queue."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.question_review import FULL_REVIEW_ISSUES, FULL_REVIEW_LABELS
from src.evaluation.testset import DatasetBuildError, sha256_file


DEFAULT_ROOT = PROJECT_ROOT / "data" / "evaluation" / "newsqa_200_1000"
DEFAULT_QUEUE = DEFAULT_ROOT / "staging" / "review" / "review_queue_readable.json"


def _load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DatasetBuildError(f"Expected a JSON object in {path}")
    return value


def _write_json_atomic(path: str | Path, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _validate_spans(spans: object, context: str, question_id: str) -> list[dict]:
    if not isinstance(spans, list) or not spans:
        raise DatasetBuildError(f"Missing evidence spans for {question_id}")
    validated = []
    for span in spans:
        if not isinstance(span, dict):
            raise DatasetBuildError(f"Invalid evidence span for {question_id}")
        try:
            start = int(span["start"])
            end = int(span["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetBuildError(f"Invalid evidence offsets for {question_id}") from error
        if not 0 <= start < end <= len(context):
            raise DatasetBuildError(f"Evidence is out of range for {question_id}")
        text = context[start:end]
        if span.get("text", text) != text:
            raise DatasetBuildError(f"Evidence text mismatch for {question_id}")
        validated.append({"start": start, "end": end, "text": text})
    return validated


def _validate_quote(
    quote: str,
    context: str,
    sensitive_spans: list[dict],
    question_id: str,
) -> None:
    if not quote:
        raise DatasetBuildError(f"Empty supporting quote for {question_id}")
    position = context.find(quote)
    while position >= 0:
        end = position + len(quote)
        if not any(
            position < span["end"] and span["start"] < end
            for span in sensitive_spans
        ):
            return
        position = context.find(quote, position + 1)
    raise DatasetBuildError(
        f"Supporting quote is absent or overlaps answer evidence for {question_id}: {quote!r}"
    )


def _question_maps(packet: dict, queue: dict) -> tuple[dict, dict]:
    packet_questions: dict[str, tuple[dict, dict]] = {}
    for article in packet.get("articles", []):
        for question in article.get("questions", []):
            question_id = question.get("question_id")
            if not question_id or question_id in packet_questions:
                raise DatasetBuildError(f"Duplicate or missing packet question ID {question_id!r}")
            packet_questions[question_id] = (article, question)

    queue_questions: dict[str, dict] = {}
    for article in queue.get("articles", []):
        for question in article.get("questions", []):
            question_id = question.get("question_id")
            if not question_id or question_id in queue_questions:
                raise DatasetBuildError(f"Duplicate or missing queue question ID {question_id!r}")
            queue_questions[question_id] = question
    return packet_questions, queue_questions


def apply_proposals(
    packet_path: str | Path,
    proposal_path: str | Path,
    queue_path: str | Path = DEFAULT_QUEUE,
    audit_path: str | Path | None = None,
) -> dict:
    packet = _load_json(packet_path)
    proposals = _load_json(proposal_path)
    queue = _load_json(queue_path)
    if queue.get("review_mode") != "full_codex_human":
        raise DatasetBuildError("Proposal batches require a full Codex/human review queue")
    packet_id = str(packet.get("packet_id") or "").strip()
    if proposals.get("packet_id") != packet_id:
        raise DatasetBuildError("Proposal packet_id does not match the packet")
    packet_hash = sha256_file(packet_path)
    if proposals.get("input_sha256") != packet_hash:
        raise DatasetBuildError("Proposal input_sha256 does not match the packet")
    actor = proposals.get("actor") or {}
    tool = str(actor.get("tool") or "").strip()
    model = str(actor.get("model") or "").strip()
    created_at = str(proposals.get("created_at") or "").strip()
    audit_id = str(proposals.get("audit_id") or "").strip()
    if not all((tool, model, created_at, audit_id)):
        raise DatasetBuildError("Proposal batch is missing actor, timestamp, or audit_id")
    output_audit = (
        Path(audit_path)
        if audit_path
        else Path(queue_path).parent / "audits" / f"{audit_id}.json"
    )
    if output_audit.exists():
        raise DatasetBuildError(f"Audit already exists and is immutable: {output_audit}")

    packet_questions, queue_questions = _question_maps(packet, queue)
    changes = proposals.get("changes")
    if not isinstance(changes, list):
        raise DatasetBuildError("Proposal changes must be a list")
    changes_by_id: dict[str, dict] = {}
    for change in changes:
        question_id = str(change.get("question_id") or "").strip()
        if not question_id or question_id in changes_by_id:
            raise DatasetBuildError(f"Duplicate or missing proposal question ID {question_id!r}")
        changes_by_id[question_id] = change
    if set(changes_by_id) != set(packet_questions):
        missing = sorted(set(packet_questions) - set(changes_by_id))[:5]
        extra = sorted(set(changes_by_id) - set(packet_questions))[:5]
        raise DatasetBuildError(
            f"Proposal coverage must exactly match {packet_id}; missing={missing}, extra={extra}"
        )

    queue_before_sha256 = sha256_file(queue_path)
    for question_id, change in changes_by_id.items():
        article, packet_question = packet_questions[question_id]
        row = queue_questions.get(question_id)
        if row is None:
            raise DatasetBuildError(f"Packet question is missing from queue: {question_id}")
        original_source = {
            "original_question": row["comparison"]["original_question"],
            "source_expected_answer": row["answer_and_evidence"]["source_expected_answer"],
            "source_evidence_spans": deepcopy(
                row["answer_and_evidence"]["source_evidence_spans"]
            ),
            "human_review": deepcopy(row["human_review"]),
        }
        if row["human_review"].get("decision") != "pending":
            raise DatasetBuildError(f"Human review is already resolved for {question_id}")

        label = str(change.get("label") or "").strip()
        issue_codes = change.get("issue_codes") or []
        rationale = str(change.get("rationale") or "").strip()
        candidate = str(change.get("candidate_clarified_question") or "").strip()
        quotes = change.get("proposed_supporting_quotes") or []
        if label not in FULL_REVIEW_LABELS - {"pending"}:
            raise DatasetBuildError(f"Invalid proposal label for {question_id}: {label!r}")
        if not isinstance(issue_codes, list) or not set(issue_codes) <= FULL_REVIEW_ISSUES:
            raise DatasetBuildError(f"Invalid issue codes for {question_id}: {issue_codes!r}")
        if not rationale:
            raise DatasetBuildError(f"Missing proposal rationale for {question_id}")
        if not isinstance(quotes, list) or not all(isinstance(quote, str) for quote in quotes):
            raise DatasetBuildError(f"Invalid supporting quotes for {question_id}")
        if label == "standalone" and (candidate or quotes):
            raise DatasetBuildError(
                f"Standalone proposal must not add a clarification for {question_id}"
            )
        if label == "non_standalone" and (not candidate or not quotes):
            raise DatasetBuildError(
                f"Non-standalone proposal requires a candidate and quotes for {question_id}"
            )

        context = article["context"]
        answer = row["answer_and_evidence"]
        answer_update = change.get("answer_update")
        if answer_update is not None:
            if not isinstance(answer_update, dict):
                raise DatasetBuildError(f"answer_update must be an object for {question_id}")
            reviewed_answer = str(answer_update.get("expected_answer") or "").strip()
            accepted_answers = answer_update.get("accepted_answers") or []
            reviewed_spans = _validate_spans(
                answer_update.get("evidence_spans"), context, question_id
            )
            notes = str(answer_update.get("answer_review_notes") or "").strip()
            if not reviewed_answer or not notes:
                raise DatasetBuildError(
                    f"Answer correction requires an answer and notes for {question_id}"
                )
            if not isinstance(accepted_answers, list) or not all(
                isinstance(value, str) and value.strip() for value in accepted_answers
            ):
                raise DatasetBuildError(f"Invalid accepted answers for {question_id}")
            if reviewed_answer not in accepted_answers:
                accepted_answers = [reviewed_answer, *accepted_answers]
            answer.update(
                {
                    "expected_answer": reviewed_answer,
                    "accepted_answers": accepted_answers,
                    "evidence_spans": reviewed_spans,
                    "evidence_text": " | ".join(span["text"] for span in reviewed_spans),
                    "answer_modified": True,
                    "answer_review_notes": notes,
                }
            )
        else:
            answer.update(
                {
                    "expected_answer": answer["source_expected_answer"],
                    "accepted_answers": [answer["source_expected_answer"]],
                    "evidence_spans": deepcopy(answer["source_evidence_spans"]),
                    "evidence_text": " | ".join(
                        span["text"] for span in answer["source_evidence_spans"]
                    ),
                    "answer_modified": False,
                    "answer_review_notes": "",
                }
            )

        sensitive_spans = [
            *answer["source_evidence_spans"],
            *answer["evidence_spans"],
        ]
        for quote in quotes:
            _validate_quote(quote, context, sensitive_spans, question_id)
        accepted_normalized = {
            " ".join(value.lower().split()).strip(".,!?;:\"'")
            for value in answer["accepted_answers"]
        }
        normalized_candidate = " ".join(candidate.lower().split())
        if any(
            len(value) >= 3 and value in normalized_candidate
            for value in accepted_normalized
        ):
            raise DatasetBuildError(f"Candidate clarification leaks answer for {question_id}")

        row["comparison"]["candidate_clarified_question"] = candidate
        row["codex_assessment"] = {
            "label": label,
            "issue_codes": issue_codes,
            "rationale": rationale,
            "proposed_supporting_quotes": quotes,
            "proposal": {
                "status": "proposed",
                "tool": tool,
                "model": model,
                "batch_id": packet_id,
                "created_at": created_at,
            },
        }
        if original_source != {
            "original_question": row["comparison"]["original_question"],
            "source_expected_answer": row["answer_and_evidence"]["source_expected_answer"],
            "source_evidence_spans": row["answer_and_evidence"]["source_evidence_spans"],
            "human_review": row["human_review"],
        }:
            raise DatasetBuildError(f"Protected review fields changed for {question_id}")

    _write_json_atomic(queue_path, queue)
    queue_after_sha256 = sha256_file(queue_path)
    audit = {
        "schema_version": "3.0",
        "audit_id": audit_id,
        "audit_type": "codex_proposal",
        "packet_id": packet_id,
        "created_at": created_at,
        "actor": {"tool": tool, "model": model},
        "input_sha256": packet_hash,
        "queue_before_sha256": queue_before_sha256,
        "queue_after_sha256": queue_after_sha256,
        "question_ids": sorted(changes_by_id),
        "changes": changes,
    }
    _write_json_atomic(output_audit, audit)
    return {
        "packet_id": packet_id,
        "question_count": len(changes_by_id),
        "queue": str(queue_path),
        "queue_sha256": queue_after_sha256,
        "audit": str(output_audit),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--audit", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = apply_proposals(args.packet, args.proposals, args.queue, args.audit)
    except DatasetBuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
