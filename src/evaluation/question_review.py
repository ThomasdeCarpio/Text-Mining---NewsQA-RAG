"""LLM-assisted standalone-question triage and human approval handling."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from rank_bm25 import BM25Okapi

from src.evaluation.testset import (
    DatasetBuildError,
    canonical_json,
    iter_jsonl,
    save_jsonl,
    sha256_text,
)


PROMPT_VERSION = "standalone-triage-v3"
ALLOWED_LABELS = {"standalone", "non_standalone", "uncertain"}
ALLOWED_REASONS = {
    "missing_subject",
    "unresolved_coreference",
    "underspecified_event",
    "missing_time",
    "missing_location",
    "generic_reference",
    "multiple_corpus_matches",
}
REVIEW_DECISIONS = {
    "pending",
    "approve",
    "edit",
    "mark_standalone",
    "exclude",
    "needs_adjudication",
}
FULL_REVIEW_LABELS = {
    "pending",
    "standalone",
    "non_standalone",
    "invalid",
    "uncertain",
}
FULL_REVIEW_ISSUES = {
    *ALLOWED_REASONS,
    "truncated_answer",
    "wrong_answer",
    "wrong_evidence",
    "yes_no_answer_mismatch",
    "malformed_question",
    "unanswerable_from_article",
    "irreparable_semantic_mismatch",
    "ambiguous_without_unique_gold",
}
EXCLUSION_REASONS = {
    "unanswerable_from_article",
    "irreparable_semantic_mismatch",
    "ambiguous_without_unique_gold",
}


class TriageClient(Protocol):
    model: str

    def classify(self, request: dict) -> list[dict]: ...


@dataclass(frozen=True)
class GeminiTriageClient:
    """Minimal native Gemini generateContent client with JSON output."""

    api_key: str
    model: str = "gemini-3.1-flash-lite"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_seconds: float = 120.0
    max_retries: int = 5

    def classify(self, request: dict) -> list[dict]:
        import httpx

        url = f"{self.base_url}/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": _system_prompt()}]},
            "contents": [{"role": "user", "parts": [{"text": canonical_json(request)}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": _response_schema(),
            },
        }
        for attempt in range(self.max_retries):
            response = httpx.post(
                url,
                params={"key": self.api_key},
                json=body,
                timeout=self.timeout_seconds,
            )
            if response.status_code < 400:
                payload = response.json()
                try:
                    text = payload["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = json.loads(text)
                    return parsed["questions"]
                except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                    raise DatasetBuildError(f"Malformed Gemini response: {error}") from error
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == self.max_retries - 1:
                raise DatasetBuildError(
                    f"Gemini request failed with HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            retry_after = response.headers.get("retry-after")
            delay = float(retry_after) if retry_after else min(2**attempt, 30)
            time.sleep(delay + random.random())
        raise DatasetBuildError("Gemini request exhausted retries")


def _system_prompt() -> str:
    return """You are annotating questions for a multi-article news retrieval benchmark.
Judge whether each ORIGINAL question identifies one intended subject or event without assuming
that the source article is already known. Use the redacted source context only to identify what
context is missing and to propose the smallest clarification. Candidate articles show competing
interpretations. Each question includes its expected answer. Use that answer only to ensure the
clarified question preserves the original meaning and can be answered by the same answer. Never
insert or paraphrase the expected answer in the clarification, and never change the answer type.

Return one result for every question_id. Use standalone when the wording is sufficiently specific,
non_standalone when essential context is missing, and uncertain when the decision requires human
judgment. For non_standalone or uncertain questions, propose a natural minimally clarified question
and quote the exact non-answer source phrase supporting every added detail. A supporting quote must
never be the answer or overlap redacted evidence. Use only the allowed
reason codes supplied by the schema. Keep rationales short."""


def _response_schema() -> dict:
    return {
        "type": "OBJECT",
        "required": ["questions"],
        "properties": {
            "questions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "required": [
                        "question_id",
                        "standalone_label",
                        "reason_codes",
                        "rationale",
                        "candidate_clarified_question",
                        "added_context",
                        "confidence",
                    ],
                    "properties": {
                        "question_id": {"type": "STRING"},
                        "standalone_label": {
                            "type": "STRING",
                            "enum": sorted(ALLOWED_LABELS),
                        },
                        "reason_codes": {
                            "type": "ARRAY",
                            "items": {"type": "STRING", "enum": sorted(ALLOWED_REASONS)},
                        },
                        "rationale": {"type": "STRING"},
                        "candidate_clarified_question": {
                            "type": "STRING",
                            "nullable": True,
                        },
                        "added_context": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "required": ["text", "supporting_context_quote"],
                                "properties": {
                                    "text": {"type": "STRING"},
                                    "supporting_context_quote": {"type": "STRING"},
                                },
                            },
                        },
                        "confidence": {"type": "NUMBER"},
                    },
                },
            }
        },
    }


class ArticleCandidateIndex:
    """Article-level BM25 used only to expose competing interpretations to triage."""

    def __init__(self, articles: Sequence[dict]):
        self.articles = list(articles)
        self.tokenized = [self._tokens(item["context"]) for item in self.articles]
        self.index = BM25Okapi(self.tokenized)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.strip(".,!?;:()[]{}\"'").lower() for token in text.split() if token]

    def candidates(self, question: str, source_article_id: str, top_k: int = 3) -> list[dict]:
        scores = self.index.get_scores(self._tokens(question))
        ranked = scores.argsort()[::-1]
        results: list[dict] = []
        for position in ranked:
            article = self.articles[int(position)]
            if article["article_id"] == source_article_id:
                continue
            results.append(
                {
                    "article_id": article["article_id"],
                    "title": article["metadata"].get("title", ""),
                    "snippet": _best_snippet(article["context"], question),
                    "bm25_score": round(float(scores[int(position)]), 6),
                }
            )
            if len(results) == top_k:
                break
        return results


def _best_snippet(context: str, question: str, max_chars: int = 420) -> str:
    query_terms = {token.lower().strip(".,!?;:") for token in question.split() if len(token) > 3}
    paragraphs = [item.strip() for item in context.split("\n\n") if item.strip()]
    if not paragraphs:
        return context[:max_chars]
    best = max(
        paragraphs,
        key=lambda item: sum(term in item.lower() for term in query_terms),
    )
    return best[:max_chars]


def _redacted_context(
    context: str,
    target_spans: Sequence[dict],
    sensitive_spans: Sequence[dict] | None = None,
    window: int = 450,
) -> str:
    """Build a local window while hiding every answer span from the article batch."""

    start = max(0, min(item["start"] for item in target_spans) - window)
    end = min(len(context), max(item["end"] for item in target_spans) + window)
    value = context[start:end]
    redactions = sensitive_spans or target_spans
    for span in sorted(redactions, key=lambda item: item["start"], reverse=True):
        local_start = max(span["start"], start) - start
        local_end = min(span["end"], end) - start
        if local_start < local_end:
            value = value[:local_start] + "[REDACTED_EVIDENCE]" + value[local_end:]
    return value


def build_article_request(
    article: dict,
    candidates: ArticleCandidateIndex,
    questions_subset: Sequence[dict] | None = None,
) -> dict:
    sensitive_spans = [
        span
        for article_question in article["questions"]
        for span in article_question["evidence_spans"]
    ]
    questions = []
    for question in questions_subset or article["questions"]:
        questions.append(
            {
                "question_id": question["question_id"],
                "original_question": question["question"],
                "expected_answer": question["ground_truth"],
                "redacted_source_context": _redacted_context(
                    article["context"], question["evidence_spans"], sensitive_spans
                ),
                "competing_articles": candidates.candidates(
                    question["question"], article["article_id"]
                ),
            }
        )
    return {
        "prompt_version": PROMPT_VERSION,
        "article": {"article_id": article["article_id"]},
        "questions": questions,
    }


def _answer_leaks(answer: str, clarified: str) -> bool:
    normalized_answer = " ".join(answer.lower().split()).strip(".,!?;:\"'")
    normalized_question = " ".join(clarified.lower().split())
    return len(normalized_answer) >= 3 and normalized_answer in normalized_question


def _quote_is_supported(article: dict, quote: str, question: dict) -> bool:
    if not quote:
        return False
    context = article["context"]
    position = context.find(quote)
    while position >= 0:
        end = position + len(quote)
        if not any(
            position < span["end"] and span["start"] < end
            for span in question["evidence_spans"]
        ):
            return True
        position = context.find(quote, position + 1)
    return False


def validate_predictions(article: dict, predictions: Sequence[dict]) -> list[dict]:
    expected = {item["question_id"]: item for item in article["questions"]}
    seen: set[str] = set()
    validated: list[dict] = []
    for prediction in predictions:
        question_id = str(prediction.get("question_id") or "")
        if question_id not in expected or question_id in seen:
            raise DatasetBuildError(f"Unexpected or duplicate triage question ID {question_id}")
        seen.add(question_id)
        label = prediction.get("standalone_label")
        reasons = prediction.get("reason_codes") or []
        if label not in ALLOWED_LABELS:
            raise DatasetBuildError(f"Invalid standalone label for {question_id}: {label}")
        if any(reason not in ALLOWED_REASONS for reason in reasons):
            raise DatasetBuildError(f"Invalid ambiguity reason for {question_id}")
        clarified = str(prediction.get("candidate_clarified_question") or "").strip()
        added_context = prediction.get("added_context") or []
        if label == "standalone":
            clarified = ""
            added_context = []
        elif not clarified:
            raise DatasetBuildError(f"Missing clarification for {question_id}")
        question = expected[question_id]
        validation_warnings: list[str] = []
        if clarified and _answer_leaks(question["ground_truth"], clarified):
            validation_warnings.append("candidate_clarification_contains_answer")
        for addition in added_context:
            quote = str(addition.get("supporting_context_quote") or "")
            if not _quote_is_supported(article, quote, question):
                validation_warnings.append(f"unsupported_or_evidence_context_quote:{quote}")
        confidence = float(prediction.get("confidence", 0.0))
        if not 0 <= confidence <= 1:
            raise DatasetBuildError(f"Invalid confidence for {question_id}: {confidence}")
        if validation_warnings:
            label = "uncertain"
        validated.append(
            {
                **prediction,
                "question_id": question_id,
                "article_id": article["article_id"],
                "standalone_label": label,
                "candidate_clarified_question": clarified or None,
                "added_context": added_context,
                "confidence": confidence,
                "prompt_version": PROMPT_VERSION,
                "validation_warnings": validation_warnings,
            }
        )
    if seen != set(expected):
        missing = sorted(set(expected) - seen)[:5]
        raise DatasetBuildError(f"Triage response omitted questions: {missing}")
    return sorted(validated, key=lambda item: item["question_id"])


def run_triage(
    evaluation_articles: Sequence[dict],
    distractor_articles: Sequence[dict],
    client: TriageClient,
    predictions_path: str | Path,
    requests_per_minute: int = 5,
    max_questions_per_request: int = 25,
) -> list[dict]:
    """Classify every question with article-level requests and resumable caching."""

    path = Path(predictions_path)
    cached = {item["article_id"]: item for item in iter_jsonl(path)} if path.exists() else {}
    candidate_index = ArticleCandidateIndex([*evaluation_articles, *distractor_articles])
    delay = 60.0 / max(requests_per_minute, 1)
    output: list[dict] = []
    persisted = dict(cached)
    last_request_at = 0.0
    for index, article in enumerate(evaluation_articles):
        article_questions = article["questions"]
        requests = [
            build_article_request(
                article,
                candidate_index,
                article_questions[start : start + max_questions_per_request],
            )
            for start in range(0, len(article_questions), max_questions_per_request)
        ]
        input_hash = sha256_text(canonical_json(requests))
        existing = cached.get(article["article_id"])
        if (
            existing
            and existing.get("input_sha256") == input_hash
            and existing.get("model") == client.model
        ):
            output.append(existing)
            continue
        raw_predictions: list[dict] = []
        for request in requests:
            elapsed = time.monotonic() - last_request_at
            if last_request_at and elapsed < delay:
                time.sleep(delay - elapsed)
            raw_predictions.extend(client.classify(request))
            last_request_at = time.monotonic()
        predictions = validate_predictions(article, raw_predictions)
        record = {
            "article_id": article["article_id"],
            "input_sha256": input_hash,
            "model": client.model,
            "prompt_version": PROMPT_VERSION,
            "predictions": predictions,
        }
        output.append(record)
        persisted[article["article_id"]] = record
        save_jsonl(sorted(persisted.values(), key=lambda item: item["article_id"]), path)
    output.sort(key=lambda item: item["article_id"])
    save_jsonl(output, path)
    return output


REVIEW_COLUMNS = [
    "question_id",
    "article_id",
    "source_title",
    "original_question",
    "ground_truth",
    "evidence_text",
    "llm_label",
    "reason_codes",
    "llm_confidence",
    "llm_rationale",
    "validation_warnings",
    "candidate_clarified_question",
    "supporting_context_quotes",
    "review_decision",
    "final_clarified_question",
    "review_supporting_quotes",
    "reviewer_id",
    "review_notes",
]


def create_review_queue(
    evaluation_articles: Sequence[dict],
    triage_records: Sequence[dict],
    jsonl_path: str | Path,
    csv_path: str | Path,
) -> list[dict]:
    existing_reviews: dict[str, dict] = {}
    existing_csv = Path(csv_path)
    if existing_csv.exists():
        with open(existing_csv, encoding="utf-8-sig", newline="") as handle:
            existing_reviews = {
                row["question_id"]: row for row in csv.DictReader(handle) if row.get("question_id")
            }
    articles = {item["article_id"]: item for item in evaluation_articles}
    questions = {
        question["question_id"]: (article, question)
        for article in evaluation_articles
        for question in article["questions"]
    }
    queue: list[dict] = []
    for record in triage_records:
        if record["article_id"] not in articles:
            raise DatasetBuildError(f"Triage references unknown article {record['article_id']}")
        for prediction in record["predictions"]:
            if prediction["standalone_label"] == "standalone":
                continue
            article, question = questions[prediction["question_id"]]
            quotes = [item["supporting_context_quote"] for item in prediction["added_context"]]
            item = {
                    "question_id": question["question_id"],
                    "article_id": article["article_id"],
                    "source_title": article["metadata"].get("title", ""),
                    "original_question": question["question"],
                    "ground_truth": question["ground_truth"],
                    "evidence_text": " | ".join(item["text"] for item in question["evidence_spans"]),
                    "llm_label": prediction["standalone_label"],
                    "reason_codes": prediction["reason_codes"],
                    "llm_confidence": prediction["confidence"],
                    "llm_rationale": prediction.get("rationale", ""),
                    "validation_warnings": prediction.get("validation_warnings", []),
                    "candidate_clarified_question": prediction["candidate_clarified_question"],
                    "supporting_context_quotes": quotes,
                    "review_decision": "pending",
                    "final_clarified_question": "",
                    "review_supporting_quotes": [],
                    "reviewer_id": "",
                    "review_notes": "",
                }
            existing = existing_reviews.get(question["question_id"])
            if (
                existing
                and existing.get("original_question") == question["question"]
                and existing.get("candidate_clarified_question")
                == (prediction["candidate_clarified_question"] or "")
            ):
                item.update(
                    {
                        "review_decision": existing.get("review_decision", "pending"),
                        "final_clarified_question": existing.get("final_clarified_question", ""),
                        "review_supporting_quotes": _parse_json_list(
                            existing.get("review_supporting_quotes", ""),
                            "review_supporting_quotes",
                            question["question_id"],
                        ),
                        "reviewer_id": existing.get("reviewer_id", ""),
                        "review_notes": existing.get("review_notes", ""),
                    }
                )
            queue.append(item)
    queue.sort(key=lambda item: item["question_id"])
    save_jsonl(queue, jsonl_path)
    csv_output = Path(csv_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_output, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for item in queue:
            row = dict(item)
            for field in (
                "reason_codes",
                "validation_warnings",
                "supporting_context_quotes",
                "review_supporting_quotes",
            ):
                row[field] = canonical_json(row[field])
            writer.writerow(row)
    return queue


def create_full_review_document(
    evaluation_articles: Sequence[dict],
    proposer_tool: str = "codex-cli",
    proposer_model: str = "sol-5.6",
) -> dict:
    """Create an authoritative queue containing every selected evaluation question."""

    articles: list[dict] = []
    seen_questions: set[str] = set()
    for article in sorted(evaluation_articles, key=lambda item: item["article_id"]):
        questions: list[dict] = []
        for question in sorted(article["questions"], key=lambda item: item["question_id"]):
            question_id = question["question_id"]
            if question_id in seen_questions:
                raise DatasetBuildError(f"Duplicate question ID {question_id}")
            seen_questions.add(question_id)
            source_spans = question["evidence_spans"]
            source_answer = question["ground_truth"]
            questions.append(
                {
                    "question_id": question_id,
                    "comparison": {
                        "original_question": question["question"],
                        "candidate_clarified_question": "",
                        "final_clarified_question": "",
                    },
                    "answer_and_evidence": {
                        "source_expected_answer": source_answer,
                        "expected_answer": source_answer,
                        "accepted_answers": [source_answer],
                        "source_evidence_spans": source_spans,
                        "evidence_spans": source_spans,
                        "evidence_text": " | ".join(
                            span["text"] for span in source_spans
                        ),
                        "answer_modified": False,
                        "answer_review_notes": "",
                        "supporting_context_quotes": [],
                    },
                    "codex_assessment": {
                        "label": "pending",
                        "issue_codes": [],
                        "rationale": "",
                        "proposed_supporting_quotes": [],
                        "proposal": {
                            "status": "pending",
                            "tool": proposer_tool,
                            "model": proposer_model,
                            "batch_id": "",
                            "created_at": "",
                        },
                    },
                    "human_review": {
                        "decision": "pending",
                        "reviewer_id": "",
                        "supporting_quotes": [],
                        "notes": "",
                    },
                }
            )
        articles.append(
            {
                "article_id": article["article_id"],
                "source_title": article.get("metadata", {}).get("title", ""),
                "questions": questions,
            }
        )
    return {
        "schema_version": "3.0",
        "review_mode": "full_codex_human",
        "instructions": {
            "scope": "Review every question and answer against its complete source article.",
            "proposal_labels": sorted(FULL_REVIEW_LABELS),
            "issue_codes": sorted(FULL_REVIEW_ISSUES),
            "human_decisions": sorted(REVIEW_DECISIONS),
            "exclusion_reasons": sorted(EXCLUSION_REASONS),
        },
        "summary": {
            "article_count": len(articles),
            "question_count": len(seen_questions),
            "proposer_tool": proposer_tool,
            "proposer_model": proposer_model,
        },
        "articles": articles,
    }


def build_full_review_packets(
    evaluation_articles: Sequence[dict],
    distractor_articles: Sequence[dict],
    review_document: dict,
    max_articles: int = 20,
    max_questions: int = 150,
    competing_top_k: int = 5,
) -> list[dict]:
    """Build deterministic, bounded Codex review packets without making model calls."""

    if max_articles <= 0 or max_questions <= 0 or competing_top_k <= 0:
        raise DatasetBuildError("Review packet limits must be positive")
    queue_articles = {
        item["article_id"]: item for item in review_document.get("articles", [])
    }
    expected_ids = {item["article_id"] for item in evaluation_articles}
    if set(queue_articles) != expected_ids:
        raise DatasetBuildError("Review document article coverage does not match evaluation corpus")

    candidate_index = ArticleCandidateIndex([*evaluation_articles, *distractor_articles])
    packet_articles: list[dict] = []
    packets: list[dict] = []
    packet_questions = 0

    def flush() -> None:
        nonlocal packet_articles, packet_questions
        if not packet_articles:
            return
        packet_number = len(packets) + 1
        packets.append(
            {
                "schema_version": "3.0",
                "packet_id": f"review_{packet_number:03d}",
                "purpose": "Codex proposal preparation; human approval remains mandatory.",
                "instructions": {
                    "review_all_questions": True,
                    "preserve_source_fields": True,
                    "do_not_use_answer_in_clarification": True,
                    "allowed_labels": sorted(FULL_REVIEW_LABELS - {"pending"}),
                    "allowed_issue_codes": sorted(FULL_REVIEW_ISSUES),
                },
                "article_count": len(packet_articles),
                "question_count": packet_questions,
                "articles": packet_articles,
            }
        )
        packet_articles = []
        packet_questions = 0

    for article in sorted(evaluation_articles, key=lambda item: item["article_id"]):
        queued = queue_articles[article["article_id"]]
        queue_by_id = {
            item["question_id"]: item for item in queued.get("questions", [])
        }
        source_by_id = {
            item["question_id"]: item for item in article["questions"]
        }
        if set(queue_by_id) != set(source_by_id):
            raise DatasetBuildError(
                f"Review question coverage mismatch for {article['article_id']}"
            )
        count = len(source_by_id)
        if count > max_questions:
            raise DatasetBuildError(
                f"Article {article['article_id']} has {count} questions, exceeding "
                f"the packet limit {max_questions}"
            )
        if packet_articles and (
            len(packet_articles) >= max_articles
            or packet_questions + count > max_questions
        ):
            flush()
        questions = []
        for question_id in sorted(source_by_id):
            source = source_by_id[question_id]
            questions.append(
                {
                    "question_id": question_id,
                    "original_question": source["question"],
                    "source_expected_answer": source["ground_truth"],
                    "source_evidence_spans": source["evidence_spans"],
                    "source_evidence_text": " | ".join(
                        span["text"] for span in source["evidence_spans"]
                    ),
                    "current_review_row": queue_by_id[question_id],
                    "competing_articles": candidate_index.candidates(
                        source["question"], article["article_id"], competing_top_k
                    ),
                }
            )
        packet_articles.append(
            {
                "article_id": article["article_id"],
                "title": article.get("metadata", {}).get("title", ""),
                "context": article["context"],
                "questions": questions,
            }
        )
        packet_questions += count
    flush()
    return packets


def _review_rows(review_path: str | Path) -> list[dict]:
    path = Path(review_path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or not isinstance(document.get("articles"), list):
        raise DatasetBuildError("Hierarchical review file must contain an articles list")

    rows: list[dict] = []
    for article in document["articles"]:
        article_id = str(article.get("article_id") or "").strip()
        for item in article.get("questions") or []:
            comparison = item.get("comparison") or {}
            answer = item.get("answer_and_evidence") or {}
            assessment = item.get("codex_assessment") or item.get("llm_assessment") or {}
            proposal = assessment.get("proposal") or {}
            human = item.get("human_review") or {}
            rows.append(
                {
                    "question_id": item.get("question_id", ""),
                    "article_id": article_id,
                    "original_question": comparison.get("original_question", ""),
                    "candidate_clarified_question": comparison.get(
                        "candidate_clarified_question", ""
                    ),
                    "final_clarified_question": comparison.get(
                        "final_clarified_question", ""
                    ),
                    "source_ground_truth": answer.get(
                        "source_expected_answer", answer.get("expected_answer", "")
                    ),
                    "ground_truth": answer.get("expected_answer", ""),
                    "accepted_answers": answer.get("accepted_answers", []),
                    "source_evidence_spans": answer.get("source_evidence_spans", []),
                    "evidence_spans": answer.get("evidence_spans", []),
                    "evidence_text": answer.get("evidence_text", ""),
                    "answer_modified": bool(answer.get("answer_modified", False)),
                    "answer_review_notes": answer.get("answer_review_notes", ""),
                    "llm_label": assessment.get("label", ""),
                    "assessment_rationale": assessment.get("rationale", ""),
                    "reason_codes": assessment.get(
                        "issue_codes", assessment.get("reason_codes", [])
                    ),
                    "validation_warnings": assessment.get("validation_warnings", []),
                    "proposal_status": proposal.get("status", "legacy"),
                    "proposer_tool": proposal.get("tool", ""),
                    "proposer_model": proposal.get("model", ""),
                    "proposal_batch_id": proposal.get("batch_id", ""),
                    "proposal_created_at": proposal.get("created_at", ""),
                    "proposed_supporting_quotes": assessment.get(
                        "proposed_supporting_quotes", []
                    ),
                    "review_decision": human.get("decision", "pending"),
                    "review_supporting_quotes": human.get("supporting_quotes", []),
                    "reviewer_id": human.get("reviewer_id", ""),
                    "review_notes": human.get("notes", ""),
                }
            )
    return rows


def review_status(review_path: str | Path) -> dict:
    counts: dict[str, int] = {decision: 0 for decision in REVIEW_DECISIONS}
    labels: dict[str, int] = {label: 0 for label in FULL_REVIEW_LABELS}
    proposal_counts = {"pending": 0, "proposed": 0, "legacy": 0}
    issues: dict[str, int] = {}
    corrected_answers = 0
    total = 0
    for row in _review_rows(review_path):
        decision = (row.get("review_decision") or "pending").strip()
        if decision not in REVIEW_DECISIONS:
            raise DatasetBuildError(f"Unknown review decision {decision!r}")
        counts[decision] += 1
        label = str(row.get("llm_label") or "pending").strip()
        if label not in labels:
            labels[label] = 0
        labels[label] += 1
        proposal_status = str(row.get("proposal_status") or "legacy").strip()
        if proposal_status not in proposal_counts:
            proposal_counts[proposal_status] = 0
        proposal_counts[proposal_status] += 1
        for issue in _review_list(row.get("reason_codes", []), "reason_codes", row.get("question_id", "")):
            issues[str(issue)] = issues.get(str(issue), 0) + 1
        corrected_answers += int(bool(row.get("answer_modified")))
        total += 1
    unresolved = counts["pending"] + counts["needs_adjudication"]
    proposals_complete = (
        proposal_counts.get("proposed", 0) + proposal_counts.get("legacy", 0)
    )
    return {
        "total": total,
        "decisions": counts,
        "labels": labels,
        "proposal_status": proposal_counts,
        "issue_codes": dict(sorted(issues.items())),
        "corrected_answers": corrected_answers,
        "excluded_questions": counts["exclude"],
        "ready": unresolved == 0 and proposals_complete == total,
    }


def _parse_json_list(value: str, field: str, question_id: str) -> list:
    if not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise DatasetBuildError(f"Invalid {field} JSON for {question_id}") from error
    if not isinstance(parsed, list):
        raise DatasetBuildError(f"{field} must be a JSON list for {question_id}")
    return parsed


def _review_list(value: object, field: str, question_id: str) -> list:
    if isinstance(value, list):
        return value
    return _parse_json_list(str(value or ""), field, question_id)


def _validated_review_spans(
    spans: object,
    article: dict,
    question_id: str,
) -> list[dict]:
    if not isinstance(spans, list) or not spans:
        raise DatasetBuildError(f"Missing reviewed evidence spans for {question_id}")
    validated: list[dict] = []
    context = article["context"]
    for span in spans:
        if not isinstance(span, dict):
            raise DatasetBuildError(f"Invalid reviewed evidence span for {question_id}")
        try:
            start = int(span["start"])
            end = int(span["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetBuildError(
                f"Invalid reviewed evidence offsets for {question_id}"
            ) from error
        if not 0 <= start < end <= len(context):
            raise DatasetBuildError(f"Reviewed evidence is out of range for {question_id}")
        text = context[start:end]
        supplied_text = str(span.get("text") or text)
        if supplied_text != text:
            raise DatasetBuildError(f"Reviewed evidence text mismatch for {question_id}")
        validated.append({"start": start, "end": end, "text": text})
    return validated


def _load_full_review_annotations(
    review_path: str | Path,
    evaluation_articles: Sequence[dict],
) -> dict[str, dict]:
    status = review_status(review_path)
    if not status["ready"]:
        raise DatasetBuildError(
            "Full human review is incomplete: "
            f"decisions={status['decisions']}, proposals={status['proposal_status']}"
        )
    article_by_id = {item["article_id"]: item for item in evaluation_articles}
    question_by_id = {
        question["question_id"]: question
        for article in evaluation_articles
        for question in article["questions"]
    }
    rows = _review_rows(review_path)
    if len(rows) != len(question_by_id):
        raise DatasetBuildError(
            f"Full review must contain {len(question_by_id)} questions, got {len(rows)}"
        )

    annotations: dict[str, dict] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "").strip()
        if question_id in annotations or question_id not in question_by_id:
            raise DatasetBuildError(f"Duplicate or unknown reviewed question {question_id}")
        question = question_by_id[question_id]
        article = article_by_id.get(question["article_id"])
        if article is None or row.get("article_id") != article["article_id"]:
            raise DatasetBuildError(f"Review article mismatch for {question_id}")
        if row.get("original_question") != question["question"]:
            raise DatasetBuildError(f"Original question changed for {question_id}")
        if row.get("source_ground_truth") != question["ground_truth"]:
            raise DatasetBuildError(f"Source answer changed for {question_id}")
        if row.get("source_evidence_spans") != question["evidence_spans"]:
            raise DatasetBuildError(f"Source evidence changed for {question_id}")

        proposal_status = str(row.get("proposal_status") or "").strip()
        label = str(row.get("llm_label") or "").strip()
        if proposal_status != "proposed" or label not in FULL_REVIEW_LABELS - {"pending"}:
            raise DatasetBuildError(
                f"Missing completed Codex proposal for {question_id}: "
                f"status={proposal_status!r}, label={label!r}"
            )
        if not all(
            str(row.get(field) or "").strip()
            for field in (
                "proposer_tool",
                "proposer_model",
                "proposal_batch_id",
                "proposal_created_at",
            )
        ):
            raise DatasetBuildError(f"Incomplete Codex proposal provenance for {question_id}")
        issues = _review_list(row.get("reason_codes", []), "reason_codes", question_id)
        unknown_issues = sorted(set(issues) - FULL_REVIEW_ISSUES)
        if unknown_issues:
            raise DatasetBuildError(
                f"Unknown issue codes for {question_id}: {unknown_issues}"
            )

        reviewed_ground_truth = str(row.get("ground_truth") or "").strip()
        if not reviewed_ground_truth:
            raise DatasetBuildError(f"Missing reviewed answer for {question_id}")
        reviewed_spans = _validated_review_spans(
            row.get("evidence_spans"), article, question_id
        )
        answer_modified = (
            reviewed_ground_truth != question["ground_truth"]
            or reviewed_spans != question["evidence_spans"]
        )
        if bool(row.get("answer_modified")) != answer_modified:
            raise DatasetBuildError(f"answer_modified is inconsistent for {question_id}")
        answer_notes = str(row.get("answer_review_notes") or "").strip()
        if answer_modified and not answer_notes:
            raise DatasetBuildError(
                f"Corrected answer requires answer_review_notes for {question_id}"
            )
        accepted_answers = row.get("accepted_answers") or [reviewed_ground_truth]
        if not isinstance(accepted_answers, list) or not all(
            isinstance(answer, str) and answer.strip() for answer in accepted_answers
        ):
            raise DatasetBuildError(f"Invalid accepted_answers for {question_id}")
        if reviewed_ground_truth not in accepted_answers:
            accepted_answers = [reviewed_ground_truth, *accepted_answers]

        reviewer = str(row.get("reviewer_id") or "").strip()
        decision = str(row.get("review_decision") or "pending").strip()
        review_notes = str(row.get("review_notes") or "").strip()
        if not reviewer:
            raise DatasetBuildError(f"Missing reviewer_id for {question_id}")
        answer_fields = {
            "source_ground_truth": question["ground_truth"],
            "source_evidence_spans": question["evidence_spans"],
            "ground_truth": reviewed_ground_truth,
            "accepted_answers": accepted_answers,
            "evidence_spans": reviewed_spans,
            "evidence": " | ".join(span["text"] for span in reviewed_spans),
            "answer_modified": answer_modified,
            "answer_review_notes": answer_notes,
        }
        provenance = {
            "reviewer_id": reviewer,
            "review_decision": decision,
            "review_notes": review_notes,
            "proposed_label": label,
            "proposal_rationale": str(row.get("assessment_rationale") or "").strip(),
            "proposal": {
                "tool": row.get("proposer_tool", ""),
                "model": row.get("proposer_model", ""),
                "batch_id": row.get("proposal_batch_id", ""),
                "created_at": row.get("proposal_created_at", ""),
            },
        }

        if decision == "exclude":
            exclusion_reasons = sorted(set(issues) & EXCLUSION_REASONS)
            if not exclusion_reasons or not review_notes:
                raise DatasetBuildError(
                    f"Excluded question {question_id} needs an exclusion reason and notes"
                )
            annotations[question_id] = {
                "final_label": "human_excluded",
                "reason_codes": issues,
                "final_clarified_question": None,
                "excluded": True,
                "exclusion_reasons": exclusion_reasons,
                **answer_fields,
                **provenance,
            }
            continue

        if decision == "mark_standalone":
            annotations[question_id] = {
                "final_label": "human_standalone",
                "reason_codes": issues,
                "final_clarified_question": None,
                "excluded": False,
                **answer_fields,
                **provenance,
            }
            continue
        if decision not in {"approve", "edit"}:
            raise DatasetBuildError(f"Unresolved review decision for {question_id}: {decision}")

        candidate = str(row.get("candidate_clarified_question") or "").strip()
        clarified = (
            candidate
            if decision == "approve"
            else str(row.get("final_clarified_question") or "").strip()
        )
        if not clarified:
            raise DatasetBuildError(f"Missing final clarification for {question_id}")
        if any(_answer_leaks(answer, clarified) for answer in accepted_answers):
            raise DatasetBuildError(f"Final clarification leaks answer for {question_id}")
        quotes = (
            _review_list(
                row.get("proposed_supporting_quotes", []),
                "proposed_supporting_quotes",
                question_id,
            )
            if decision == "approve"
            else _review_list(
                row.get("review_supporting_quotes", []),
                "review_supporting_quotes",
                question_id,
            )
        )
        sensitive_question = {
            **question,
            "evidence_spans": [*question["evidence_spans"], *reviewed_spans],
        }
        if not quotes or any(
            not _quote_is_supported(article, quote, sensitive_question)
            for quote in quotes
        ):
            raise DatasetBuildError(
                f"Unsupported final clarification context for {question_id}"
            )
        annotations[question_id] = {
            "final_label": "human_non_standalone",
            "reason_codes": issues,
            "final_clarified_question": clarified,
            "supporting_context_quotes": quotes,
            "excluded": False,
            **answer_fields,
            **provenance,
        }

    if set(annotations) != set(question_by_id):
        missing = sorted(set(question_by_id) - set(annotations))[:5]
        raise DatasetBuildError(f"Full review is missing questions: {missing}")
    return annotations


def load_review_annotations(
    review_path: str | Path,
    evaluation_articles: Sequence[dict],
    triage_records: Sequence[dict] | None = None,
) -> dict[str, dict]:
    """Load either the full Codex/human workflow or the legacy Gemini workflow."""

    path = Path(review_path)
    if path.suffix.lower() != ".csv":
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("review_mode") == "full_codex_human":
            return _load_full_review_annotations(path, evaluation_articles)
    if triage_records is None:
        raise DatasetBuildError("Legacy review finalization requires Gemini predictions")
    return load_approved_annotations(review_path, evaluation_articles, triage_records)


def load_approved_annotations(
    review_path: str | Path,
    evaluation_articles: Sequence[dict],
    triage_records: Sequence[dict],
) -> dict[str, dict]:
    status = review_status(review_path)
    if not status["ready"]:
        raise DatasetBuildError(f"Human review is incomplete: {status['decisions']}")
    article_by_id = {item["article_id"]: item for item in evaluation_articles}
    question_by_id = {
        item["question_id"]: item
        for article in evaluation_articles
        for item in article["questions"]
    }
    predictions = {
        item["question_id"]: item
        for record in triage_records
        for item in record["predictions"]
    }
    annotations: dict[str, dict] = {}
    for question_id, prediction in predictions.items():
        if prediction["standalone_label"] == "standalone":
            annotations[question_id] = {
                "final_label": "llm_standalone_unreviewed",
                "reason_codes": [],
                "final_clarified_question": None,
            }

    seen: set[str] = set()
    for row in _review_rows(review_path):
        question_id = row["question_id"].strip()
        if question_id in seen or question_id not in question_by_id:
            raise DatasetBuildError(f"Duplicate or unknown reviewed question {question_id}")
        seen.add(question_id)
        decision = row["review_decision"].strip()
        reviewer = row.get("reviewer_id", "").strip()
        if not reviewer:
            raise DatasetBuildError(f"Missing reviewer_id for {question_id}")
        prediction = predictions.get(question_id)
        if not prediction or prediction["standalone_label"] == "standalone":
            raise DatasetBuildError(f"Review row does not match triage queue: {question_id}")
        question = question_by_id[question_id]
        if row.get("original_question", "") != question["question"]:
            raise DatasetBuildError(
                f"Original question changed in review file for {question_id}"
            )
        candidate = str(row.get("candidate_clarified_question") or "")
        if candidate != str(prediction.get("candidate_clarified_question") or ""):
            raise DatasetBuildError(
                f"Gemini candidate changed in review file for {question_id}"
            )
        article = article_by_id[question["article_id"]]
        source_ground_truth = str(
            row.get("source_ground_truth") or question["ground_truth"]
        ).strip()
        if source_ground_truth != question["ground_truth"]:
            raise DatasetBuildError(
                f"Source answer changed in review file for {question_id}"
            )
        source_spans = row.get("source_evidence_spans") or question["evidence_spans"]
        if source_spans != question["evidence_spans"]:
            raise DatasetBuildError(
                f"Source evidence changed in review file for {question_id}"
            )
        reviewed_ground_truth = str(
            row.get("ground_truth") or question["ground_truth"]
        ).strip()
        reviewed_spans = (
            _validated_review_spans(row.get("evidence_spans"), article, question_id)
            if row.get("evidence_spans")
            else question["evidence_spans"]
        )
        answer_modified = (
            reviewed_ground_truth != question["ground_truth"]
            or reviewed_spans != question["evidence_spans"]
        )
        if "answer_modified" in row and bool(row["answer_modified"]) != answer_modified:
            raise DatasetBuildError(
                f"answer_modified flag is inconsistent for {question_id}"
            )
        if answer_modified and not str(row.get("answer_review_notes") or "").strip():
            raise DatasetBuildError(
                f"Corrected answer requires answer_review_notes for {question_id}"
            )
        accepted_answers = row.get("accepted_answers") or [reviewed_ground_truth]
        if not isinstance(accepted_answers, list) or not all(
            isinstance(answer, str) and answer.strip() for answer in accepted_answers
        ):
            raise DatasetBuildError(f"Invalid accepted_answers for {question_id}")
        if reviewed_ground_truth not in accepted_answers:
            accepted_answers = [reviewed_ground_truth, *accepted_answers]
        answer_fields = {
            "source_ground_truth": question["ground_truth"],
            "source_evidence_spans": question["evidence_spans"],
            "ground_truth": reviewed_ground_truth,
            "accepted_answers": accepted_answers,
            "evidence_spans": reviewed_spans,
            "evidence": " | ".join(span["text"] for span in reviewed_spans),
            "answer_modified": answer_modified,
            "answer_review_notes": str(row.get("answer_review_notes") or "").strip(),
        }
        if decision == "mark_standalone":
            annotations[question_id] = {
                "final_label": "human_standalone",
                "reason_codes": [],
                "final_clarified_question": None,
                "reviewer_id": reviewer,
                **answer_fields,
            }
            continue
        if decision not in {"approve", "edit"}:
            raise DatasetBuildError(f"Unresolved review decision for {question_id}: {decision}")
        if decision == "approve" and prediction.get("validation_warnings"):
            raise DatasetBuildError(
                f"Question {question_id} has LLM validation warnings and must be edited "
                "or marked standalone"
            )
        clarified = (
            prediction["candidate_clarified_question"]
            if decision == "approve"
            else row.get("final_clarified_question", "").strip()
        )
        if not clarified:
            raise DatasetBuildError(f"Missing final clarification for {question_id}")
        if _answer_leaks(reviewed_ground_truth, clarified):
            raise DatasetBuildError(f"Final clarification leaks answer for {question_id}")
        quotes = (
            [item["supporting_context_quote"] for item in prediction["added_context"]]
            if decision == "approve"
            else _review_list(
                row.get("review_supporting_quotes", ""),
                "review_supporting_quotes",
                question_id,
            )
        )
        sensitive_question = {
            **question,
            "evidence_spans": [*question["evidence_spans"], *reviewed_spans],
        }
        if not quotes or any(
            not _quote_is_supported(article, quote, sensitive_question)
            for quote in quotes
        ):
            raise DatasetBuildError(f"Unsupported final clarification context for {question_id}")
        annotations[question_id] = {
            "final_label": "human_non_standalone",
            "reason_codes": prediction["reason_codes"],
            "final_clarified_question": clarified,
            "supporting_context_quotes": quotes,
            "reviewer_id": reviewer,
            **answer_fields,
        }

    expected_review = {
        question_id
        for question_id, prediction in predictions.items()
        if prediction["standalone_label"] != "standalone"
    }
    if seen != expected_review:
        missing = sorted(expected_review - seen)[:5]
        raise DatasetBuildError(f"Review file is missing triage rows: {missing}")
    return annotations


def client_from_environment(model: str = "gemini-3.1-flash-lite") -> GeminiTriageClient:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise DatasetBuildError("GEMINI_API_KEY is required for Stage 1 LLM triage")
    return GeminiTriageClient(api_key=api_key, model=model)
