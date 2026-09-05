import json
from pathlib import Path

from newsqa_rag.evaluation.abstention import (
    build_review_queue,
    validate_cases,
)
from scripts.score_abstention_predictions import _cluster_bootstrap, _metrics
from scripts.collect_abstention_predictions import _contexts_for, _parse_json


def _questions(count=12):
    return [
        {
            "question_id": f"q{i}",
            "source_question_id": f"q{i}",
            "article_key": f"article-{i}",
            "question": f"Who won event {i}?",
            "ground_truth": f"Person {i}",
            "accepted_answers": [f"Person {i}"],
            "relevant_chunk_ids": [f"chunk-{i}-0"],
        }
        for i in range(count)
    ]


def _chunks(count=12):
    rows = []
    for i in range(count):
        for index in range(2):
            rows.append(
                {
                    "id": f"chunk-{i}-{index}",
                    "text": f"Article {i}, section {index}",
                    "metadata": {
                        "canonical_article_id": f"article-{i}",
                        "article_id": f"physical-{i}",
                    },
                }
            )
    return rows


def _retrievals(count=12):
    return [
        {
            "question_id": f"q{i}",
            "status": "success",
            "trace": {
                "reranked_chunks": [
                    {"id": f"chunk-{i}-0"},
                    {"id": f"chunk-{i}-1"},
                ]
            },
        }
        for i in range(count)
    ]


def test_builder_uses_overlays_without_mutating_chunks(monkeypatch):
    questions = _questions(12)
    chunks = _chunks(12)
    authored = []
    for i in range(5):
        authored.extend(
            [
                {
                    "case_type": "counterfactual",
                    "base_question_id": f"q{i}",
                    "question": f"Who lost event {i} in a different country?",
                    "construction": {"changed_field": "event"},
                },
                {
                    "case_type": "external_unanswerable",
                    "base_question_id": f"external-{i}",
                    "question": f"What happened in absent article {i}?",
                    "partition": "development",
                    "construction": {"source": "withheld_newsqa"},
                },
            ]
        )
    # Keep this fixture small while exercising every construction branch.
    monkeypatch.setitem(
        __import__("newsqa_rag.evaluation.abstention", fromlist=["TARGETS"]).TARGETS,
        "fixture",
        {
            "answerable_control": 2,
            "natural_retrieval_miss": 0,
            "controlled_context_ablation": 2,
            "removed_article": 2,
            "external_unanswerable": 2,
            "counterfactual": 2,
            "partial_weak_evidence": 2,
        },
    )
    original_chunks = json.dumps(chunks, sort_keys=True)
    cases, manifest = build_review_queue(
        questions,
        chunks,
        mode="fixture",
        seed=42,
        development_articles=10,
        retrieval_records=_retrievals(12),
        authored_cases=authored,
    )
    assert not manifest["deficits"]
    assert json.dumps(chunks, sort_keys=True) == original_chunks
    assert validate_cases(cases, chunks)["status"] == "passed"
    for case in cases:
        if case["case_type"] in {"controlled_context_ablation", "partial_weak_evidence"}:
            assert set(case["source_gold_chunk_ids"]).isdisjoint(
                case["provided_context_chunk_ids"]
            )
        if case["case_type"] == "removed_article":
            assert len(case["excluded_chunk_ids"]) == 2


def test_validator_rejects_gold_leakage():
    case = {
        "case_id": "abs-one",
        "base_question_id": "q0",
        "case_type": "controlled_context_ablation",
        "partition": "development",
        "question": "Who won?",
        "answerability_label": "insufficient_evidence",
        "scope": "provided_context",
        "source_article_id": "article-0",
        "gold_relevant_chunk_ids": [],
        "source_gold_chunk_ids": ["chunk-0-0"],
        "provided_context_chunk_ids": ["chunk-0-0"],
        "excluded_chunk_ids": [],
        "excluded_article_ids": [],
        "human_review": {"decision": "pending"},
    }
    report = validate_cases([case], _chunks(1))
    assert report["status"] == "failed"
    assert any("gold_context_leakage" in error for error in report["errors"])


def test_validator_rejects_answer_text_in_adjacent_chunk():
    chunks = _chunks(1)
    chunks[1]["text"] = "An overlapping window still says Person 0 won."
    case = {
        "case_id": "abs-overlap",
        "base_question_id": "q0",
        "case_type": "partial_weak_evidence",
        "partition": "development",
        "question": "Who won?",
        "answerability_label": "insufficient_evidence",
        "scope": "provided_context",
        "source_article_id": "article-0",
        "accepted_answers": ["Person 0"],
        "gold_relevant_chunk_ids": [],
        "source_gold_chunk_ids": ["chunk-0-0"],
        "provided_context_chunk_ids": ["chunk-0-1"],
        "excluded_chunk_ids": ["chunk-0-0"],
        "excluded_article_ids": [],
        "human_review": {"decision": "pending"},
    }
    report = validate_cases([case], chunks)
    assert any("answer_text_leakage" in error for error in report["errors"])


def test_abstention_metrics_report_both_error_directions():
    result = _metrics(
        [
            ("insufficient_evidence", "insufficient_evidence"),
            ("insufficient_evidence", "answerable"),
            ("answerable", "insufficient_evidence"),
            ("answerable", "answerable"),
        ]
    )
    assert result["abstention_precision"] == 0.5
    assert result["abstention_recall"] == 0.5
    assert result["false_answer_rate"] == 0.5
    assert result["false_abstention_rate"] == 0.5


def test_structured_output_parser_enforces_abstention_contract():
    assert _parse_json(
        '```json\n{"answerability":"insufficient_evidence","answer":null,"citations":[]}\n```',
        0,
    )["answerability"] == "insufficient_evidence"
    try:
        _parse_json(
            '{"answerability":"insufficient_evidence","answer":"guess","citations":[]}',
            1,
        )
    except ValueError as error:
        assert "answer=null" in str(error)
    else:
        raise AssertionError("invalid abstention output was accepted")


def test_context_overlay_filters_removed_article():
    chunks = {row["id"]: row for row in _chunks(2)}
    case = {
        "case_id": "abs-remove",
        "base_question_id": "q0",
        "scope": "full_corpus",
        "provided_context_chunk_ids": [],
        "excluded_chunk_ids": ["chunk-0-0", "chunk-0-1"],
        "excluded_article_ids": ["physical-0"],
    }
    ids, contexts = _contexts_for(
        case,
        chunks,
        {"q0": ["chunk-0-0", "chunk-1-0", "chunk-0-1", "chunk-1-1"]},
        5,
    )
    assert ids == ["chunk-1-0", "chunk-1-1"]
    assert len(contexts) == 2


def test_confidence_interval_bootstraps_base_questions_not_variants():
    intervals = _cluster_bootstrap(
        [
            ("base-a", "answerable", "answerable"),
            ("base-a", "insufficient_evidence", "insufficient_evidence"),
            ("base-b", "insufficient_evidence", "answerable"),
        ],
        repetitions=50,
        seed=42,
    )
    assert set(intervals["abstention_recall"]) == {"lower", "upper"}
