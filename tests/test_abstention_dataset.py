import json
from pathlib import Path

from newsqa_rag.evaluation.abstention import (
    PARTITION_TARGETS,
    TARGETS,
    build_review_queue,
    validate_cases,
)
from scripts.calibrate_abstention_threshold import _apply_threshold
from scripts.prepare_abstention_dataset import _load_review_cases, _readable_review_document
from scripts.score_abstention_predictions import (
    _cluster_bootstrap,
    _conservative_label,
    _metrics,
)
from scripts.collect_abstention_predictions import (
    _contexts_for,
    _parse_baseline,
    _parse_json,
    _retrieval_features,
)


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
                    "source_article_id": f"withheld-article-{i}",
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
    assert result["selective_risk"] == 0.5


def test_exhausted_prediction_is_counted_as_worst_case_error():
    assert _conservative_label("answerable", {"status": "exhausted"}) == (
        "insufficient_evidence",
        False,
    )
    assert _conservative_label(
        "insufficient_evidence", {"status": "success", "answerability": None}
    ) == ("answerable", False)


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


def test_compact_profile_has_exact_preregistered_totals():
    assert sum(TARGETS["compact_200"].values()) == 200
    assert sum(PARTITION_TARGETS["compact_200"]["development"].values()) == 140
    assert sum(PARTITION_TARGETS["compact_200"]["final_test"].values()) == 60
    for case_type, total in TARGETS["compact_200"].items():
        assert sum(
            PARTITION_TARGETS["compact_200"][partition][case_type]
            for partition in ("development", "final_test")
        ) == total


def test_baseline_parser_only_accepts_canonical_abstention():
    abstained = _parse_baseline(
        "I cannot find this information in the provided context.", 2
    )
    assert abstained == {
        "answerability": "insufficient_evidence",
        "answer": None,
        "citations": [],
    }
    answered = _parse_baseline("Natalie Cole [2]", 2)
    assert answered["answerability"] == "answerable"
    assert answered["citations"] == [2]


def test_retrieval_features_are_only_applicable_to_full_corpus_cases():
    trace = {
        "reranked_chunks": [
            {"id": "one", "reranker_score": 0.8},
            {"id": "two", "reranker_score": 0.3},
        ]
    }
    full = {"case_id": "c1", "base_question_id": "q1", "scope": "full_corpus"}
    controlled = {"case_id": "c2", "base_question_id": "q2", "scope": "provided_context"}
    assert _retrieval_features(full, {"c1": trace}) == {
        "applicable": True,
        "top1_reranker_score": 0.8,
        "top1_top2_margin": 0.5,
    }
    assert not _retrieval_features(controlled, {"c2": trace})["applicable"]


def test_score_gate_preserves_inapplicable_cases_and_rejects_low_scores():
    cases = [
        {"case_id": "low"},
        {"case_id": "high"},
        {"case_id": "controlled"},
    ]
    predictions = [
        {
            "case_id": "low",
            "answerability": "answerable",
            "answer": "guess",
            "citations": [1],
            "retrieval_features": {"applicable": True, "top1_reranker_score": 0.2},
        },
        {
            "case_id": "high",
            "answerability": "answerable",
            "answer": "supported",
            "citations": [1],
            "retrieval_features": {"applicable": True, "top1_reranker_score": 0.8},
        },
        {
            "case_id": "controlled",
            "answerability": "answerable",
            "answer": "unchanged",
            "citations": [1],
            "retrieval_features": {"applicable": False, "top1_reranker_score": None},
        },
    ]
    by_id = {row["case_id"]: row for row in _apply_threshold(cases, predictions, 0.5)}
    assert by_id["low"]["answerability"] == "insufficient_evidence"
    assert by_id["low"]["gate"]["rejected"] is True
    assert by_id["high"]["answerability"] == "answerable"
    assert by_id["controlled"]["answer"] == "unchanged"


def test_readable_review_round_trip(tmp_path):
    cases = [
        {
            "case_id": "one",
            "partition": "development",
            "case_type": "answerable_control",
            "source_gold_chunk_ids": ["chunk-0-0"],
            "provided_context_chunk_ids": ["chunk-0-1"],
            "excluded_chunk_ids": [],
            "human_review": {"decision": "pending"},
        }
    ]
    manifest = {
        "mode": "compact_200",
        "seed": 42,
        "targets": TARGETS["compact_200"],
        "deficits": {},
    }
    document = _readable_review_document(cases, _chunks(1), manifest)
    path = tmp_path / "review_queue_readable.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = _load_review_cases(path)
    assert loaded[0]["case_id"] == "one"
    assert loaded[0]["review_material"]["source_gold_chunks"][0]["chunk_id"] == "chunk-0-0"


def test_compact_builder_fulfils_exact_type_and_partition_quotas():
    questions = _questions(100)
    chunks = _chunks(100)
    retrievals = []
    for index in range(100):
        ranked_id = f"chunk-{(index + 20) % 100}-1" if index < 10 else f"chunk-{index}-0"
        fallback_id = f"chunk-{(index + 30) % 100}-1"
        retrievals.append(
            {
                "question_id": f"q{index}",
                "status": "success",
                "trace": {
                    "reranked_chunks": [{"id": ranked_id}, {"id": fallback_id}]
                },
            }
        )
    authored = [
        {
            "case_type": "counterfactual",
            "base_question_id": f"q{index}",
            "question": f"Did Person {index} lose event {index}?",
            "construction": {"changed_field": "event"},
        }
        for index in range(100)
    ]
    authored.extend(
        {
            "case_type": "external_unanswerable",
            "base_question_id": f"external-{index}",
            "source_article_id": f"withheld-{index}",
            "question": f"What happened in withheld report {index}?",
            "partition": "development" if index < 15 else "final_test",
            "construction": {"source": "withheld_newsqa"},
        }
        for index in range(22)
    )
    cases, manifest = build_review_queue(
        questions,
        chunks,
        mode="compact_200",
        seed=42,
        development_articles=70,
        retrieval_records=retrievals,
        authored_cases=authored,
    )
    assert len(cases) == 200
    assert not manifest["deficits"]
    assert manifest["observed"] == TARGETS["compact_200"]
    assert manifest["observed_by_partition"] == PARTITION_TARGETS["compact_200"]
    assert validate_cases(cases, chunks)["status"] == "passed"
