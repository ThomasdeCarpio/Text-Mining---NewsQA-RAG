"""Focused tests for benchmark score assembly."""

from newsqa_rag.evaluation.metrics import _gemini_judge_options
from scripts.score_benchmark_predictions import _merge_judge_scores


def test_merge_judge_scores_adds_only_successful_rows():
    rows = [{"question_id": "q1"}, {"question_id": "q2"}, {"question_id": "q3"}]
    records = {
        "q1": {"status": "success", "scores": {"faithfulness": 0.75}},
        "q2": {"status": "exhausted", "scores": {}},
    }

    assert _merge_judge_scores(rows, records) == 1
    assert rows[0]["ragas"] == {"faithfulness": 0.75}
    assert "ragas" not in rows[1]
    assert "ragas" not in rows[2]


def test_gemini_37_judge_omits_deprecated_sampling_controls():
    options = _gemini_judge_options("gemini-3.7-flash", "secret")

    assert options["model"] == "gemini-3.7-flash"
    assert "temperature" not in options


def test_older_gemini_judge_remains_deterministic():
    options = _gemini_judge_options("gemini-3.1-flash-lite", "secret")

    assert options["temperature"] == 0
