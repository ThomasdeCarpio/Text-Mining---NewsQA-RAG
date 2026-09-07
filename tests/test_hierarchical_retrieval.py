"""Regression tests for hierarchical child-to-parent context expansion."""

from newsqa_rag.retrieval.hierarchical import expand_ranked_children_to_parents


def test_parent_expansion_backfills_duplicate_children():
    children = [
        {"id": "c1", "score": 0.9},
        {"id": "c2", "score": 0.8},
        {"id": "c3", "score": 0.7},
        {"id": "c4", "score": 0.6},
    ]
    child_parent = {"c1": "p1", "c2": "p1", "c3": "p2", "c4": "p3"}
    parents = {
        "p1": {"id": "p1", "text": "Parent one", "metadata": {}},
        "p2": {"id": "p2", "text": "Parent two", "metadata": {}},
        "p3": {"id": "p3", "text": "Parent three", "metadata": {}},
    }

    expanded = expand_ranked_children_to_parents(
        children, child_parent, parents, limit=3
    )

    assert [row["id"] for row in expanded] == ["p1", "p2", "p3"]
    assert [row["metadata"]["source_child_id"] for row in expanded] == [
        "c1",
        "c3",
        "c4",
    ]
    assert expanded[0]["score"] == 0.9


def test_parent_expansion_reports_exhausted_unique_parent_pool():
    expanded = expand_ranked_children_to_parents(
        [{"id": "c1", "score": 0.9}, {"id": "c2", "score": 0.8}],
        {"c1": "p1", "c2": "p1"},
        {"p1": {"id": "p1", "text": "Parent", "metadata": {}}},
        limit=2,
    )

    assert [row["id"] for row in expanded] == ["p1"]
