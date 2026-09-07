"""Utilities for converting ranked hierarchical children into parent contexts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def expand_ranked_children_to_parents(
    ranked_children: Sequence[dict],
    child_parent: Mapping[str, str],
    parent_by_id: Mapping[str, dict],
    limit: int,
) -> list[dict]:
    """Return the first ``limit`` unique parents in child-reranker order."""

    if limit < 1:
        raise ValueError("limit must be at least 1")

    expanded: list[dict] = []
    seen: set[str] = set()
    for child in ranked_children:
        child_id = child["id"]
        parent_id = child_parent[child_id]
        if parent_id in seen:
            continue

        seen.add(parent_id)
        parent = dict(parent_by_id[parent_id])
        parent["metadata"] = dict(parent.get("metadata") or {})
        parent["metadata"].update(
            {
                "source_child_id": child_id,
                "source_child_score": child.get("score"),
            }
        )
        parent["score"] = child.get("score", 0.0)
        expanded.append(parent)
        if len(expanded) == limit:
            break

    return expanded
