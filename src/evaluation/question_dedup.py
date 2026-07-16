"""Validated semantic-question deduplication for reviewed NewsQA testsets."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from typing import Iterable, Sequence

from src.evaluation.testset import DatasetBuildError, canonical_json


DEDUP_SCHEMA_VERSION = "1.0"
DEDUP_APPROVAL_SCHEMA_VERSION = "1.0"


def normalize_question(value: str) -> str:
    """Normalize superficial wording differences without changing semantics."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def stable_cluster_id(article_id: str, member_ids: Sequence[str]) -> str:
    payload = {"article_id": article_id, "member_question_ids": sorted(member_ids)}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"qcluster_{digest[:16]}"


def _rows_by_id(rows: Sequence[dict], name: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        question_id = str(row.get("source_question_id") or row.get("question_id") or "")
        if not question_id or question_id in indexed:
            raise DatasetBuildError(f"{name} has a missing or duplicate question ID")
        indexed[question_id] = row
    return indexed


def validate_cluster_decisions(
    decisions: dict,
    resolved_rows: Sequence[dict],
) -> list[dict]:
    """Validate reviewed multi-question clusters and add singleton clusters."""

    if decisions.get("schema_version") != DEDUP_SCHEMA_VERSION:
        raise DatasetBuildError("Unsupported semantic dedup schema version")
    resolved_by_id = _rows_by_id(resolved_rows, "resolved testset")
    claimed: set[str] = set()
    clusters: list[dict] = []

    for raw_cluster in decisions.get("clusters") or []:
        members = sorted(set(raw_cluster.get("member_question_ids") or []))
        representative = str(raw_cluster.get("representative_question_id") or "")
        if len(members) < 2:
            raise DatasetBuildError("Semantic clusters must contain at least two questions")
        if representative not in members:
            raise DatasetBuildError("Cluster representative must be a cluster member")
        unknown = sorted(set(members) - set(resolved_by_id))
        if unknown:
            raise DatasetBuildError(f"Semantic cluster contains unknown questions: {unknown[:5]}")
        repeated = sorted(set(members) & claimed)
        if repeated:
            raise DatasetBuildError(f"Questions occur in multiple clusters: {repeated[:5]}")
        article_ids = {resolved_by_id[item]["article_key"] for item in members}
        article_id = str(raw_cluster.get("article_id") or "")
        if len(article_ids) != 1 or article_id not in article_ids:
            raise DatasetBuildError("Semantic clusters cannot cross article boundaries")
        if not str(raw_cluster.get("semantic_target") or "").strip():
            raise DatasetBuildError("Semantic cluster is missing its target description")
        if not str(raw_cluster.get("rationale") or "").strip():
            raise DatasetBuildError("Semantic cluster is missing its rationale")

        cluster = {
            **raw_cluster,
            "cluster_id": stable_cluster_id(article_id, members),
            "member_question_ids": members,
            "representative_question_id": representative,
        }
        clusters.append(cluster)
        claimed.update(members)

    for question_id in sorted(set(resolved_by_id) - claimed):
        article_id = resolved_by_id[question_id]["article_key"]
        clusters.append(
            {
                "cluster_id": stable_cluster_id(article_id, [question_id]),
                "article_id": article_id,
                "member_question_ids": [question_id],
                "representative_question_id": question_id,
                "semantic_target": resolved_by_id[question_id]["question"],
                "rationale": "No equivalent question was identified in the same article.",
            }
        )

    clusters.sort(key=lambda item: item["cluster_id"])
    covered = [item for cluster in clusters for item in cluster["member_question_ids"]]
    if len(covered) != len(resolved_rows) or set(covered) != set(resolved_by_id):
        raise DatasetBuildError("Semantic clusters do not partition the resolved testset")
    return clusters


def validate_human_approval(
    approval: dict,
    *,
    proposal_sha256: str,
    base_testset_sha256: str,
    clusters: Sequence[dict],
) -> None:
    """Require explicit, exact-coverage human approval for every proposed cluster."""

    if approval.get("schema_version") != DEDUP_APPROVAL_SCHEMA_VERSION:
        raise DatasetBuildError("Unsupported semantic dedup approval schema version")
    if approval.get("proposal_sha256") != proposal_sha256:
        raise DatasetBuildError("Human approval targets a different cluster proposal")
    if approval.get("base_testset_sha256") != base_testset_sha256:
        raise DatasetBuildError("Human approval targets a different resolved testset")

    review = approval.get("human_review") or {}
    if review.get("status") != "approved":
        raise DatasetBuildError("Semantic deduplication has not received human approval")
    if not str(review.get("reviewer_id") or "").strip():
        raise DatasetBuildError("Semantic deduplication approval has no reviewer ID")
    if not str(review.get("reviewed_at") or "").strip():
        raise DatasetBuildError("Semantic deduplication approval has no timestamp")

    expected_ids = {
        cluster["cluster_id"]
        for cluster in clusters
        if len(cluster["member_question_ids"]) > 1
    }
    reviews = approval.get("cluster_reviews") or []
    reviewed: dict[str, str] = {}
    for item in reviews:
        cluster_id = str(item.get("cluster_id") or "")
        decision = str(item.get("decision") or "")
        if not cluster_id or cluster_id in reviewed:
            raise DatasetBuildError("Human approval has a missing or duplicate cluster ID")
        if decision not in {"approve", "reject"}:
            raise DatasetBuildError(f"Unsupported human cluster decision: {decision!r}")
        reviewed[cluster_id] = decision

    if set(reviewed) != expected_ids:
        missing = sorted(expected_ids - set(reviewed))
        unexpected = sorted(set(reviewed) - expected_ids)
        raise DatasetBuildError(
            "Human approval does not exactly cover proposed clusters: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    rejected = sorted(
        cluster_id for cluster_id, decision in reviewed.items() if decision != "approve"
    )
    if rejected:
        raise DatasetBuildError(
            "Rejected semantic clusters must be removed or split before finalization: "
            f"{rejected[:5]}"
        )


def _unique(values: Iterable[object]) -> list:
    output = []
    seen: set[str] = set()
    for value in values:
        key = canonical_json(value)
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _merge_reviewed_fields(representative: dict, members: Sequence[dict]) -> dict:
    answers = _unique(
        answer
        for row in members
        for answer in (row.get("accepted_answers") or [row["ground_truth"]])
    )
    evidence_spans = _unique(
        span for row in members for span in (row.get("evidence_spans") or [])
    )
    relevant_chunk_ids = sorted(
        {chunk_id for row in members for chunk_id in row.get("relevant_chunk_ids", [])}
    )
    return {
        **representative,
        "accepted_answers": answers,
        "evidence_spans": evidence_spans,
        "evidence": " | ".join(str(span.get("text") or "") for span in evidence_spans),
        "relevant_chunk_ids": relevant_chunk_ids,
    }


def derive_deduplicated_artifacts(
    reviewed_original_rows: Sequence[dict],
    resolved_rows: Sequence[dict],
    clusters: Sequence[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """Apply one semantic partition to paired reviewed-original/resolved rows."""

    original_by_id = _rows_by_id(reviewed_original_rows, "reviewed-original testset")
    resolved_by_id = _rows_by_id(resolved_rows, "resolved testset")
    if set(original_by_id) != set(resolved_by_id):
        raise DatasetBuildError("Reviewed-original and resolved source IDs differ")

    dedup_original: list[dict] = []
    dedup_resolved: list[dict] = []
    dedup_clarified: list[dict] = []
    removed: list[dict] = []
    cluster_records: list[dict] = []

    for cluster in clusters:
        member_ids = cluster["member_question_ids"]
        representative_id = cluster["representative_question_id"]
        original_members = [original_by_id[item] for item in member_ids]
        resolved_members = [resolved_by_id[item] for item in member_ids]
        original = _merge_reviewed_fields(
            original_by_id[representative_id], original_members
        )
        resolved = _merge_reviewed_fields(resolved_by_id[representative_id], resolved_members)
        shared = {
            "dedup_cluster_id": cluster["cluster_id"],
            "dedup_member_count": len(member_ids),
            "merged_source_question_ids": member_ids,
            "dedup_aggregated": len(member_ids) > 1,
        }
        original.update(shared)
        resolved.update(shared)
        dedup_original.append(original)
        dedup_resolved.append(resolved)

        if resolved.get("question_variant") == "clarified":
            clarified = {
                **resolved,
                "question_id": f"{representative_id}::clarified",
                "source_question_id": representative_id,
            }
            dedup_clarified.append(clarified)

        for question_id in member_ids:
            if question_id == representative_id:
                continue
            removed.append(
                {
                    "question_id": question_id,
                    "article_key": cluster["article_id"],
                    "original_question": original_by_id[question_id]["question"],
                    "resolved_question": resolved_by_id[question_id]["question"],
                    "representative_question_id": representative_id,
                    "dedup_cluster_id": cluster["cluster_id"],
                    "reason": "semantic_duplicate",
                }
            )

        cluster_records.append(
            {
                **cluster,
                "member_count": len(member_ids),
                "member_questions": [
                    {
                        "question_id": item,
                        "original_question": original_by_id[item]["question"],
                        "resolved_question": resolved_by_id[item]["question"],
                        "ground_truth": resolved_by_id[item]["ground_truth"],
                    }
                    for item in member_ids
                ],
            }
        )

    for rows in (dedup_original, dedup_resolved, dedup_clarified, removed):
        rows.sort(key=lambda item: item["question_id"])
    cluster_records.sort(key=lambda item: item["cluster_id"])

    normalized_keys = [
        (row["article_key"], normalize_question(row["question"]))
        for row in dedup_resolved
    ]
    if len(normalized_keys) != len(set(normalized_keys)):
        raise DatasetBuildError("Deduplicated resolved testset still has normalized duplicates")
    if [row["source_question_id"] for row in dedup_original] != [
        row["source_question_id"] for row in dedup_resolved
    ]:
        raise DatasetBuildError("Deduplicated paired variants have different representatives")
    return dedup_original, dedup_resolved, dedup_clarified, removed, cluster_records
