"""Private cloud packaging and verification for the NewsQA evaluation dataset."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from newsqa_rag.evaluation.question_dedup import (
    validate_cluster_decisions,
    validate_human_approval,
)
from newsqa_rag.evaluation.question_review import review_status
from newsqa_rag.evaluation.testset import (
    DatasetBuildError,
    canonical_json,
    load_testset,
    sha256_file,
    sha256_text,
)


CLOUD_SCHEMA_VERSION = "1.0"
CANONICAL_ARTIFACTS = {
    "evaluation_articles": "corpus/evaluation_articles.jsonl",
    "distractor_articles": "corpus/distractor_articles.jsonl",
    "original_questions": "questions/original_questions.jsonl",
    "review_queue": "review/review_queue_readable.json",
    "review_manifest": "review/manifest.json",
    "review_schema": "review/schema.json",
    "dedup_decisions": "dedup/semantic_clusters.json",
    "dedup_approval": "dedup/human_approval.json",
    "selection_manifest": "metadata/selection_manifest.json",
}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _record_count(path: Path) -> int | None:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    return None


def _artifact_record(path: Path, relative_path: str) -> dict:
    record = {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    count = _record_count(path)
    if count is not None:
        record["records"] = count
    return record


def _source_paths(project_root: Path, source_root: Path) -> dict[str, Path]:
    return {
        "evaluation_articles": source_root / "staging/corpus/evaluation_articles.jsonl",
        "distractor_articles": source_root / "staging/corpus/distractor_articles.jsonl",
        "original_questions": source_root / "staging/questions/original_questions.jsonl",
        "review_queue": source_root / "staging/review/review_queue_readable.json",
        "review_manifest": source_root / "staging/review/manifest.json",
        "review_schema": source_root / "staging/review/audits/schema.json",
        "dedup_decisions": project_root
        / "evaluation/question_dedup/newsqa_200_11064.semantic_clusters.json",
        "dedup_approval": project_root
        / "evaluation/question_dedup/newsqa_200_11064.human_approval.json",
        "selection_manifest": project_root
        / "evaluation/manifests/newsqa_200_11064.selection.json",
        "resolved_testset": source_root / "final/testset_resolved.jsonl",
    }


def _validate_evidence(articles: Iterable[dict]) -> tuple[set[str], set[str]]:
    article_ids: set[str] = set()
    question_ids: set[str] = set()
    for article in articles:
        article_id = str(article.get("article_id") or "")
        if not article_id or article_id in article_ids:
            raise DatasetBuildError("Corpus has a missing or duplicate article ID")
        article_ids.add(article_id)
        context = str(article.get("context") or "")
        for question in article.get("questions") or []:
            question_id = str(question.get("question_id") or "")
            if not question_id or question_id in question_ids:
                raise DatasetBuildError("Evaluation corpus has a missing or duplicate question ID")
            question_ids.add(question_id)
            for span in question.get("evidence_spans") or []:
                start, end = int(span["start"]), int(span["end"])
                if not 0 <= start < end <= len(context):
                    raise DatasetBuildError(f"Invalid evidence span for {question_id}")
                if context[start:end] != span.get("text"):
                    raise DatasetBuildError(f"Evidence text mismatch for {question_id}")
    return article_ids, question_ids


def validate_source_dataset(project_root: Path, source_root: Path) -> dict:
    """Validate the reviewed source payload before it can be published."""

    paths = _source_paths(project_root, source_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise DatasetBuildError(f"Missing canonical source artifacts: {missing}")

    evaluation_articles = load_testset(paths["evaluation_articles"])
    distractor_articles = load_testset(paths["distractor_articles"])
    original_questions = load_testset(paths["original_questions"])
    evaluation_ids, nested_question_ids = _validate_evidence(evaluation_articles)
    distractor_ids, distractor_question_ids = _validate_evidence(distractor_articles)
    if distractor_question_ids:
        raise DatasetBuildError("Distractor articles must not contain evaluation questions")
    overlap = evaluation_ids & distractor_ids
    if overlap:
        raise DatasetBuildError(f"Evaluation and distractor corpora overlap: {sorted(overlap)[:5]}")

    flat_question_ids = {str(row.get("question_id") or "") for row in original_questions}
    if "" in flat_question_ids or len(flat_question_ids) != len(original_questions):
        raise DatasetBuildError("Original questions have missing or duplicate IDs")
    if flat_question_ids != nested_question_ids:
        raise DatasetBuildError("Article questions and flat original questions do not match")

    status = review_status(paths["review_queue"])
    if not status.get("ready"):
        raise DatasetBuildError("Human question review is not complete")

    resolved = load_testset(paths["resolved_testset"])
    decisions = json.loads(paths["dedup_decisions"].read_text(encoding="utf-8"))
    approval = json.loads(paths["dedup_approval"].read_text(encoding="utf-8"))
    resolved_sha = sha256_file(paths["resolved_testset"])
    if decisions.get("base_testset_sha256") != resolved_sha:
        raise DatasetBuildError("Dedup decisions target a different resolved testset")
    clusters = validate_cluster_decisions(decisions, resolved)
    validate_human_approval(
        approval,
        proposal_sha256=sha256_file(paths["dedup_decisions"]),
        base_testset_sha256=resolved_sha,
        clusters=clusters,
    )
    return {
        "evaluation_articles": len(evaluation_articles),
        "distractor_articles": len(distractor_articles),
        "questions": len(original_questions),
        "resolved_questions": len(resolved),
        "dedup_multi_question_clusters": sum(
            len(cluster["member_question_ids"]) > 1 for cluster in clusters
        ),
        "review": status,
    }


def _dataset_card(version: str, statistics: dict) -> str:
    return f"""---
pretty_name: Private NewsQA RAG Evaluation Dataset
language:
- en
task_categories:
- question-answering
---

# Private NewsQA RAG Evaluation Dataset

Private, human-reviewed evaluation source data for the NewsQA RAG project.
This repository is not a prebuilt retrieval index. Chunk, BM25, Chroma, and
ground-truth chunk mappings must be rebuilt from the pinned release.

- Version: `{version}`
- Evaluation articles: {statistics['evaluation_articles']}
- Distractor articles: {statistics['distractor_articles']}
- Source questions: {statistics['questions']}

Redistribution rights for the upstream NewsQA-derived text must be verified
before changing this repository from private to public.
"""


def export_canonical_bundle(
    *,
    project_root: str | Path,
    source_root: str | Path,
    output_dir: str | Path,
    version: str,
    overwrite: bool = False,
) -> dict:
    """Export a chunk-independent, checksum-locked upload bundle."""

    project = Path(project_root).resolve()
    source = Path(source_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        if not overwrite:
            raise DatasetBuildError(f"Export directory already exists: {output}")
        shutil.rmtree(output)

    statistics = validate_source_dataset(project, source)
    source_paths = _source_paths(project, source)
    output.mkdir(parents=True)
    artifacts: dict[str, dict] = {}
    for key, relative_path in CANONICAL_ARTIFACTS.items():
        destination = output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_paths[key], destination)
        artifacts[key] = _artifact_record(destination, relative_path)

    selection = json.loads(source_paths["selection_manifest"].read_text(encoding="utf-8"))
    manifest = {
        "schema_version": CLOUD_SCHEMA_VERSION,
        "dataset_version": version,
        "private_distribution_only": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "git_commit": _git_commit(project),
            "python": platform.python_version(),
            "tool": "scripts/publish_evaluation_dataset.py",
        },
        "source": {
            "dataset": selection.get("dataset", {}),
            "seed": selection.get("sampling", {}).get("seed", 42),
        },
        "statistics": statistics,
        "review_provenance": {
            "dedup_proposal_sha256": sha256_file(source_paths["dedup_decisions"]),
            "dedup_base_testset_sha256": json.loads(
                source_paths["dedup_decisions"].read_text(encoding="utf-8")
            ).get("base_testset_sha256"),
        },
        "artifacts": artifacts,
    }
    manifest["dataset_sha256"] = sha256_text(canonical_json(artifacts))
    _write_json(output / "cloud_manifest.json", manifest)
    (output / "README.md").write_text(
        _dataset_card(version, statistics), encoding="utf-8"
    )
    return manifest


def verify_canonical_bundle(bundle_dir: str | Path) -> dict:
    """Verify every canonical artifact in a downloaded release."""

    bundle = Path(bundle_dir).resolve()
    manifest_path = bundle / "cloud_manifest.json"
    if not manifest_path.exists():
        raise DatasetBuildError("Downloaded dataset is missing cloud_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CLOUD_SCHEMA_VERSION:
        raise DatasetBuildError("Unsupported cloud dataset schema version")
    if not manifest.get("private_distribution_only"):
        raise DatasetBuildError("Dataset manifest does not enforce private distribution")
    artifacts = manifest.get("artifacts") or {}
    if set(artifacts) != set(CANONICAL_ARTIFACTS):
        raise DatasetBuildError("Cloud manifest has missing or unexpected canonical artifacts")
    for key, record in artifacts.items():
        expected_path = CANONICAL_ARTIFACTS[key]
        if record.get("path") != expected_path:
            raise DatasetBuildError(f"Unexpected path for artifact {key}")
        path = bundle / expected_path
        if not path.is_file():
            raise DatasetBuildError(f"Downloaded artifact is missing: {expected_path}")
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise DatasetBuildError(f"Downloaded artifact failed integrity check: {expected_path}")
        if "records" in record and _record_count(path) != record["records"]:
            raise DatasetBuildError(f"Record count mismatch: {expected_path}")
    if sha256_text(canonical_json(artifacts)) != manifest.get("dataset_sha256"):
        raise DatasetBuildError("Dataset-level checksum does not match the artifact manifest")
    return manifest


def materialize_canonical_source(
    bundle_dir: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Copy verified cloud artifacts into the local evaluation staging layout."""

    bundle = Path(bundle_dir).resolve()
    output = Path(output_root).resolve()
    manifest = verify_canonical_bundle(bundle)
    marker = output / "manifests/source.json"
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            if marker.exists():
                current = json.loads(marker.read_text(encoding="utf-8"))
                if current.get("dataset_sha256") == manifest.get("dataset_sha256"):
                    return local_materialized_paths(output)
            raise DatasetBuildError(f"Output directory already exists: {output}")
        shutil.rmtree(output)

    paths = local_materialized_paths(output)
    mapping = {
        "evaluation_articles": paths["evaluation_articles"],
        "distractor_articles": paths["distractor_articles"],
        "original_questions": paths["original_questions"],
        "review_queue": paths["review_queue"],
        "review_manifest": paths["review_manifest"],
        "review_schema": paths["review_schema"],
        "dedup_decisions": paths["dedup_decisions"],
        "dedup_approval": paths["dedup_approval"],
        "selection_manifest": paths["selection_manifest"],
    }
    for key, destination in mapping.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / CANONICAL_ARTIFACTS[key], destination)
    source_manifest = {
        **manifest,
        "materialized_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(marker, source_manifest)
    return paths


def local_materialized_paths(output_root: str | Path) -> dict[str, Path]:
    output = Path(output_root).resolve()
    return {
        "evaluation_articles": output / "staging/corpus/evaluation_articles.jsonl",
        "distractor_articles": output / "staging/corpus/distractor_articles.jsonl",
        "original_questions": output / "staging/questions/original_questions.jsonl",
        "review_queue": output / "staging/review/review_queue_readable.json",
        "review_manifest": output / "staging/review/manifest.json",
        "review_schema": output / "staging/review/audits/schema.json",
        "dedup_decisions": output / "staging/dedup/semantic_clusters.json",
        "dedup_approval": output / "staging/dedup/human_approval.json",
        "selection_manifest": output / "manifests/selection.json",
        "source_manifest": output / "manifests/source.json",
        "variant_manifest": output / "manifests/variant.json",
        "dedup_variant_manifest": output / "manifests/deduplicated.variant.json",
    }


def rebind_dedup_approval(
    *,
    decisions_path: str | Path,
    approval_path: str | Path,
    resolved_testset: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Bind unchanged reviewed semantic decisions to newly mapped chunk IDs."""

    decisions_source = Path(decisions_path).resolve()
    approval_source = Path(approval_path).resolve()
    resolved_path = Path(resolved_testset).resolve()
    decisions = json.loads(decisions_source.read_text(encoding="utf-8"))
    approval = json.loads(approval_source.read_text(encoding="utf-8"))
    original_proposal_sha = sha256_file(decisions_source)
    original_base_sha = decisions.get("base_testset_sha256")
    if approval.get("proposal_sha256") != original_proposal_sha:
        raise DatasetBuildError("Downloaded approval targets a different dedup proposal")
    if approval.get("base_testset_sha256") != original_base_sha:
        raise DatasetBuildError("Downloaded dedup proposal and approval target different testsets")

    resolved = load_testset(resolved_path)
    clusters = validate_cluster_decisions(decisions, resolved)
    expected_cluster_ids = {
        cluster["cluster_id"]
        for cluster in clusters
        if len(cluster["member_question_ids"]) > 1
    }
    reviewed_cluster_ids = {
        str(item.get("cluster_id") or "") for item in approval.get("cluster_reviews") or []
    }
    if reviewed_cluster_ids != expected_cluster_ids:
        raise DatasetBuildError("Human dedup approval does not cover the rebuilt testset")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    new_base_sha = sha256_file(resolved_path)
    rebound_decisions = {
        **decisions,
        "base_testset_sha256": new_base_sha,
        "chunk_remap": {
            "method": "mechanical_rebind_after_chunk_mapping",
            "original_base_testset_sha256": original_base_sha,
            "original_proposal_sha256": original_proposal_sha,
        },
    }
    rebound_decisions_path = output / "semantic_clusters.rebound.json"
    _write_json(rebound_decisions_path, rebound_decisions)
    rebound_approval = {
        **approval,
        "base_testset_sha256": new_base_sha,
        "proposal_sha256": sha256_file(rebound_decisions_path),
        "chunk_remap": rebound_decisions["chunk_remap"],
    }
    rebound_approval_path = output / "human_approval.rebound.json"
    _write_json(rebound_approval_path, rebound_approval)
    validate_human_approval(
        rebound_approval,
        proposal_sha256=sha256_file(rebound_decisions_path),
        base_testset_sha256=new_base_sha,
        clusters=clusters,
    )
    return rebound_decisions_path, rebound_approval_path


def index_fingerprint(
    *, dataset_sha256: str, dataset_commit: str, config: dict
) -> str:
    """Fingerprint every input that can change chunks or retrieval indexes."""

    payload = {
        "dataset_sha256": dataset_sha256,
        "dataset_commit": dataset_commit,
        "cleaning": config.get("cleaning", {}),
        "chunking": config.get("chunking", {}),
        "embedding": config.get("embedding", {}),
        "database": config.get("database", {}),
    }
    return sha256_text(canonical_json(payload))
