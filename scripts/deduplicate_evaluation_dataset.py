#!/usr/bin/env python3
"""Build a reviewed semantic-deduplicated variant of the final NewsQA set."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.question_dedup import (
    DEDUP_APPROVAL_SCHEMA_VERSION,
    DEDUP_SCHEMA_VERSION,
    derive_deduplicated_artifacts,
    validate_cluster_decisions,
    validate_human_approval,
)
from src.evaluation.testset import (
    DatasetBuildError,
    artifact_record,
    load_testset,
    save_jsonl,
    sha256_file,
)


DEFAULT_BASE_ROOT = PROJECT_ROOT / "data/evaluation/newsqa_200_11064/final"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/evaluation/newsqa_200_11064/final_deduplicated"
DEFAULT_DECISIONS = PROJECT_ROOT / "evaluation/question_dedup/newsqa_200_11064.semantic_clusters.json"
DEFAULT_APPROVAL = PROJECT_ROOT / "evaluation/question_dedup/newsqa_200_11064.human_approval.json"
DEFAULT_BASE_MANIFEST = PROJECT_ROOT / "evaluation/manifests/newsqa_200_11064.variant.json"
DEFAULT_OUTPUT_MANIFEST = PROJECT_ROOT / "evaluation/manifests/newsqa_200_11064.deduplicated.variant.json"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _verify_base_artifact(base_manifest: dict, key: str, path: Path) -> None:
    record = base_manifest.get("artifacts", {}).get(key)
    if not record or sha256_file(path) != record.get("sha256"):
        raise DatasetBuildError(f"Base artifact {key!r} does not match its manifest")


def build(args: argparse.Namespace) -> None:
    base_root = Path(args.base_root).resolve()
    output_root = Path(args.output_root).resolve()
    decisions_path = Path(args.decisions).resolve()
    approval_path = Path(args.approval).resolve()
    base_manifest_path = Path(args.base_manifest).resolve()
    output_manifest_path = Path(args.output_manifest).resolve()

    if output_root.exists() and not args.overwrite:
        raise DatasetBuildError(f"Output already exists: {output_root}; pass --overwrite")
    required = {
        name: base_root / name
        for name in (
            "testset_original.jsonl",
            "testset_reviewed_original.jsonl",
            "testset_resolved.jsonl",
            "excluded_questions.jsonl",
            "review_annotations.jsonl",
            "chunks.jsonl",
            "bm25.pkl",
        )
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise DatasetBuildError(f"Missing finalized base artifacts: {missing}")
    with base_manifest_path.open(encoding="utf-8") as handle:
        base_manifest = json.load(handle)
    with decisions_path.open(encoding="utf-8") as handle:
        decisions = json.load(handle)
    if not approval_path.exists():
        raise DatasetBuildError(f"Human approval is missing: {approval_path}")
    with approval_path.open(encoding="utf-8") as handle:
        approval = json.load(handle)
    if decisions.get("base_testset_sha256") != sha256_file(required["testset_resolved.jsonl"]):
        raise DatasetBuildError("Cluster decisions target a different resolved testset")

    for key, filename in (
        ("testset_original", "testset_original.jsonl"),
        ("testset_reviewed_original", "testset_reviewed_original.jsonl"),
        ("testset_resolved", "testset_resolved.jsonl"),
        ("excluded_questions", "excluded_questions.jsonl"),
        ("review_annotations", "review_annotations.jsonl"),
        ("chunks", "chunks.jsonl"),
        ("bm25", "bm25.pkl"),
    ):
        _verify_base_artifact(base_manifest, key, required[filename])

    reviewed = load_testset(required["testset_reviewed_original.jsonl"])
    resolved = load_testset(required["testset_resolved.jsonl"])
    source_questions = load_testset(required["testset_original.jsonl"])
    review_annotations = load_testset(required["review_annotations.jsonl"])
    exclusions = load_testset(required["excluded_questions.jsonl"])
    clusters = validate_cluster_decisions(decisions, resolved)
    validate_human_approval(
        approval,
        proposal_sha256=sha256_file(decisions_path),
        base_testset_sha256=sha256_file(required["testset_resolved.jsonl"]),
        clusters=clusters,
    )
    dedup_original, dedup_resolved, dedup_clarified, removed, cluster_records = (
        derive_deduplicated_artifacts(reviewed, resolved, clusters)
    )

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    for name in (
        "testset_original.jsonl",
        "excluded_questions.jsonl",
        "review_annotations.jsonl",
        "chunks.jsonl",
        "bm25.pkl",
    ):
        shutil.copy2(required[name], output_root / name)
    save_jsonl(dedup_original, output_root / "testset_reviewed_original.jsonl")
    save_jsonl(dedup_resolved, output_root / "testset_resolved.jsonl")
    save_jsonl(dedup_clarified, output_root / "testset_clarified.jsonl")
    save_jsonl(removed, output_root / "duplicate_questions.jsonl")
    save_jsonl(cluster_records, output_root / "question_clusters.jsonl")

    multi_clusters = [item for item in clusters if len(item["member_question_ids"]) > 1]
    integrity = {
        "schema_version": base_manifest["schema_version"],
        "deduplication_schema_version": DEDUP_SCHEMA_VERSION,
        "status": "passed",
        "phase": "semantic_deduplicated",
        "human_review_status": "approved",
        "human_reviewer_id": approval["human_review"]["reviewer_id"],
        "source_questions": len(source_questions),
        "source_review_annotations": len(review_annotations),
        "invalid_exclusions": len(exclusions),
        "pre_dedup_scored_questions": len(resolved),
        "deduplicated_questions": len(dedup_resolved),
        "deduplicated_clarified_questions": len(dedup_clarified),
        "semantic_clusters": len(clusters),
        "singleton_clusters": len(clusters) - len(multi_clusters),
        "multi_question_clusters": len(multi_clusters),
        "duplicate_questions_removed": len(removed),
        "corpus_articles": base_manifest["statistics"]["corpus_articles"],
        "distractor_articles": base_manifest["statistics"]["distractor_articles"],
        "chunks": base_manifest["database"]["chunk_count"],
        "known_distractor_normalized_duplicate_representations": 225,
    }
    _write_json(output_root / "integrity_report.json", integrity)

    artifacts = {
        key: artifact_record(output_root / filename, PROJECT_ROOT)
        for key, filename in {
            "testset_original": "testset_original.jsonl",
            "testset_reviewed_original": "testset_reviewed_original.jsonl",
            "testset_resolved": "testset_resolved.jsonl",
            "testset_clarified": "testset_clarified.jsonl",
            "excluded_questions": "excluded_questions.jsonl",
            "review_annotations": "review_annotations.jsonl",
            "question_clusters": "question_clusters.jsonl",
            "duplicate_questions": "duplicate_questions.jsonl",
            "chunks": "chunks.jsonl",
            "bm25": "bm25.pkl",
            "integrity": "integrity_report.json",
        }.items()
    }
    manifest = {
        "schema_version": base_manifest["schema_version"],
        "status": "human_approved_semantic_deduplicated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "script": "scripts/deduplicate_evaluation_dataset.py",
            "git_commit": _git_commit(),
            "python": platform.python_version(),
        },
        "parent_manifest": {
            "path": os.path.relpath(base_manifest_path, PROJECT_ROOT),
            "sha256": sha256_file(base_manifest_path),
        },
        "cluster_decisions": {
            "schema_version": DEDUP_SCHEMA_VERSION,
            "path": os.path.relpath(decisions_path, PROJECT_ROOT),
            "sha256": sha256_file(decisions_path),
            "review": decisions.get("review", {}),
        },
        "human_approval": {
            "schema_version": DEDUP_APPROVAL_SCHEMA_VERSION,
            "path": os.path.relpath(approval_path, PROJECT_ROOT),
            "sha256": sha256_file(approval_path),
            "review": approval["human_review"],
        },
        "selection_manifest": base_manifest["selection_manifest"],
        "pipeline": base_manifest["pipeline"],
        "database": base_manifest["database"],
        "statistics": integrity,
        "artifacts": artifacts,
    }
    _write_json(output_manifest_path, manifest)
    print(f"Deduplicated questions: {len(dedup_resolved)}")
    print(f"Duplicate questions removed: {len(removed)}")
    print(f"Output: {output_root}")
    print(f"Manifest: {output_manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", default=str(DEFAULT_BASE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--approval", default=str(DEFAULT_APPROVAL))
    parser.add_argument("--base-manifest", default=str(DEFAULT_BASE_MANIFEST))
    parser.add_argument("--output-manifest", default=str(DEFAULT_OUTPUT_MANIFEST))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    try:
        build(build_parser().parse_args())
        return 0
    except DatasetBuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
