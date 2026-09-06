#!/usr/bin/env python3
"""Prepare and finalize a human-reviewed NewsQA abstention benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from newsqa_rag.evaluation.abstention import (
    PARTITION_TARGETS,
    SCHEMA_VERSION,
    TARGETS,
    artifact_record,
    build_review_queue,
    load_jsonl,
    save_jsonl,
    validate_cases,
)
from newsqa_rag.evaluation.benchmark_io import atomic_write_json, utc_now
from newsqa_rag.evaluation.testset import DatasetBuildError, sha256_file

DEFAULT_LOCKED_ROOT = PROJECT_ROOT / "results/datasets/phase2/data/locked-bge-m3-512-64-deduplicated-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Create a deterministic review queue")
    prepare.add_argument("--locked-root", default=str(DEFAULT_LOCKED_ROOT))
    prepare.add_argument("--output-dir", default=str(PROJECT_ROOT / "evaluation/abstention/pilot"))
    prepare.add_argument("--mode", choices=sorted(TARGETS), default="pilot")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--development-articles", type=int, default=50)
    prepare.add_argument("--retrievals", help="Successful locked-pipeline retrievals.jsonl")
    prepare.add_argument("--authored-cases", help="Reviewed proposals for external/counterfactual cases")
    prepare.add_argument(
        "--question-ids-file",
        help="Optional JSON list restricting source questions, such as Phase 2 heldout_reserve IDs",
    )
    prepare.add_argument("--overwrite", action="store_true")

    finalize = subparsers.add_parser("finalize", help="Freeze an approved review queue")
    finalize.add_argument("--review-queue", required=True)
    finalize.add_argument("--chunks", required=True)
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument("--mode", choices=sorted(TARGETS), required=True)
    finalize.add_argument("--source-manifest")
    finalize.add_argument("--overwrite", action="store_true")
    return parser


def _ensure_output(directory: Path, overwrite: bool) -> None:
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise DatasetBuildError(f"Output directory is not empty: {directory}; pass --overwrite")
    directory.mkdir(parents=True, exist_ok=True)


def _load_review_cases(path: Path) -> list[dict]:
    """Load either the canonical JSONL queue or its editable grouped JSON view."""
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("groups"), list):
            return [
                case
                for group in value["groups"]
                for case in (group.get("cases") or [])
                if isinstance(case, dict)
            ]
        raise DatasetBuildError("Readable review JSON must be a list or contain groups[].cases")
    return load_jsonl(path)


def _readable_review_document(cases: list[dict], chunks: list[dict], manifest: dict) -> dict:
    by_chunk = {str(row["id"]): row for row in chunks}

    def material(chunk_ids: list[str]) -> list[dict]:
        return [
            {"chunk_id": chunk_id, "text": by_chunk[chunk_id].get("text", "")}
            for chunk_id in chunk_ids
            if chunk_id in by_chunk
        ]

    groups = []
    for partition in ("development", "final_test"):
        for case_type in TARGETS[manifest["mode"]]:
            selected = []
            for source in cases:
                if source.get("partition") != partition or source.get("case_type") != case_type:
                    continue
                case = deepcopy(source)
                case["review_material"] = {
                    "source_gold_chunks": material(case.get("source_gold_chunk_ids") or []),
                    "provided_context_chunks": material(case.get("provided_context_chunk_ids") or []),
                    "excluded_chunks": material(case.get("excluded_chunk_ids") or []),
                }
                selected.append(case)
            if selected:
                groups.append(
                    {
                        "partition": partition,
                        "case_type": case_type,
                        "case_count": len(selected),
                        "cases": selected,
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "awaiting_human_review",
        "instructions": {
            "editable_fields": ["human_review", "secondary_review"],
            "finalize_with_this_file": True,
            "review_material_is_ignored_by_validation": True,
        },
        "summary": {
            "mode": manifest["mode"],
            "seed": manifest["seed"],
            "targets": manifest["targets"],
            "deficits": manifest["deficits"],
        },
        "groups": groups,
    }


def prepare(args: argparse.Namespace) -> dict:
    locked_root = Path(args.locked_root).resolve()
    testset_path = locked_root / "testset_resolved.jsonl"
    chunks_path = locked_root / "chunks.jsonl"
    bundle_manifest = locked_root / "bundle_manifest.json"
    for path in (testset_path, chunks_path, bundle_manifest):
        if not path.exists():
            raise DatasetBuildError(f"Missing locked artifact input: {path}")
    output = Path(args.output_dir).resolve()
    _ensure_output(output, args.overwrite)
    questions = load_jsonl(testset_path)
    if args.question_ids_file:
        requested = json.loads(Path(args.question_ids_file).read_text(encoding="utf-8"))
        if not isinstance(requested, list) or len(requested) != len(set(requested)):
            raise DatasetBuildError("--question-ids-file must contain a unique JSON list")
        requested_set = {str(value) for value in requested}
        available = {str(row["question_id"]) for row in questions}
        missing = sorted(requested_set - available)
        if missing:
            raise DatasetBuildError(f"Unknown source question IDs: {missing[:5]}")
        questions = [row for row in questions if str(row["question_id"]) in requested_set]
    chunks = load_jsonl(chunks_path)
    retrievals = load_jsonl(args.retrievals) if args.retrievals else []
    authored = load_jsonl(args.authored_cases) if args.authored_cases else []
    cases, manifest = build_review_queue(
        questions,
        chunks,
        mode=args.mode,
        seed=args.seed,
        development_articles=args.development_articles,
        retrieval_records=retrievals,
        authored_cases=authored,
    )
    review_path = output / "review_queue.jsonl"
    readable_review_path = output / "review_queue_readable.json"
    overlays_path = output / "corpus_overlays.jsonl"
    validation_path = output / "validation_report.json"
    manifest_path = output / "manifest.json"
    save_jsonl(cases, review_path)
    atomic_write_json(
        readable_review_path,
        _readable_review_document(cases, chunks, manifest),
    )
    save_jsonl(
        [
            {
                "case_id": row["case_id"],
                "scope": row["scope"],
                "provided_context_chunk_ids": row["provided_context_chunk_ids"],
                "excluded_chunk_ids": row["excluded_chunk_ids"],
                "excluded_article_ids": row["excluded_article_ids"],
            }
            for row in cases
        ],
        overlays_path,
    )
    validation = validate_cases(cases, chunks)
    atomic_write_json(validation_path, validation)
    source_bundle = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    manifest.update(
        {
            "source": {
                "locked_artifact": source_bundle.get("artifact_version"),
                "bundle_manifest_sha256": sha256_file(bundle_manifest),
                "testset_sha256": sha256_file(testset_path),
                "chunks_sha256": sha256_file(chunks_path),
            },
            "inputs": {
                "retrievals_sha256": sha256_file(args.retrievals) if args.retrievals else None,
                "authored_cases_sha256": sha256_file(args.authored_cases) if args.authored_cases else None,
                "question_ids_sha256": sha256_file(args.question_ids_file) if args.question_ids_file else None,
            },
            "source_selection": {
                "questions": len(questions),
                "articles": len({str(row["article_key"]) for row in questions}),
            },
            "artifacts": {
                "review_queue": artifact_record(review_path),
                "review_queue_readable": artifact_record(readable_review_path),
                "corpus_overlays": artifact_record(overlays_path),
                "validation_report": artifact_record(validation_path),
            },
        }
    )
    atomic_write_json(manifest_path, manifest)
    return {"output_dir": str(output), "cases": len(cases), "deficits": manifest["deficits"]}


def finalize(args: argparse.Namespace) -> dict:
    review_path = Path(args.review_queue).resolve()
    chunks_path = Path(args.chunks).resolve()
    output = Path(args.output_dir).resolve()
    if args.mode == "compact_200" and not args.source_manifest:
        raise DatasetBuildError("compact_200 finalization requires --source-manifest")
    if args.source_manifest:
        source_manifest_path = Path(args.source_manifest).resolve()
        if not source_manifest_path.is_file():
            raise DatasetBuildError(f"Missing source manifest: {source_manifest_path}")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest.get("mode") != args.mode:
            raise DatasetBuildError("Source manifest mode does not match --mode")
    _ensure_output(output, args.overwrite)
    cases = _load_review_cases(review_path)
    for case in cases:
        case.pop("review_material", None)
    chunks = load_jsonl(chunks_path)
    validation = validate_cases(cases, chunks, require_approved=True)
    counts = Counter(row.get("case_type") for row in cases)
    expected = TARGETS[args.mode]
    count_errors = {
        name: {"expected": target, "observed": counts.get(name, 0)}
        for name, target in expected.items()
        if counts.get(name, 0) != target
    }
    if count_errors:
        validation["status"] = "failed"
        validation["errors"].append(f"target_count_mismatch:{count_errors}")
    partition_errors = {}
    for partition, targets in (PARTITION_TARGETS.get(args.mode) or {}).items():
        observed = Counter(
            row.get("case_type") for row in cases if row.get("partition") == partition
        )
        mismatches = {
            name: {"expected": target, "observed": observed.get(name, 0)}
            for name, target in targets.items()
            if observed.get(name, 0) != target
        }
        if mismatches:
            partition_errors[partition] = mismatches
    if partition_errors:
        validation["status"] = "failed"
        validation["errors"].append(f"partition_target_mismatch:{partition_errors}")
    if args.mode == "compact_200":
        internal_articles = {
            str(row.get("source_article_id"))
            for row in cases
            if row.get("case_type") != "external_unanswerable"
            and row.get("source_article_id")
        }
        external_articles = {
            str(row.get("source_article_id"))
            for row in cases
            if row.get("case_type") == "external_unanswerable"
            and row.get("source_article_id")
        }
        if len(internal_articles) < 60:
            validation["status"] = "failed"
            validation["errors"].append(
                f"insufficient_internal_article_diversity:required=60:observed={len(internal_articles)}"
            )
        if len(external_articles) != 22:
            validation["status"] = "failed"
            validation["errors"].append(
                f"external_articles_must_be_unique:required=22:observed={len(external_articles)}"
            )
    validation_path = output / "validation_report.json"
    atomic_write_json(validation_path, validation)
    if validation["status"] != "passed":
        raise DatasetBuildError(
            f"Abstention dataset validation failed; inspect {validation_path}"
        )
    cases_path = output / "cases.jsonl"
    development_path = output / "development_cases.jsonl"
    final_test_path = output / "final_test_cases.jsonl"
    overlays_path = output / "corpus_overlays.jsonl"
    approval_path = output / "human_approval.json"
    save_jsonl(cases, cases_path)
    save_jsonl(
        [row for row in cases if row.get("partition") == "development"],
        development_path,
    )
    save_jsonl(
        [row for row in cases if row.get("partition") == "final_test"],
        final_test_path,
    )
    save_jsonl(
        [
            {
                "case_id": row["case_id"],
                "scope": row["scope"],
                "provided_context_chunk_ids": row["provided_context_chunk_ids"],
                "excluded_chunk_ids": row["excluded_chunk_ids"],
                "excluded_article_ids": row["excluded_article_ids"],
            }
            for row in cases
        ],
        overlays_path,
    )
    reviewers = sorted({row["human_review"].get("reviewer_id", "") for row in cases} - {""})
    approval = {
        "schema_version": SCHEMA_VERSION,
        "approved_at": utc_now(),
        "reviewers": reviewers,
        "approved_cases": len(cases),
        "review_queue_sha256": sha256_file(review_path),
    }
    atomic_write_json(approval_path, approval)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "finalized",
        "mode": args.mode,
        "targets": expected,
        "partition_targets": PARTITION_TARGETS.get(args.mode),
        "source_manifest_sha256": sha256_file(args.source_manifest) if args.source_manifest else None,
        "source_chunks_sha256": sha256_file(chunks_path),
        "artifacts": {},
    }
    manifest_path = output / "manifest.json"
    for name, path in (
        ("cases", cases_path),
        ("development_cases", development_path),
        ("final_test_cases", final_test_path),
        ("corpus_overlays", overlays_path),
        ("human_approval", approval_path),
        ("validation_report", validation_path),
    ):
        manifest["artifacts"][name] = artifact_record(path)
    atomic_write_json(manifest_path, manifest)
    return {"output_dir": str(output), "cases": len(cases), "status": "finalized"}


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = prepare(args) if args.command == "prepare" else finalize(args)
    except DatasetBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
