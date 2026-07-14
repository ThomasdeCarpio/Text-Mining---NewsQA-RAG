#!/usr/bin/env python3
"""Prepare, review, and finalize the locked NewsQA RAG benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.question_review import (
    PROMPT_VERSION,
    client_from_environment,
    create_review_queue,
    load_approved_annotations,
    review_status,
    run_triage,
)
from src.evaluation.testset import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_REVISION,
    DatasetBuildError,
    artifact_record,
    build_selection_bundle,
    canonical_json,
    chunk_char_ranges,
    derive_chunked_testsets,
    derive_reviewed_testsets,
    load_testset,
    map_spans_to_chunks,
    save_jsonl,
    sha256_file,
    sha256_text,
)
from src.ingestion.chunker import get_chunker


DEFAULT_ROOT = PROJECT_ROOT / "data" / "evaluation" / "newsqa_200_1000"
DEFAULT_SELECTION_MANIFEST = PROJECT_ROOT / "evaluation" / "manifests" / "newsqa_200_1000.selection.json"
DEFAULT_VARIANT_MANIFEST = PROJECT_ROOT / "evaluation" / "manifests" / "newsqa_200_1000.variant.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _write_json(path: str | Path, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "evaluation_articles": root / "staging" / "corpus" / "evaluation_articles.jsonl",
        "distractor_articles": root / "staging" / "corpus" / "distractor_articles.jsonl",
        "original_questions": root / "staging" / "questions" / "original_questions.jsonl",
        "predictions": root / "staging" / "triage" / "all_predictions.jsonl",
        "triage_manifest": root / "staging" / "triage" / "manifest.json",
        "review_jsonl": root / "staging" / "review" / "review_queue.jsonl",
        "review_csv": root / "staging" / "review" / "review_queue.csv",
        "review_readable": root / "staging" / "review" / "review_queue_readable.json",
        "testset_original": root / "final" / "testset_original.jsonl",
        "testset_clarified": root / "final" / "testset_clarified.jsonl",
        "testset_resolved": root / "final" / "testset_resolved.jsonl",
        "review_annotations": root / "final" / "review_annotations.jsonl",
        "chunks": root / "final" / "chunks.jsonl",
        "bm25": root / "final" / "bm25.pkl",
        "integrity": root / "final" / "integrity_report.json",
    }


def stage1(args: argparse.Namespace) -> None:
    root = Path(args.output_root).resolve()
    paths = _paths(root)
    print("Scanning complete NewsQA splits and selecting the locked corpus ...")
    evaluation_articles, distractor_articles, manifest = build_selection_bundle(
        dataset_name=args.dataset,
        revision=args.revision,
        evaluation_count=args.evaluation_articles,
        distractor_count=args.distractor_articles,
        seed=args.seed,
    )
    save_jsonl(evaluation_articles, paths["evaluation_articles"])
    save_jsonl(distractor_articles, paths["distractor_articles"])
    questions = [
        {
            **question,
            "article_metadata": article["metadata"],
            "dataset_split": article["split"],
        }
        for article in evaluation_articles
        for question in article["questions"]
    ]
    questions.sort(key=lambda item: item["question_id"])
    save_jsonl(questions, paths["original_questions"])

    manifest.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": {
                "script": "scripts/prepare_evaluation_dataset.py stage1",
                "git_commit": _git_commit(),
                "python": platform.python_version(),
            },
            "artifacts": {
                key: artifact_record(paths[key], PROJECT_ROOT)
                for key in ("evaluation_articles", "distractor_articles", "original_questions")
            },
            "statistics": {
                "corpus_articles": len(evaluation_articles) + len(distractor_articles),
                "evaluation_articles": len(evaluation_articles),
                "distractor_articles": len(distractor_articles),
                "evaluation_questions": len(questions),
                "evidence_spans": sum(len(item["evidence_spans"]) for item in questions),
            },
        }
    )
    _write_json(args.selection_manifest, manifest)
    print(f"Selection manifest: {args.selection_manifest}")
    print(f"Evaluation questions: {len(questions)}")

    if args.selection_only:
        print("Selection-only mode requested; LLM triage was not run.")
        return

    _run_triage(args)


def _run_triage(args: argparse.Namespace) -> None:
    root = Path(args.output_root).resolve()
    evaluation_articles, distractor_articles, paths = _load_corpus(root)
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass
    client = client_from_environment(args.model)
    print(f"Running article-batched triage with {client.model} ...")
    triage_records = run_triage(
        evaluation_articles,
        distractor_articles,
        client,
        paths["predictions"],
        requests_per_minute=args.requests_per_minute,
        max_questions_per_request=args.max_questions_per_request,
    )
    queue = create_review_queue(
        evaluation_articles,
        triage_records,
        paths["review_jsonl"],
        paths["review_csv"],
    )
    triage_manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "gemini",
        "model": client.model,
        "prompt_version": PROMPT_VERSION,
        "classified_questions": sum(len(article["questions"]) for article in evaluation_articles),
        "review_queue_questions": len(queue),
        "artifacts": {
            key: artifact_record(paths[key], PROJECT_ROOT)
            for key in ("predictions", "review_jsonl", "review_csv")
        },
    }
    if Path(args.selection_manifest).exists():
        triage_manifest["selection_manifest"] = {
            "path": os.path.relpath(args.selection_manifest, PROJECT_ROOT),
            "sha256": sha256_file(args.selection_manifest),
        }
    _write_json(paths["triage_manifest"], triage_manifest)
    print(f"Flat review queue: {paths['review_jsonl']}")
    print("Run scripts/format_review_queue.py once to create the authoritative review JSON.")
    print("Triage complete. Review can proceed independently of baseline evaluation.")


def triage_command(args: argparse.Namespace) -> None:
    _run_triage(args)


def status_command(args: argparse.Namespace) -> None:
    status = review_status(args.review_file)
    print(json.dumps(status, indent=2, sort_keys=True))
    if not status["ready"]:
        raise SystemExit(2)


def _load_corpus(root: Path) -> tuple[list[dict], list[dict], dict[str, Path]]:
    paths = _paths(root)
    required = ("evaluation_articles", "distractor_articles")
    missing = [str(paths[key]) for key in required if not paths[key].exists()]
    if missing:
        raise DatasetBuildError(f"Missing selected corpus artifacts: {missing}")
    return (
        load_testset(paths["evaluation_articles"]),
        load_testset(paths["distractor_articles"]),
        paths,
    )


def _build_context(args: argparse.Namespace, evaluation_articles: list[dict], distractor_articles: list[dict]) -> dict:
    if not Path(args.selection_manifest).exists():
        raise DatasetBuildError(f"Selection manifest is missing: {args.selection_manifest}")
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config_hash = sha256_text(canonical_json(config))
    selection_ids = [item["article_id"] for item in [*evaluation_articles, *distractor_articles]]
    selection_hash = sha256_text(canonical_json(sorted(selection_ids)))
    with open(args.selection_manifest, encoding="utf-8") as handle:
        selection_manifest = json.load(handle)
    selection_seed = selection_manifest.get("sampling", {}).get("seed", 42)
    collection_name = getattr(args, "collection", None) or (
        f"newsqa_val{len(evaluation_articles)}_s{selection_seed}_"
        f"{selection_hash[:6]}_{config_hash[:6]}"
    )
    return {
        "config": config,
        "config_hash": config_hash,
        "selection_hash": selection_hash,
        "collection_name": collection_name,
    }


def _validate_relevant_chunks(original_rows: list[dict], chunks: list[dict]) -> None:
    all_chunk_ids = {item["id"] for item in chunks}
    missing_relevant = sorted(
        {
            chunk_id
            for row in original_rows
            for chunk_id in row["relevant_chunk_ids"]
            if chunk_id not in all_chunk_ids
        }
    )
    if missing_relevant:
        raise DatasetBuildError(f"Relevant chunks are absent from corpus: {missing_relevant[:5]}")


def build_baseline(args: argparse.Namespace) -> None:
    root = Path(args.output_root).resolve()
    evaluation_articles, distractor_articles, paths = _load_corpus(root)
    context = _build_context(args, evaluation_articles, distractor_articles)
    config = context["config"]
    collection_name = context["collection_name"]

    print(
        f"Chunking the {len(evaluation_articles) + len(distractor_articles):,}-article "
        "corpus and mapping evidence ..."
    )
    original_rows, clarified_rows, chunks = derive_chunked_testsets(
        evaluation_articles,
        distractor_articles,
        get_chunker(config),
    )
    if clarified_rows:
        raise DatasetBuildError("Baseline construction unexpectedly produced clarified questions")
    save_jsonl(original_rows, paths["testset_original"])
    save_jsonl(chunks, paths["chunks"])

    indexed_count = 0
    if not args.skip_index:
        from src.indexing.bm25_index import BM25Index
        from src.indexing.chroma_store import ChromaStore
        from src.indexing.embeddings import get_embedding_function

        store = ChromaStore(args.db_path, get_embedding_function(config))
        stats = store.get_collection_stats(collection_name)
        if stats["exists"]:
            if not args.overwrite:
                raise DatasetBuildError(
                    f"Collection {collection_name!r} already exists; pass --overwrite to replace it"
                )
            store.delete_collection(collection_name)
        store.get_or_create_collection(
            collection_name, hnsw_config=config.get("database", {}).get("hnsw")
        )
        store.upsert_chunks(collection_name, chunks)
        indexed_count = store.get_collection_stats(collection_name)["count"]
        if indexed_count != len(chunks):
            raise DatasetBuildError(
                f"Chroma count mismatch: expected {len(chunks)}, got {indexed_count}"
            )
        bm25 = BM25Index()
        bm25.build(chunks)
        bm25.save(paths["bm25"])

    _validate_relevant_chunks(original_rows, chunks)

    integrity = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "status": "passed",
        "phase": "baseline_ready",
        "corpus_articles": len(evaluation_articles) + len(distractor_articles),
        "evaluation_articles": len(evaluation_articles),
        "distractor_articles": len(distractor_articles),
        "original_questions": len(original_rows),
        "chunks": len(chunks),
        "indexed_chunks": indexed_count if not args.skip_index else None,
        "review": {"state": "pending", "ready": False},
    }
    _write_json(paths["integrity"], integrity)

    variant_manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "status": "baseline_ready",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_manifest": {
            "path": os.path.relpath(args.selection_manifest, PROJECT_ROOT),
            "sha256": sha256_file(args.selection_manifest),
            "selection_sha256": context["selection_hash"],
        },
        "generator": {"git_commit": _git_commit(), "python": platform.python_version()},
        "pipeline": {
            "config_path": os.path.relpath(args.config, PROJECT_ROOT),
            "config_sha256": context["config_hash"],
            "chunking": config.get("chunking", {}),
            "embedding": config.get("embedding", {}),
        },
        "database": {
            "path": os.path.relpath(args.db_path, PROJECT_ROOT),
            "collection": context["collection_name"],
            "indexed": not args.skip_index,
            "chunk_count": len(chunks),
        },
        "statistics": integrity,
        "artifacts": {
            key: artifact_record(paths[key], PROJECT_ROOT)
            for key in ("testset_original", "chunks", "integrity")
        },
    }
    if paths["bm25"].exists() and not args.skip_index:
        variant_manifest["artifacts"]["bm25"] = artifact_record(paths["bm25"], PROJECT_ROOT)
    _write_json(args.variant_manifest, variant_manifest)
    print(f"Baseline original testset: {paths['testset_original']}")
    print(f"Collection: {collection_name} ({len(chunks)} chunks)")
    print(f"Variant manifest: {args.variant_manifest}")


def _verify_artifact(manifest: dict, key: str, expected_path: Path) -> None:
    record = manifest.get("artifacts", {}).get(key)
    if not record:
        raise DatasetBuildError(f"Baseline manifest is missing artifact {key!r}")
    if not expected_path.exists():
        raise DatasetBuildError(f"Baseline artifact is missing: {expected_path}")
    actual_hash = sha256_file(expected_path)
    if actual_hash != record.get("sha256"):
        raise DatasetBuildError(
            f"Baseline artifact {key!r} hash mismatch: expected {record.get('sha256')}, got {actual_hash}"
        )


def _map_reviewed_evidence_to_chunks(
    annotations: dict[str, dict],
    original_rows: list[dict],
    evaluation_articles: list[dict],
    chunks: list[dict],
) -> None:
    rows_by_id = {row["question_id"]: row for row in original_rows}
    articles_by_id = {article["article_id"]: article for article in evaluation_articles}
    chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
    for question_id, annotation in annotations.items():
        if not annotation.get("answer_modified"):
            continue
        row = rows_by_id[question_id]
        article = articles_by_id[row["article_key"]]
        article_chunks = [chunks_by_id[chunk_id] for chunk_id in row["article_chunk_ids"]]
        ranges = chunk_char_ranges(article["context"], article_chunks)
        annotation["relevant_chunk_ids"] = map_spans_to_chunks(
            article_chunks, ranges, annotation["evidence_spans"]
        )


def finalize(args: argparse.Namespace) -> None:
    root = Path(args.output_root).resolve()
    evaluation_articles, distractor_articles, paths = _load_corpus(root)
    required = ("predictions", "review_readable", "testset_original", "chunks")
    missing = [str(paths[key]) for key in required if not paths[key].exists()]
    if missing:
        raise DatasetBuildError(f"Missing baseline or review artifacts: {missing}")

    triage_records = load_testset(paths["predictions"])
    context = _build_context(args, evaluation_articles, distractor_articles)
    if not Path(args.variant_manifest).exists():
        raise DatasetBuildError("Baseline manifest is missing; run build-baseline before finalize")
    with open(args.variant_manifest, encoding="utf-8") as handle:
        variant_manifest = json.load(handle)
    if variant_manifest.get("status") not in {"baseline_ready", "review_complete"}:
        raise DatasetBuildError(
            f"Expected a prepared baseline manifest, got {variant_manifest.get('status')!r}"
        )
    if variant_manifest.get("pipeline", {}).get("config_sha256") != context["config_hash"]:
        raise DatasetBuildError("Current pipeline config does not match the baseline manifest")
    if variant_manifest.get("selection_manifest", {}).get("selection_sha256") != context["selection_hash"]:
        raise DatasetBuildError("Selected corpus does not match the baseline manifest")
    _verify_artifact(variant_manifest, "testset_original", paths["testset_original"])
    _verify_artifact(variant_manifest, "chunks", paths["chunks"])
    if variant_manifest.get("database", {}).get("indexed"):
        _verify_artifact(variant_manifest, "bm25", paths["bm25"])
        from src.indexing.chroma_store import ChromaStore
        from src.indexing.embeddings import get_embedding_function

        database = variant_manifest["database"]
        db_path = Path(database["path"])
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        store = ChromaStore(str(db_path), get_embedding_function(context["config"]))
        stats = store.get_collection_stats(database["collection"])
        if not stats["exists"] or stats["count"] != database["chunk_count"]:
            raise DatasetBuildError(
                f"Baseline collection mismatch: expected {database['chunk_count']} chunks, "
                f"got {stats['count']}"
            )

    original_rows = load_testset(paths["testset_original"])
    chunks = load_testset(paths["chunks"])
    _validate_relevant_chunks(original_rows, chunks)
    annotations = load_approved_annotations(
        paths["review_readable"], evaluation_articles, triage_records
    )
    _map_reviewed_evidence_to_chunks(
        annotations, original_rows, evaluation_articles, chunks
    )
    clarified_rows, resolved_rows = derive_reviewed_testsets(original_rows, annotations)
    save_jsonl(clarified_rows, paths["testset_clarified"])
    save_jsonl(resolved_rows, paths["testset_resolved"])
    save_jsonl(
        ({"question_id": question_id, **annotation} for question_id, annotation in sorted(annotations.items())),
        paths["review_annotations"],
    )

    integrity = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "status": "passed",
        "phase": "review_complete",
        "corpus_articles": len(evaluation_articles) + len(distractor_articles),
        "evaluation_articles": len(evaluation_articles),
        "distractor_articles": len(distractor_articles),
        "original_questions": len(original_rows),
        "resolved_questions": len(resolved_rows),
        "clarified_questions": len(clarified_rows),
        "chunks": len(chunks),
        "indexed_chunks": variant_manifest["database"]["chunk_count"]
        if variant_manifest["database"].get("indexed")
        else None,
        "review": review_status(paths["review_readable"]),
    }
    _write_json(paths["integrity"], integrity)
    variant_manifest.update(
        {
            "status": "review_complete",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "statistics": integrity,
        }
    )
    variant_manifest["artifacts"].update(
        {
            key: artifact_record(paths[key], PROJECT_ROOT)
            for key in (
                "testset_clarified",
                "testset_resolved",
                "review_annotations",
                "integrity",
            )
        }
    )
    _write_json(args.variant_manifest, variant_manifest)
    print(f"Resolved testset: {paths['testset_resolved']}")
    print(f"Paired clarified testset: {paths['testset_clarified']}")
    print("Review finalization reused the baseline chunks and retrieval indexes.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("stage1", help="Select corpus, run triage, and create review queue")
    prepare.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    prepare.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    prepare.add_argument("--evaluation-articles", type=_positive_int, default=200)
    prepare.add_argument("--distractor-articles", type=_positive_int, default=800)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--model", default="gemini-3.1-flash-lite")
    prepare.add_argument("--requests-per-minute", type=_positive_int, default=5)
    prepare.add_argument("--max-questions-per-request", type=_positive_int, default=25)
    prepare.add_argument("--selection-only", action="store_true")
    prepare.add_argument("--output-root", default=str(DEFAULT_ROOT))
    prepare.add_argument("--selection-manifest", default=str(DEFAULT_SELECTION_MANIFEST))
    prepare.set_defaults(func=stage1)

    triage = subparsers.add_parser(
        "triage", help="Run LLM triage over an already selected corpus"
    )
    triage.add_argument("--model", default="gemini-3.1-flash-lite")
    triage.add_argument("--requests-per-minute", type=_positive_int, default=5)
    triage.add_argument("--max-questions-per-request", type=_positive_int, default=25)
    triage.add_argument("--output-root", default=str(DEFAULT_ROOT))
    triage.add_argument("--selection-manifest", default=str(DEFAULT_SELECTION_MANIFEST))
    triage.set_defaults(func=triage_command)

    status = subparsers.add_parser("review-status", help="Validate whether human review is complete")
    status.add_argument(
        "--review-file", default=str(_paths(DEFAULT_ROOT)["review_readable"])
    )
    status.set_defaults(func=status_command)

    baseline = subparsers.add_parser(
        "build-baseline", help="Build the original testset and shared retrieval indexes"
    )
    baseline.add_argument("--output-root", default=str(DEFAULT_ROOT))
    baseline.add_argument("--selection-manifest", default=str(DEFAULT_SELECTION_MANIFEST))
    baseline.add_argument("--variant-manifest", default=str(DEFAULT_VARIANT_MANIFEST))
    baseline.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    baseline.add_argument("--db-path", default=str(PROJECT_ROOT / "data" / "chroma_db"))
    baseline.add_argument("--collection", default=None)
    baseline.add_argument("--overwrite", action="store_true")
    baseline.add_argument("--skip-index", action="store_true", help="Build files without Chroma/BM25")
    baseline.set_defaults(func=build_baseline)

    finish = subparsers.add_parser(
        "finalize", help="Validate review and derive reviewed question variants"
    )
    finish.add_argument("--output-root", default=str(DEFAULT_ROOT))
    finish.add_argument("--selection-manifest", default=str(DEFAULT_SELECTION_MANIFEST))
    finish.add_argument("--variant-manifest", default=str(DEFAULT_VARIANT_MANIFEST))
    finish.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    finish.set_defaults(func=finalize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
        return 0
    except DatasetBuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
