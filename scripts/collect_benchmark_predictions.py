#!/usr/bin/env python3
"""Collect resumable retrieval and generation traces for the NewsQA benchmark."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from newsqa_rag.agents.rag_agent import RAGAgent
from newsqa_rag.evaluation.benchmark_io import (
    apply_manifest_preflight,
    append_jsonl,
    atomic_write_json,
    latest_by_question,
    load_jsonl,
    run_with_retries,
    stable_hash,
    utc_now,
)
from newsqa_rag.evaluation.testset import load_testset, sha256_file
from newsqa_rag.indexing.chroma_store import ChromaStore
from newsqa_rag.indexing.embeddings import get_embedding_function
from newsqa_rag.ingestion.chunker import load_chunks
from newsqa_rag.llm import OpenAILLM, get_llm
from newsqa_rag.model_gateway import DEEPSEEK_BASE_URL, load_generation_client_settings
from newsqa_rag.retrieval.reranker import get_reranker
from newsqa_rag.retrieval.retriever_factory import get_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a durable, resumable RAG inference trace."
    )
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--reranker", choices=["noop", "cross-encoder"], default="noop")
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    parser.add_argument("--testset", required=True)
    parser.add_argument("--variant-manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--chunks-path", default=None)
    parser.add_argument("--bm25-path", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rerank-top-n", type=int, default=None)
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--question-ids-file",
        default=None,
        help="JSON list of locked question IDs to evaluate after manifest preflight.",
    )
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _implementation_hash() -> str:
    paths = [
        Path(__file__),
        PROJECT_ROOT / "backend/newsqa_rag/agents/rag_agent.py",
        PROJECT_ROOT / "backend/newsqa_rag/llm.py",
        PROJECT_ROOT / "backend/newsqa_rag/retrieval/retriever_factory.py",
        PROJECT_ROOT / "backend/newsqa_rag/retrieval/reranker.py",
    ]
    return stable_hash({str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths})


def _common_record(entry: dict, fingerprint: str) -> dict:
    return {
        "run_fingerprint": fingerprint,
        "question_id": entry.get("question_id"),
        "source_question_id": entry.get("source_question_id", entry.get("question_id")),
        "article_key": entry.get("article_key"),
        "question_variant": entry.get("question_variant", "original"),
        "standalone_label": entry.get("standalone_label", "unlabeled"),
        "answer_modified": bool(entry.get("answer_modified", False)),
        "question": entry["question"],
        "ground_truth": entry["ground_truth"],
        "accepted_answers": entry.get("accepted_answers") or [entry["ground_truth"]],
        "relevant_chunk_ids": entry["relevant_chunk_ids"],
    }


def main() -> None:
    args = parse_args()
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")
    if args.n_eval is not None and args.n_eval < 1:
        raise SystemExit("--n-eval must be at least 1")

    config_path = PROJECT_ROOT / args.config
    with config_path.open(encoding="utf-8") as handle:
        original_config = yaml.safe_load(handle) or {}

    apply_manifest_preflight(args, original_config)
    config = copy.deepcopy(original_config)
    config.setdefault("retrieval", {}).setdefault("reranker", {}).update(
        {"type": args.reranker, "model": args.reranker_model}
    )
    if args.generator_model:
        config.setdefault("llm", {})["model"] = args.generator_model

    retrieval_config = config.get("retrieval", {})
    top_k = args.top_k or int(retrieval_config.get("top_k", 10))
    top_n = args.rerank_top_n or int(
        retrieval_config.get("reranker", {}).get("top_n", 5)
    )
    entries = [entry for entry in load_testset(args.testset) if entry.get("relevant_chunk_ids")]
    if args.question_ids_file:
        requested = json.loads(Path(args.question_ids_file).read_text(encoding="utf-8"))
        if not isinstance(requested, list) or not requested or not all(
            isinstance(question_id, str) for question_id in requested
        ):
            raise SystemExit("--question-ids-file must contain a non-empty JSON string list")
        if len(requested) != len(set(requested)):
            raise SystemExit("--question-ids-file contains duplicate question IDs")
        available = {entry["question_id"] for entry in entries}
        missing = sorted(set(requested) - available)
        if missing:
            raise SystemExit(f"Question IDs are absent from the testset: {missing[:5]}")
        requested_set = set(requested)
        entries = [entry for entry in entries if entry["question_id"] in requested_set]
    if args.n_eval:
        entries = sorted(
            entries,
            key=lambda entry: stable_hash(
                {
                    "seed": args.seed,
                    "question_id": entry.get("source_question_id", entry["question_id"]),
                }
            ),
        )[: args.n_eval]
    if not entries:
        raise SystemExit("No scorable questions were found")

    generator_model = None
    generator_provider = None
    if not args.retrieval_only:
        generator_model = config.get("llm", {}).get("model", "gpt-4o-mini")
        settings = load_generation_client_settings(generator_model)
        generator_model = settings.model
        generator_provider = (
            "deepseek" if settings.base_url == DEEPSEEK_BASE_URL else "openai-compatible"
        )

    selected_ids = [entry["question_id"] for entry in entries]
    fingerprint_payload = {
        "testset_sha256": sha256_file(args.testset),
        "variant_manifest_sha256": sha256_file(args.variant_manifest),
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": _implementation_hash(),
        "question_ids": selected_ids,
        "retriever": args.retriever,
        "reranker": args.reranker,
        "reranker_model": args.reranker_model if args.reranker == "cross-encoder" else None,
        "generator_provider": generator_provider,
        "generator_model": generator_model,
        "rag_prompt": OpenAILLM.DEFAULT_SYSTEM_PROMPT,
        "top_k": top_k,
        "rerank_top_n": top_n,
        "retrieval_only": args.retrieval_only,
        "seed": args.seed,
    }
    fingerprint = stable_hash(fingerprint_payload)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "run_fingerprint": fingerprint,
        "created_at": utc_now(),
        "status": "running",
        "n_questions": len(entries),
        "inputs": fingerprint_payload,
        "paths": {
            "testset": str(Path(args.testset)),
            "variant_manifest": str(Path(args.variant_manifest)),
            "database": str(Path(args.db_path)),
            "collection": args.collection,
            "chunks": args.chunks_path,
            "bm25": args.bm25_path,
        },
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("run_fingerprint") != fingerprint:
            raise SystemExit("Run directory belongs to an incompatible benchmark fingerprint")
        manifest = existing_manifest
        manifest["status"] = "running"
        manifest["resumed_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)

    attempts_path = run_dir / "attempts.jsonl"
    retrievals_path = run_dir / "retrievals.jsonl"
    predictions_path = run_dir / "predictions.jsonl"
    retrieval_cache = latest_by_question(
        load_jsonl(retrievals_path, recover_final_line=True)
    )
    predictions = latest_by_question(
        load_jsonl(predictions_path, recover_final_line=True)
    )

    embedding_function = get_embedding_function(original_config)
    store = ChromaStore(args.db_path, embedding_function)
    chunks = None
    if args.retriever in {"bm25", "hybrid"}:
        chunks = load_chunks(args.chunks_path)
    retriever = get_retriever(
        args.retriever,
        config,
        store,
        args.collection,
        chunks=chunks,
        bm25_path=args.bm25_path,
    )
    reranker = get_reranker(config)
    llm = None if args.retrieval_only else get_llm(config)
    agent = RAGAgent(retriever, reranker, llm, top_k=top_k, rerank_top_n=top_n)

    iterable = entries
    if args.progress:
        from tqdm import tqdm

        iterable = tqdm(entries, desc="Collect", unit="question")

    try:
        for entry in iterable:
            question_id = entry["question_id"]
            prior = predictions.get(question_id)
            if prior and prior.get("status") == "success":
                continue
            if prior and prior.get("status") == "exhausted" and not args.retry_failed:
                continue

            trace_record = retrieval_cache.get(question_id)
            if trace_record is None:
                trace, error, retrieval_attempts = run_with_retries(
                    lambda: agent.retrieve_and_rerank(entry["question"]),
                    stage="retrieval",
                    question_id=question_id,
                    attempts_path=attempts_path,
                    max_attempts=args.max_attempts,
                )
                if trace is None:
                    record = {
                        **_common_record(entry, fingerprint),
                        "status": "exhausted",
                        "failed_stage": "retrieval",
                        "attempt_count": retrieval_attempts,
                        "error": error,
                        "finished_at": utc_now(),
                    }
                    append_jsonl(predictions_path, record)
                    predictions[question_id] = record
                    continue
                trace_record = {
                    **_common_record(entry, fingerprint),
                    "status": "success",
                    "trace": trace,
                    "attempt_count": retrieval_attempts,
                    "finished_at": utc_now(),
                }
                append_jsonl(retrievals_path, trace_record)
                retrieval_cache[question_id] = trace_record

            trace = trace_record["trace"]
            if args.retrieval_only:
                result = {
                    **trace,
                    "answer": "",
                    "citation_indices": [],
                    "citation_chunk_ids": [],
                    "invalid_citation_indices": [],
                    "cited_chunks": [],
                }
                generation_attempts = 0
                error = None
            else:
                result, error, generation_attempts = run_with_retries(
                    lambda: agent.generate_from_trace(trace),
                    stage="generation",
                    question_id=question_id,
                    attempts_path=attempts_path,
                    max_attempts=args.max_attempts,
                )

            if result is None:
                record = {
                    **_common_record(entry, fingerprint),
                    "status": "exhausted",
                    "failed_stage": "generation",
                    "retrieval_trace": trace,
                    "attempt_count": generation_attempts,
                    "error": error,
                    "finished_at": utc_now(),
                }
            else:
                record = {
                    **_common_record(entry, fingerprint),
                    "status": "success",
                    "result": result,
                    "attempt_count": generation_attempts,
                    "generator_provider": generator_provider,
                    "generator_model": getattr(llm, "_effective_model", generator_model),
                    "finished_at": utc_now(),
                }
            append_jsonl(predictions_path, record)
            predictions[question_id] = record
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["interrupted_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        raise SystemExit(130)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["failure_type"] = exc.__class__.__name__
        atomic_write_json(manifest_path, manifest)
        raise

    successful = sum(record.get("status") == "success" for record in predictions.values())
    exhausted = sum(record.get("status") == "exhausted" for record in predictions.values())
    manifest.update(
        {
            "status": "complete" if successful + exhausted == len(entries) else "partial",
            "completed_at": utc_now(),
            "successful_questions": successful,
            "exhausted_questions": exhausted,
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(f"Run status: {manifest['status']} ({successful} success, {exhausted} exhausted)")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
