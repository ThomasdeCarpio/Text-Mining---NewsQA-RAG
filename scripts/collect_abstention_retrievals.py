#!/usr/bin/env python3
"""Collect case-specific retrieval traces for the abstention benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from newsqa_rag.evaluation.abstention import load_jsonl
from newsqa_rag.evaluation.benchmark_io import (
    append_jsonl,
    atomic_write_json,
    run_with_retries,
    stable_hash,
    utc_now,
)
from newsqa_rag.evaluation.testset import sha256_file
from newsqa_rag.retrieval.reranker import get_reranker
from newsqa_rag.retrieval.retriever_factory import get_retriever


def _filter_results(results: list[dict], case: dict) -> list[dict]:
    excluded_chunks = set(case.get("excluded_chunk_ids") or [])
    excluded_articles = set(case.get("excluded_article_ids") or [])
    return [
        row
        for row in results
        if row.get("id") not in excluded_chunks
        and str((row.get("metadata") or {}).get("article_id") or "")
        not in excluded_articles
    ]


def _explicit_trace(case: dict, chunks: dict[str, dict], depth: int) -> dict:
    selected = []
    for chunk_id in case.get("provided_context_chunk_ids") or []:
        if chunk_id in chunks:
            selected.append({**chunks[chunk_id], "score": None, "reranker_score": None})
        if len(selected) == depth:
            break
    return {
        "question": case["question"],
        "retrieved_chunks": selected,
        "reranked_chunks": selected,
        "retrieved_ids": [row["id"] for row in selected],
        "contexts": [row.get("text", "") for row in selected],
        "timing_ms": {"retrieve_ms": 0.0, "rerank_ms": 0.0, "retrieval_total_ms": 0.0},
        "retrieval_mode": "provided_context",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--sparse-index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rerank-top-n", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1 or args.rerank_top_n < 1 or args.rerank_top_n > args.top_k:
        raise SystemExit("Require top-k >= rerank-top-n >= 1")

    cases = load_jsonl(args.cases)
    chunk_rows = load_jsonl(args.chunks)
    chunks = {row["id"]: row for row in chunk_rows}
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    retriever = get_retriever(
        "sparse",
        config,
        None,
        None,
        chunks=chunk_rows,
        bm25_path=args.sparse_index,
    )
    reranker = get_reranker(config)
    fingerprint = stable_hash(
        {
            "cases": sha256_file(args.cases),
            "chunks": sha256_file(args.chunks),
            "sparse_index": sha256_file(args.sparse_index),
            "config": sha256_file(args.config),
            "top_k": args.top_k,
            "rerank_top_n": args.rerank_top_n,
        }
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "case_retrievals.jsonl"
    attempts = run_dir / "attempts.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("run_fingerprint") != fingerprint:
            raise SystemExit("Run directory belongs to another retrieval configuration")
    completed = {
        row["case_id"]
        for row in load_jsonl(output)
        if row.get("status") == "success"
    } if output.exists() else set()
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_fingerprint": fingerprint,
            "status": "running",
            "created_at": utc_now(),
            "case_ids": [row["case_id"] for row in cases],
        },
    )

    iterable = cases
    if args.progress:
        from tqdm import tqdm

        iterable = tqdm(cases, desc="Abstention retrieval", unit="case")
    for case in iterable:
        if case["case_id"] in completed:
            continue

        def operation() -> dict:
            if case["scope"] == "provided_context":
                return _explicit_trace(case, chunks, args.rerank_top_n)
            fetch_k = min(
                len(chunk_rows),
                max(args.top_k * 5, args.top_k + len(case.get("excluded_chunk_ids") or [])),
            )
            started = time.perf_counter()
            retrieved = retriever.retrieve(case["question"], fetch_k)
            retrieved = _filter_results(retrieved, case)[: args.top_k]
            retrieve_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            reranked = reranker.rerank(case["question"], retrieved, args.rerank_top_n)
            rerank_ms = (time.perf_counter() - started) * 1000
            return {
                "question": case["question"],
                "retrieved_chunks": retrieved,
                "reranked_chunks": reranked,
                "retrieved_ids": [row["id"] for row in reranked],
                "contexts": [row.get("text", "") for row in reranked],
                "timing_ms": {
                    "retrieve_ms": round(retrieve_ms, 1),
                    "rerank_ms": round(rerank_ms, 1),
                    "retrieval_total_ms": round(retrieve_ms + rerank_ms, 1),
                },
                "retrieval_mode": "case_specific",
            }

        trace, error, attempt_count = run_with_retries(
            operation,
            stage="abstention_retrieval",
            question_id=case["case_id"],
            attempts_path=attempts,
            max_attempts=args.max_attempts,
        )
        append_jsonl(
            output,
            {
                "case_id": case["case_id"],
                "base_question_id": case["base_question_id"],
                "case_type": case["case_type"],
                "status": "success" if trace is not None else "exhausted",
                "trace": trace,
                "error": error,
                "attempt_count": attempt_count,
                "run_fingerprint": fingerprint,
                "finished_at": utc_now(),
            },
        )

    rows = load_jsonl(output)
    successful = len({row["case_id"] for row in rows if row.get("status") == "success"})
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_fingerprint": fingerprint,
            "status": "complete" if successful == len(cases) else "partial",
            "case_ids": [row["case_id"] for row in cases],
            "successful_cases": successful,
            "completed_at": utc_now(),
        },
    )
    print(f"Retrievals: {output} ({successful}/{len(cases)} successful)")
    return 0 if successful == len(cases) else 2


if __name__ == "__main__":
    raise SystemExit(main())
