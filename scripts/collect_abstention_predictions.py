#!/usr/bin/env python3
"""Collect resumable structured predictions for reviewed abstention cases."""

from __future__ import annotations

import argparse
import json
import re
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
from newsqa_rag.llm import get_llm
from newsqa_rag.response_parsing import split_citation_indices

SYSTEM_PROMPT = """You answer only from the numbered contexts. Return exactly one JSON object.
If the contexts support one defensible answer, return:
{"answerability":"answerable","answer":"concise answer","citations":[1]}
If they do not, return:
{"answerability":"insufficient_evidence","answer":null,"citations":[]}
Do not use outside knowledge. Do not repair a false premise. Citations are one-based context numbers."""

CANONICAL_ABSTENTION = "I cannot find this information in the provided context."


def _parse_json(text: str, context_count: int) -> dict:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    value = json.loads(cleaned)
    if not isinstance(value, dict) or value.get("answerability") not in {
        "answerable",
        "insufficient_evidence",
    }:
        raise ValueError("Response does not satisfy the answerability schema")
    citations = value.get("citations")
    if not isinstance(citations, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in citations
    ):
        raise ValueError("citations must be a list of integer context indices")
    if any(item < 1 or item > context_count for item in citations):
        raise ValueError("citation index is outside the supplied context")
    if value["answerability"] == "insufficient_evidence":
        if value.get("answer") is not None or citations:
            raise ValueError("Abstention requires answer=null and citations=[]")
    elif not isinstance(value.get("answer"), str) or not value["answer"].strip():
        raise ValueError("Answerable output requires a non-empty answer")
    return {
        "answerability": value["answerability"],
        "answer": value.get("answer"),
        "citations": citations,
    }


def _parse_baseline(text: str, context_count: int) -> dict:
    answer = text.strip()
    citations, invalid = split_citation_indices(answer, context_count)
    normalized = answer.strip().strip("\"'").strip()
    if normalized == CANONICAL_ABSTENTION:
        return {"answerability": "insufficient_evidence", "answer": None, "citations": []}
    return {
        "answerability": "answerable",
        "answer": answer,
        "citations": citations,
        "invalid_citations": invalid,
    }


def _ranked_ids(record: dict) -> list[str]:
    trace = record.get("trace") or record.get("result") or record
    chunks = trace.get("reranked_chunks") or trace.get("retrieved_chunks") or []
    return [str(item.get("id")) for item in chunks if isinstance(item, dict) and item.get("id")]


def _contexts_for(case: dict, chunks: dict[str, dict], retrievals: dict[str, object], depth: int) -> tuple[list[str], list[str]]:
    explicit = list(case.get("provided_context_chunk_ids") or [])
    if case["scope"] == "provided_context":
        ranked = explicit
    else:
        source = retrievals.get(case["case_id"]) or retrievals.get(case["base_question_id"]) or explicit
        ranked = _ranked_ids(source) if isinstance(source, dict) else list(source)
    excluded_chunks = set(case.get("excluded_chunk_ids") or [])
    excluded_articles = set(case.get("excluded_article_ids") or [])
    selected = []
    for chunk_id in ranked:
        chunk = chunks.get(chunk_id)
        if chunk is None or chunk_id in excluded_chunks:
            continue
        if str((chunk.get("metadata") or {}).get("article_id") or "") in excluded_articles:
            continue
        selected.append(chunk_id)
        if len(selected) == depth:
            break
    return selected, [chunks[item]["text"] for item in selected]


def _retrieval_features(case: dict, retrievals: dict[str, object]) -> dict:
    source = retrievals.get(case["case_id"]) or retrievals.get(case["base_question_id"])
    if not isinstance(source, dict):
        return {"applicable": False, "top1_reranker_score": None, "top1_top2_margin": None}
    ranked = source.get("reranked_chunks") or []
    scores = [row.get("reranker_score", row.get("score")) for row in ranked]
    numeric = [float(value) for value in scores if isinstance(value, (int, float))]
    return {
        "applicable": case.get("scope") == "full_corpus" and bool(numeric),
        "top1_reranker_score": numeric[0] if numeric else None,
        "top1_top2_margin": numeric[0] - numeric[1] if len(numeric) > 1 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--retrievals", help="Retrieval traces keyed by case_id or base question_id")
    parser.add_argument(
        "--policy",
        choices=["phase2_baseline", "structured_abstention"],
        default="structured_abstention",
    )
    parser.add_argument("--system-prompt-file", help="Required exact Phase 2 prompt for baseline policy")
    parser.add_argument("--context-depth", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--generation-min-interval-seconds", type=float, default=0.0)
    parser.add_argument("--n-eval", type=int)
    args = parser.parse_args()
    if args.context_depth < 1:
        raise SystemExit("--context-depth must be positive")
    if args.generation_min_interval_seconds < 0:
        raise SystemExit("--generation-min-interval-seconds cannot be negative")
    if args.policy == "phase2_baseline" and not args.system_prompt_file:
        raise SystemExit("--system-prompt-file is required for phase2_baseline")
    cases = load_jsonl(args.cases)
    if args.n_eval is not None:
        cases = cases[: args.n_eval]
    chunk_rows = load_jsonl(args.chunks)
    chunks = {row["id"]: row for row in chunk_rows}
    trace_rows = load_jsonl(args.retrievals) if args.retrievals else []
    retrievals: dict[str, object] = {}
    for record in trace_rows:
        key = str(record.get("case_id") or record.get("question_id") or "")
        trace = record.get("trace") or record.get("result") or record
        if key and isinstance(trace, dict):
            retrievals[key] = trace
    missing_retrievals = [
        case["case_id"]
        for case in cases
        if case.get("scope") == "full_corpus"
        and case["case_id"] not in retrievals
    ]
    if missing_retrievals:
        raise SystemExit(
            "Missing case-specific retrieval traces for full-corpus cases: "
            f"{missing_retrievals[:5]}"
        )
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    llm = get_llm(config)
    system_prompt = SYSTEM_PROMPT
    if args.policy == "phase2_baseline":
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise SystemExit("--system-prompt-file cannot be empty")
    fingerprint = stable_hash(
        {
            "cases": sha256_file(args.cases),
            "chunks": sha256_file(args.chunks),
            "retrievals": sha256_file(args.retrievals) if args.retrievals else None,
            "config": sha256_file(args.config),
            "context_depth": args.context_depth,
            "policy": args.policy,
            "system_prompt": system_prompt,
            "case_ids": [row["case_id"] for row in cases],
        }
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != fingerprint:
            raise SystemExit("Run directory belongs to another abstention configuration")
    manifest = {
        "schema_version": 1,
        "run_fingerprint": fingerprint,
        "created_at": utc_now(),
        "status": "running",
        "case_ids": [row["case_id"] for row in cases],
    }
    atomic_write_json(manifest_path, manifest)
    predictions_path = run_dir / "predictions.jsonl"
    successful = {
        row["case_id"]
        for row in load_jsonl(predictions_path) if row.get("status") == "success"
    } if predictions_path.exists() else set()
    attempts_path = run_dir / "attempts.jsonl"
    last_started_at: float | None = None
    for case in cases:
        if case["case_id"] in successful:
            continue
        context_ids, contexts = _contexts_for(case, chunks, retrievals, args.context_depth)
        context_block = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(contexts))
        prompt = f"Context:\n{context_block or '[no context supplied]'}\n\nQuestion: {case['question']}"

        def operation() -> dict:
            nonlocal last_started_at
            now = time.monotonic()
            if last_started_at is not None:
                remaining = args.generation_min_interval_seconds - (now - last_started_at)
                if remaining > 0:
                    time.sleep(remaining)
            last_started_at = time.monotonic()
            request_started = time.perf_counter()
            text = llm.generate(system_prompt, prompt)
            if args.policy == "phase2_baseline":
                parsed = _parse_baseline(text, len(contexts))
            else:
                parsed = _parse_json(text, len(contexts))
            parsed["generation_ms"] = round((time.perf_counter() - request_started) * 1000, 1)
            return parsed

        result, error, attempts = run_with_retries(
            operation,
            stage="abstention_generation",
            question_id=case["case_id"],
            attempts_path=attempts_path,
            max_attempts=args.max_attempts,
        )
        append_jsonl(
            predictions_path,
            {
                "case_id": case["case_id"],
                "question_id": case["case_id"],
                "status": "success" if result is not None else "exhausted",
                "attempt_count": attempts,
                "context_chunk_ids": context_ids,
                "answerability": result.get("answerability") if result else None,
                "answer": result.get("answer") if result else None,
                "citations": result.get("citations") if result else [],
                "invalid_citations": result.get("invalid_citations", []) if result else [],
                "policy": args.policy,
                "retrieval_features": _retrieval_features(case, retrievals),
                "generation_ms": result.get("generation_ms") if result else None,
                "error": error,
                "usage": dict(llm.last_usage or {}),
                "run_fingerprint": fingerprint,
                "finished_at": utc_now(),
            },
        )
    manifest["status"] = "complete"
    manifest["finished_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    print(f"Predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
