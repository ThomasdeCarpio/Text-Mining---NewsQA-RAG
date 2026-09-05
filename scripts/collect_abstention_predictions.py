#!/usr/bin/env python3
"""Collect resumable structured predictions for reviewed abstention cases."""

from __future__ import annotations

import argparse
import json
import re
import sys
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

SYSTEM_PROMPT = """You answer only from the numbered contexts. Return exactly one JSON object.
If the contexts support one defensible answer, return:
{"answerability":"answerable","answer":"concise answer","citations":[1]}
If they do not, return:
{"answerability":"insufficient_evidence","answer":null,"citations":[]}
Do not use outside knowledge. Do not repair a false premise. Citations are one-based context numbers."""


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


def _ranked_ids(record: dict) -> list[str]:
    trace = record.get("trace") or record.get("result") or record
    chunks = trace.get("reranked_chunks") or trace.get("retrieved_chunks") or []
    return [str(item.get("id")) for item in chunks if isinstance(item, dict) and item.get("id")]


def _contexts_for(case: dict, chunks: dict[str, dict], retrievals: dict[str, list[str]], depth: int) -> tuple[list[str], list[str]]:
    explicit = list(case.get("provided_context_chunk_ids") or [])
    if case["scope"] == "provided_context":
        ranked = explicit
    else:
        ranked = retrievals.get(case["case_id"]) or retrievals.get(case["base_question_id"]) or explicit
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--retrievals", help="Retrieval traces keyed by case_id or base question_id")
    parser.add_argument("--context-depth", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--n-eval", type=int)
    args = parser.parse_args()
    if args.context_depth < 1:
        raise SystemExit("--context-depth must be positive")
    cases = load_jsonl(args.cases)
    if args.n_eval is not None:
        cases = cases[: args.n_eval]
    chunk_rows = load_jsonl(args.chunks)
    chunks = {row["id"]: row for row in chunk_rows}
    trace_rows = load_jsonl(args.retrievals) if args.retrievals else []
    retrievals = {}
    for record in trace_rows:
        key = str(record.get("case_id") or record.get("question_id") or "")
        if key:
            retrievals[key] = _ranked_ids(record)
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    llm = get_llm(config)
    fingerprint = stable_hash(
        {
            "cases": sha256_file(args.cases),
            "chunks": sha256_file(args.chunks),
            "retrievals": sha256_file(args.retrievals) if args.retrievals else None,
            "config": sha256_file(args.config),
            "context_depth": args.context_depth,
            "case_ids": [row["case_id"] for row in cases],
            "prompt": SYSTEM_PROMPT,
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
    for case in cases:
        if case["case_id"] in successful:
            continue
        context_ids, contexts = _contexts_for(case, chunks, retrievals, args.context_depth)
        context_block = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(contexts))
        prompt = f"Context:\n{context_block or '[no context supplied]'}\n\nQuestion: {case['question']}"

        def operation() -> dict:
            return _parse_json(llm.generate(SYSTEM_PROMPT, prompt), len(contexts))

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
