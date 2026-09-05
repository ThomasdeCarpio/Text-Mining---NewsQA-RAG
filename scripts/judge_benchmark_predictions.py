#!/usr/bin/env python3
"""Run resumable per-question RAGAS judging over saved predictions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from newsqa_rag.evaluation.benchmark_io import (
    append_jsonl,
    latest_by_question,
    load_jsonl,
    run_with_retries,
    stable_hash,
    utc_now,
)
from newsqa_rag.evaluation.metrics import evaluate_ragas_rows

DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge saved RAG answers without rerunning retrieval or generation."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--judge-provider",
        choices=["openai", "deepseek", "gemini", "fireworks"],
        required=True,
    )
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "max"],
        default=None,
    )
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument(
        "--results-file",
        default="judge_results.jsonl",
        help="Run-relative JSONL filename, allowing isolated judge ablations.",
    )
    parser.add_argument(
        "--require-complete-metrics",
        action="store_true",
        help="Retry a batch unless every requested metric is present for every row.",
    )
    parser.add_argument("--metrics", nargs="+", choices=DEFAULT_METRICS, default=DEFAULT_METRICS)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument(
        "--question-ids-file",
        default=None,
        help="JSON list selecting an exact, ordered subset of successful predictions.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--allow-same-judge", action="store_true")
    parser.add_argument("--enable-langsmith-tracing", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _judge_sample(record: dict) -> dict:
    result = record["result"]
    answer = re.sub(r"\[\d+]", "", result.get("answer", "")).strip()
    return {
        "question": record["question"],
        "answer": answer,
        "contexts": result.get("contexts", []),
        "ground_truth": record["ground_truth"],
    }


def main() -> None:
    args = parse_args()
    if not args.enable_langsmith_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
    if (
        args.batch_size < 1
        or args.max_workers < 1
        or args.max_attempts < 1
        or args.judge_max_tokens < 1
    ):
        raise SystemExit(
            "Batch size, workers, attempts, and judge max tokens must be at least 1"
        )
    if Path(args.results_file).name != args.results_file or not args.results_file.endswith(".jsonl"):
        raise SystemExit("--results-file must be a run-relative .jsonl filename")
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("inputs", {}).get("retrieval_only"):
        raise SystemExit("A retrieval-only run has no generated answers to judge")

    generator_model = manifest.get("inputs", {}).get("generator_model")
    generator_provider = manifest.get("inputs", {}).get("generator_provider")
    same_model = generator_model == args.judge_model and (
        generator_provider == args.judge_provider
        or generator_provider == "openai-compatible" and args.judge_provider == "openai"
    )
    if same_model and not args.allow_same_judge:
        raise SystemExit(
            "Judge and generator are the same model. Use a different judge or "
            "pass --allow-same-judge and disclose the bias."
        )

    predictions = latest_by_question(load_jsonl(run_dir / "predictions.jsonl"))
    successful = [
        predictions[question_id]
        for question_id in manifest.get("inputs", {}).get("question_ids", [])
        if question_id in predictions and predictions[question_id].get("status") == "success"
    ]
    if args.question_ids_file:
        requested = json.loads(Path(args.question_ids_file).read_text(encoding="utf-8"))
        if not isinstance(requested, list) or not requested or not all(
            isinstance(question_id, str) for question_id in requested
        ):
            raise SystemExit("--question-ids-file must contain a non-empty JSON string list")
        if len(requested) != len(set(requested)):
            raise SystemExit("--question-ids-file contains duplicate question IDs")
        available = {record["question_id"]: record for record in successful}
        missing = sorted(set(requested) - set(available))
        if missing:
            raise SystemExit(
                "Question IDs are absent from successful predictions: "
                f"{missing[:5]}"
            )
        successful = [available[question_id] for question_id in requested]
    elif args.n_eval:
        successful = random.Random(args.seed).sample(
            successful, min(args.n_eval, len(successful))
        )

    judge_fingerprint = stable_hash(
        {
            "run_fingerprint": manifest.get("run_fingerprint"),
            "judge_provider": args.judge_provider,
            "judge_model": args.judge_model,
            "reasoning_effort": args.reasoning_effort,
            "judge_max_tokens": args.judge_max_tokens,
            "metrics": args.metrics,
            "ragas_version": importlib.metadata.version("ragas"),
        }
    )
    results_path = run_dir / args.results_file
    existing_records = load_jsonl(results_path, recover_final_line=True)
    for record in existing_records:
        if record.get("judge_fingerprint") != judge_fingerprint:
            raise SystemExit(
                "Existing judge results use a different model, metric set, or run fingerprint"
            )
    existing = latest_by_question(existing_records)
    pending = []
    for record in successful:
        prior = existing.get(record["question_id"])
        if prior and prior.get("status") == "success":
            continue
        if prior and prior.get("status") == "exhausted" and not args.retry_failed:
            continue
        pending.append(record)

    batches = [pending[index:index + args.batch_size] for index in range(0, len(pending), args.batch_size)]
    iterable = batches
    if args.progress:
        from tqdm import tqdm

        iterable = tqdm(batches, desc="Judge", unit="batch")

    attempts_path = run_dir / "attempts.jsonl"
    for batch_index, batch in enumerate(iterable, 1):
        batch_id = stable_hash([record["question_id"] for record in batch])[:16]
        def evaluate_batch():
            rows, usage = evaluate_ragas_rows(
                [_judge_sample(record) for record in batch],
                metrics=args.metrics,
                llm_model=args.judge_model,
                provider=args.judge_provider,
                max_workers=args.max_workers,
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.judge_max_tokens,
                return_metadata=True,
            )
            missing = [
                sorted(set(args.metrics) - set(row))
                for row in rows
            ]
            if args.require_complete_metrics and any(missing):
                raise RuntimeError(f"RAGAS returned incomplete metrics: {missing}")
            return rows, usage

        started_at = time.perf_counter()
        payload, error, attempt_count = run_with_retries(
            evaluate_batch,
            stage="judge",
            question_id=f"judge:{judge_fingerprint[:8]}:batch:{batch_id}",
            attempts_path=attempts_path,
            max_attempts=args.max_attempts,
        )
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        if payload is None:
            for record in batch:
                append_jsonl(
                    results_path,
                    {
                        "question_id": record["question_id"],
                        "status": "exhausted",
                        "judge_fingerprint": judge_fingerprint,
                        "attempt_count": attempt_count,
                        "error": error,
                        "finished_at": utc_now(),
                    },
                )
            continue
        scores, usage = payload
        if len(scores) != len(batch):
            raise RuntimeError("RAGAS returned a different number of rows than it received")
        for record, row_scores in zip(batch, scores):
            missing_metrics = sorted(set(args.metrics) - set(row_scores))
            append_jsonl(
                results_path,
                {
                    "question_id": record["question_id"],
                    "status": "success" if not missing_metrics else "partial",
                    "judge_fingerprint": judge_fingerprint,
                    "judge_provider": args.judge_provider,
                    "judge_model": args.judge_model,
                    "reasoning_effort": args.reasoning_effort,
                    "judge_max_tokens": args.judge_max_tokens,
                    "metrics": args.metrics,
                    "scores": row_scores,
                    "missing_metrics": missing_metrics,
                    "batch_id": batch_id,
                    "batch_elapsed_ms": elapsed_ms,
                    "batch_usage": usage,
                    "attempt_count": attempt_count,
                    "finished_at": utc_now(),
                },
            )
        if not args.progress:
            print(f"Judged batch {batch_index}/{len(batches)}")

    final_records = latest_by_question(load_jsonl(results_path))
    judged = sum(record.get("status") == "success" for record in final_records.values())
    partial = sum(record.get("status") == "partial" for record in final_records.values())
    print(f"Judge results: {results_path} ({judged} complete, {partial} partial)")
    print("Run score_benchmark_predictions.py again to merge judge scores into report.json")


if __name__ == "__main__":
    main()
