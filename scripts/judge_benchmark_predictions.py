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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_io import (
    append_jsonl,
    latest_by_question,
    load_jsonl,
    run_with_retries,
    stable_hash,
    utc_now,
)
from src.evaluation.metrics import evaluate_ragas_rows

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
        choices=["openai", "deepseek", "gemini"],
        required=True,
    )
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--metrics", nargs="+", choices=DEFAULT_METRICS, default=DEFAULT_METRICS)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--n-eval", type=int, default=None)
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
    if args.batch_size < 1 or args.max_workers < 1 or args.max_attempts < 1:
        raise SystemExit(
            "--batch-size, --max-workers, and --max-attempts must be at least 1"
        )
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
    if args.n_eval:
        successful = random.Random(args.seed).sample(
            successful, min(args.n_eval, len(successful))
        )

    judge_fingerprint = stable_hash(
        {
            "run_fingerprint": manifest.get("run_fingerprint"),
            "judge_provider": args.judge_provider,
            "judge_model": args.judge_model,
            "metrics": args.metrics,
            "ragas_version": importlib.metadata.version("ragas"),
        }
    )
    results_path = run_dir / "judge_results.jsonl"
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
        scores, error, attempt_count = run_with_retries(
            lambda: evaluate_ragas_rows(
                [_judge_sample(record) for record in batch],
                metrics=args.metrics,
                llm_model=args.judge_model,
                provider=args.judge_provider,
                max_workers=args.max_workers,
            ),
            stage="judge",
            question_id=f"batch:{batch_id}",
            attempts_path=attempts_path,
            max_attempts=args.max_attempts,
        )
        if scores is None:
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
        if len(scores) != len(batch):
            raise RuntimeError("RAGAS returned a different number of rows than it received")
        if any(not row_scores for row_scores in scores):
            raise RuntimeError("RAGAS returned an empty score row")
        for record, row_scores in zip(batch, scores):
            append_jsonl(
                results_path,
                {
                    "question_id": record["question_id"],
                    "status": "success",
                    "judge_fingerprint": judge_fingerprint,
                    "judge_provider": args.judge_provider,
                    "judge_model": args.judge_model,
                    "metrics": args.metrics,
                    "scores": row_scores,
                    "attempt_count": attempt_count,
                    "finished_at": utc_now(),
                },
            )
        if not args.progress:
            print(f"Judged batch {batch_index}/{len(batches)}")

    final_records = latest_by_question(load_jsonl(results_path))
    judged = sum(record.get("status") == "success" for record in final_records.values())
    print(f"Judge results: {results_path} ({judged} successful)")
    print("Run score_benchmark_predictions.py again to merge judge scores into report.json")


if __name__ == "__main__":
    main()
