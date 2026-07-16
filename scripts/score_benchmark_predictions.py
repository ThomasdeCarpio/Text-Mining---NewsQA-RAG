#!/usr/bin/env python3
"""Compute deterministic metrics from a saved benchmark inference trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_benchmark import _write_summary
from src.evaluation.benchmark_io import (
    atomic_write_json,
    latest_by_question,
    load_jsonl,
    utc_now,
)
from src.evaluation.metrics import (
    evaluate_citations,
    evaluate_qa,
    evaluate_retrieval,
    mrr_at_k,
    ndcg_at_k,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a benchmark run without calling the RAG system or an LLM judge."
    )
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def _latency_summary(values: list[float]) -> dict:
    if not values:
        return {"n_samples": 0}
    return {
        "mean_ms": round(float(np.mean(values)), 1),
        "p50_ms": round(float(np.percentile(values, 50)), 1),
        "p95_ms": round(float(np.percentile(values, 95)), 1),
        "max_ms": round(float(np.max(values)), 1),
        "n_samples": len(values),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = latest_by_question(load_jsonl(run_dir / "predictions.jsonl"))
    if not predictions:
        raise SystemExit("No prediction records are available to score")
    expected_ids = manifest.get("inputs", {}).get("question_ids", [])
    missing_ids = [question_id for question_id in expected_ids if question_id not in predictions]

    initial_samples = []
    reranked_samples = []
    qa_all = []
    qa_success = []
    citation_samples = []
    score_rows = []
    failures = []
    total_latencies = []
    stage_latencies: dict[str, list[float]] = {
        "retrieve_ms": [],
        "rerank_ms": [],
        "llm_ms": [],
    }

    for question_id in expected_ids:
        record = predictions.get(question_id)
        if record is None:
            continue
        success = record.get("status") == "success"
        result = record.get("result") if success else record.get("retrieval_trace")
        result = result or {}
        relevant_ids = record.get("relevant_chunk_ids") or []
        initial_ids = [chunk["id"] for chunk in result.get("retrieved_chunks", [])]
        reranked_ids = [chunk["id"] for chunk in result.get("reranked_chunks", [])]
        initial_sample = {
            "relevant_chunk_ids": relevant_ids,
            "retrieved_ids": initial_ids,
        }
        reranked_sample = {
            "relevant_chunk_ids": relevant_ids,
            "retrieved_ids": reranked_ids,
        }
        initial_samples.append(initial_sample)
        reranked_samples.append(reranked_sample)

        answer = result.get("answer", "") if success else ""
        qa_sample = {
            "prediction": answer,
            "ground_truth": record["ground_truth"],
            "accepted_answers": record.get("accepted_answers") or [record["ground_truth"]],
        }
        qa_all.append(qa_sample)
        if success and not manifest.get("inputs", {}).get("retrieval_only"):
            qa_success.append(qa_sample)

        citation_sample = {
            "citation_chunk_ids": result.get("citation_chunk_ids", []),
            "invalid_citation_indices": result.get("invalid_citation_indices", []),
            "relevant_chunk_ids": relevant_ids,
        }
        citation_samples.append(citation_sample)
        timing = result.get("timing_ms", {})
        for name in stage_latencies:
            if name in timing:
                stage_latencies[name].append(float(timing[name]))
        if "total_ms" in timing:
            total_latencies.append(float(timing["total_ms"]))
        elif "retrieval_total_ms" in timing:
            total_latencies.append(float(timing["retrieval_total_ms"]))

        top_k = int(manifest.get("inputs", {}).get("top_k", 10))
        top_n = int(manifest.get("inputs", {}).get("rerank_top_n", 5))
        initial_k_values = [value for value in (1, 3, 5, 10) if value <= top_k]
        reranked_k_values = [value for value in (1, 3, 5, 10) if value <= top_n]
        row = {
            "question_id": question_id,
            "status": record.get("status"),
            "retrieval": evaluate_retrieval([reranked_sample], reranked_k_values),
            "retrieval_initial": evaluate_retrieval([initial_sample], initial_k_values),
            "qa": evaluate_qa([qa_sample]),
            "citations": evaluate_citations([citation_sample]),
        }
        score_rows.append(row)
        if not success or row["retrieval"].get("hit_rate@5") == 0:
            failures.append(
                {
                    "question_id": question_id,
                    "question": record.get("question"),
                    "expected": record.get("ground_truth"),
                    "reason": (
                        record.get("error", {}).get("message", "pipeline failure")
                        if not success
                        else "No ground-truth chunk in reranked top 5"
                    ),
                }
            )

    top_k = int(manifest.get("inputs", {}).get("top_k", 10))
    top_n = int(manifest.get("inputs", {}).get("rerank_top_n", 5))
    initial_k_values = [value for value in (1, 3, 5, 10) if value <= top_k]
    reranked_k_values = [value for value in (1, 3, 5, 10) if value <= top_n]
    initial_metrics = evaluate_retrieval(initial_samples, initial_k_values)
    reranked_metrics = evaluate_retrieval(reranked_samples, reranked_k_values)
    delta_metrics = {}
    for k in sorted(set(initial_k_values) & set(reranked_k_values)):
        delta_metrics[f"delta_mrr@{k}"] = round(
            float(
                np.mean(
                    [
                        mrr_at_k(sample["relevant_chunk_ids"], sample["retrieved_ids"], k)
                        for sample in reranked_samples
                    ]
                )
                - np.mean(
                    [
                        mrr_at_k(sample["relevant_chunk_ids"], sample["retrieved_ids"], k)
                        for sample in initial_samples
                    ]
                )
            ),
            4,
        )
        delta_metrics[f"delta_ndcg@{k}"] = round(
            float(
                np.mean(
                    [
                        ndcg_at_k(sample["relevant_chunk_ids"], sample["retrieved_ids"], k)
                        for sample in reranked_samples
                    ]
                )
                - np.mean(
                    [
                        ndcg_at_k(sample["relevant_chunk_ids"], sample["retrieved_ids"], k)
                        for sample in initial_samples
                    ]
                )
            ),
            4,
        )

    successful = sum(record.get("status") == "success" for record in predictions.values())
    retrieval_only = bool(manifest.get("inputs", {}).get("retrieval_only"))
    report = {
        "schema_version": 2,
        "generated_at": utc_now(),
        "run_fingerprint": manifest.get("run_fingerprint"),
        "config": {
            **manifest.get("inputs", {}),
            "n_eval": len(expected_ids),
            "timestamp": utc_now(),
            "collection": manifest.get("paths", {}).get("collection"),
            "embedding": {},
        },
        "coverage": {
            "expected": len(expected_ids),
            "recorded": len(predictions),
            "successful": successful,
            "failed": len(predictions) - successful,
            "missing": len(missing_ids),
            "success_rate": round(successful / len(expected_ids), 4) if expected_ids else 0.0,
        },
        "retrieval_initial": initial_metrics,
        "retrieval": reranked_metrics,
        "reranker_delta": delta_metrics,
        "latency": {
            "total": _latency_summary(total_latencies),
            **{name: _latency_summary(values) for name, values in stage_latencies.items()},
        },
        "failures": failures[:100],
    }
    if not retrieval_only:
        report["qa"] = evaluate_qa(qa_all)
        report["qa_success_only"] = (
            evaluate_qa(qa_success) if qa_success else {"n_samples": 0}
        )
        report["citations"] = evaluate_citations(citation_samples)

    judge_records = latest_by_question(load_jsonl(run_dir / "judge_results.jsonl"))
    judge_success = [
        record for record in judge_records.values() if record.get("status") == "success"
    ]
    if judge_success:
        metric_names = sorted(
            {
                name
                for record in judge_success
                for name in record.get("scores", {})
            }
        )
        report["ragas"] = {
            name: round(
                float(np.mean([record["scores"][name] for record in judge_success if name in record["scores"]])),
                4,
            )
            for name in metric_names
        }
        report["ragas"]["n_samples"] = len(judge_success)
        report["ragas"]["coverage"] = round(
            len(judge_success) / successful, 4
        ) if successful else 0.0

    deterministic_path = run_dir / "deterministic_scores.jsonl"
    temporary = deterministic_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in score_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(deterministic_path)
    atomic_write_json(run_dir / "report.json", report)
    _write_summary(str(run_dir / "report_summary.txt"), report)
    print(f"Report: {run_dir / 'report.json'}")
    print(f"Coverage: {report['coverage']}")


if __name__ == "__main__":
    main()
