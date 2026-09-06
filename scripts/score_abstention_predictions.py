#!/usr/bin/env python3
"""Score structured answerability predictions for an abstention benchmark."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from newsqa_rag.evaluation.abstention import load_jsonl
from newsqa_rag.evaluation.benchmark_io import atomic_write_json, utc_now
from newsqa_rag.evaluation.metrics import evaluate_citations, evaluate_qa


CONTROLLED_CONTEXT_TYPES = {
    "controlled_context_ablation",
    "partial_weak_evidence",
}


def _conservative_label(gold: str, prediction: dict) -> tuple[str, bool]:
    label = prediction.get("answerability")
    valid = prediction.get("status") == "success" and label in {
        "answerable",
        "insufficient_evidence",
    }
    if valid:
        return str(label), True
    opposite = "answerable" if gold == "insufficient_evidence" else "insufficient_evidence"
    return opposite, False


def _metrics(pairs: list[tuple[str, str]]) -> dict:
    tp = sum(gold == pred == "insufficient_evidence" for gold, pred in pairs)
    fp = sum(gold == "answerable" and pred == "insufficient_evidence" for gold, pred in pairs)
    fn = sum(gold == "insufficient_evidence" and pred == "answerable" for gold, pred in pairs)
    tn = sum(gold == pred == "answerable" for gold, pred in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(pairs),
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),
        "abstention_f1": round(f1, 4),
        "false_answer_rate": round(fn / (tp + fn), 4) if tp + fn else 0.0,
        "false_abstention_rate": round(fp / (tn + fp), 4) if tn + fp else 0.0,
        "coverage": round((tn + fn) / len(pairs), 4) if pairs else 0.0,
        "selective_risk": round(fn / (tn + fn), 4) if tn + fn else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _cluster_bootstrap(rows: list[tuple[str, str, str]], repetitions: int, seed: int) -> dict:
    """Bootstrap by base question so derived variants are not treated as independent."""
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for base_id, gold, predicted in rows:
        grouped[base_id].append((gold, predicted))
    cluster_ids = sorted(grouped)
    if not cluster_ids or repetitions < 1:
        return {}
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(repetitions):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        metrics = _metrics([pair for cluster in sampled for pair in grouped[cluster]])
        for name in (
            "abstention_precision",
            "abstention_recall",
            "abstention_f1",
            "false_answer_rate",
            "false_abstention_rate",
            "coverage",
            "selective_risk",
        ):
            values[name].append(float(metrics[name]))
    return {
        name: {
            "lower": round(float(np.percentile(samples, 2.5)), 4),
            "upper": round(float(np.percentile(samples, 97.5)), 4),
        }
        for name, samples in sorted(values.items())
    }


def _answerable_quality(cases: list[dict], by_id: dict[str, dict]) -> dict:
    qa_samples = []
    citation_samples = []
    for case in cases:
        if case.get("answerability_label") != "answerable":
            continue
        prediction = by_id.get(case["case_id"]) or {}
        answer = prediction.get("answer") if prediction.get("answerability") == "answerable" else ""
        answer = answer or ""
        qa_samples.append(
            {
                "prediction": answer,
                "ground_truth": case.get("ground_truth") or "",
                "accepted_answers": case.get("accepted_answers") or [],
            }
        )
        context_ids = prediction.get("context_chunk_ids") or []
        citation_indices = prediction.get("citations") or []
        citation_samples.append(
            {
                "citation_chunk_ids": [
                    context_ids[index - 1]
                    for index in citation_indices
                    if isinstance(index, int) and 1 <= index <= len(context_ids)
                ],
                "invalid_citation_indices": prediction.get("invalid_citations") or [],
                "relevant_chunk_ids": case.get("gold_relevant_chunk_ids") or [],
            }
        )
    return {"qa": evaluate_qa(qa_samples), "citations": evaluate_citations(citation_samples)}


def _execution_summary(predictions: list[dict], expected: int) -> dict:
    successful = [row for row in predictions if row.get("status") == "success"]
    usage_keys = ("input_tokens", "output_tokens", "total_tokens")
    usage = {
        key: sum(int((row.get("usage") or {}).get(key, 0)) for row in successful)
        for key in usage_keys
    }
    latency = [float(row["generation_ms"]) for row in successful if isinstance(row.get("generation_ms"), (int, float))]
    return {
        "expected": expected,
        "successful": len(successful),
        "failed": expected - len(successful),
        "success_rate": round(len(successful) / expected, 4) if expected else 0.0,
        "usage": usage,
        "generation_latency_ms": {
            "mean": round(float(np.mean(latency)), 1) if latency else None,
            "p50": round(float(np.percentile(latency, 50)), 1) if latency else None,
            "p95": round(float(np.percentile(latency, 95)), 1) if latency else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cases = load_jsonl(args.cases)
    predictions = load_jsonl(args.predictions)
    by_id = {row.get("case_id"): row for row in predictions if row.get("case_id")}
    pairs = []
    successful_pairs = []
    clustered_pairs = []
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    grouped_tracks: dict[str, list[tuple[str, str]]] = defaultdict(list)
    invalid = []
    missing = []
    for case in cases:
        prediction = by_id.get(case["case_id"])
        if prediction is None:
            missing.append(case["case_id"])
            continue
        label, valid = _conservative_label(case["answerability_label"], prediction)
        if not valid:
            invalid.append(case["case_id"])
        pair = (case["answerability_label"], label)
        pairs.append(pair)
        if valid:
            successful_pairs.append(pair)
        # Article-level clusters also keep variants of different questions from
        # the same source article together. External cases use their unique
        # withheld article ID; base_question_id remains the fallback.
        cluster_id = str(case.get("source_article_id") or case["base_question_id"])
        clustered_pairs.append((cluster_id, *pair))
        grouped[case["case_type"]].append(pair)
        track = "controlled_context" if case["case_type"] in CONTROLLED_CONTEXT_TYPES else "end_to_end"
        grouped_tracks[track].append(pair)
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "coverage": {
            "expected": len(cases),
            "scored": len(successful_pairs),
            "conservative_denominator": len(pairs),
            "missing": len(missing),
            "failed_or_invalid": len(invalid),
        },
        "prediction_policies": sorted(
            {str(row.get("policy")) for row in by_id.values() if row.get("policy")}
        ),
        "overall": _metrics(pairs),
        "successful_only": _metrics(successful_pairs),
        "confidence_intervals_95": _cluster_bootstrap(
            clustered_pairs, args.bootstrap_repetitions, args.seed
        ),
        "by_case_type": {name: _metrics(values) for name, values in sorted(grouped.items())},
        "by_track": {name: _metrics(values) for name, values in sorted(grouped_tracks.items())},
        "answerable_quality": _answerable_quality(cases, by_id),
        "execution": _execution_summary(list(by_id.values()), len(cases)),
        "missing_case_ids": missing,
        "invalid_case_ids": invalid,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
