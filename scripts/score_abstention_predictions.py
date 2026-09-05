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
        ):
            values[name].append(float(metrics[name]))
    return {
        name: {
            "lower": round(float(np.percentile(samples, 2.5)), 4),
            "upper": round(float(np.percentile(samples, 97.5)), 4),
        }
        for name, samples in sorted(values.items())
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
    clustered_pairs = []
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    invalid = []
    missing = []
    for case in cases:
        prediction = by_id.get(case["case_id"])
        if prediction is None:
            missing.append(case["case_id"])
            continue
        label = prediction.get("answerability")
        if label not in {"answerable", "insufficient_evidence"}:
            invalid.append(case["case_id"])
            continue
        pair = (case["answerability_label"], label)
        pairs.append(pair)
        clustered_pairs.append((case["base_question_id"], *pair))
        grouped[case["case_type"]].append(pair)
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "coverage": {
            "expected": len(cases),
            "scored": len(pairs),
            "missing": len(missing),
            "invalid_schema": len(invalid),
        },
        "overall": _metrics(pairs),
        "confidence_intervals_95": _cluster_bootstrap(
            clustered_pairs, args.bootstrap_repetitions, args.seed
        ),
        "by_case_type": {name: _metrics(values) for name, values in sorted(grouped.items())},
        "missing_case_ids": missing,
        "invalid_case_ids": invalid,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not missing and not invalid else 2


if __name__ == "__main__":
    raise SystemExit(main())
