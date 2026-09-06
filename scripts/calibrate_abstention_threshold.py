#!/usr/bin/env python3
"""Calibrate a development-only reranker threshold and derive gated predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))
sys.path.insert(0, str(PROJECT_ROOT))

from newsqa_rag.evaluation.abstention import load_jsonl, save_jsonl
from newsqa_rag.evaluation.benchmark_io import atomic_write_json, stable_hash, utc_now
from newsqa_rag.evaluation.testset import sha256_file
from scripts.score_abstention_predictions import _conservative_label, _metrics


def _apply_threshold(cases: list[dict], predictions: list[dict], threshold: float) -> list[dict]:
    by_id = {row["case_id"]: row for row in predictions}
    output = []
    for case in cases:
        source = by_id.get(case["case_id"])
        if source is None:
            continue
        row = dict(source)
        features = row.get("retrieval_features") or {}
        score = features.get("top1_reranker_score")
        rejected = bool(features.get("applicable")) and isinstance(score, (int, float)) and score < threshold
        if rejected:
            row.update(
                {
                    "answerability": "insufficient_evidence",
                    "answer": None,
                    "citations": [],
                }
            )
        row["policy"] = "structured_abstention_with_score_gate"
        row["gate"] = {
            "applicable": bool(features.get("applicable")),
            "rejected": rejected,
            "threshold": threshold,
            "score": score,
        }
        output.append(row)
    return output


def _evaluate(cases: list[dict], predictions: list[dict]) -> dict:
    by_id = {row["case_id"]: row for row in predictions}
    pairs = [
        (
            case["answerability_label"],
            _conservative_label(case["answerability_label"], by_id[case["case_id"]])[0],
        )
        for case in cases
        if case["case_id"] in by_id
    ]
    return _metrics(pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-cases", required=True)
    parser.add_argument("--development-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-false-abstention-rate", type=float, default=0.10)
    parser.add_argument("--apply-cases")
    parser.add_argument("--apply-predictions")
    args = parser.parse_args()
    if not 0 <= args.max_false_abstention_rate <= 1:
        raise SystemExit("--max-false-abstention-rate must be between zero and one")
    dev_cases = load_jsonl(args.development_cases)
    dev_predictions = load_jsonl(args.development_predictions)
    scores = sorted(
        {
            float((row.get("retrieval_features") or {})["top1_reranker_score"])
            for row in dev_predictions
            if (row.get("retrieval_features") or {}).get("applicable")
            and isinstance((row.get("retrieval_features") or {}).get("top1_reranker_score"), (int, float))
        }
    )
    if not scores:
        raise SystemExit("No applicable top-one reranker scores were found")
    candidates = [scores[0] - 1e-12, *scores, scores[-1] + 1e-12]
    rows = []
    for threshold in candidates:
        metrics = _evaluate(dev_cases, _apply_threshold(dev_cases, dev_predictions, threshold))
        rows.append({"threshold": threshold, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
    eligible = [row for row in rows if row["false_abstention_rate"] <= args.max_false_abstention_rate]
    selected = None
    if eligible:
        selected = sorted(
            eligible,
            key=lambda row: (
                row["false_answer_rate"],
                -row["abstention_f1"],
                row["threshold"],
            ),
        )[0]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    curve_path = output / "threshold_curve.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decision = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "selected" if selected else "infeasible",
        "feature": "top1_reranker_score",
        "rule": "abstain_when_score_below_threshold",
        "max_false_abstention_rate": args.max_false_abstention_rate,
        "selected": selected,
        "development_cases_sha256": sha256_file(args.development_cases),
        "development_predictions_sha256": sha256_file(args.development_predictions),
        "fingerprint": stable_hash(
            {
                "cases": sha256_file(args.development_cases),
                "predictions": sha256_file(args.development_predictions),
                "constraint": args.max_false_abstention_rate,
            }
        ),
    }
    atomic_write_json(output / "threshold_decision.json", decision)
    if selected:
        dev_gated = _apply_threshold(dev_cases, dev_predictions, selected["threshold"])
        save_jsonl(dev_gated, output / "development_gated_predictions.jsonl")
        if bool(args.apply_cases) != bool(args.apply_predictions):
            raise SystemExit("--apply-cases and --apply-predictions must be supplied together")
        if args.apply_cases:
            save_jsonl(
                _apply_threshold(
                    load_jsonl(args.apply_cases),
                    load_jsonl(args.apply_predictions),
                    selected["threshold"],
                ),
                output / "final_gated_predictions.jsonl",
            )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
