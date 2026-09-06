#!/usr/bin/env python3
"""Select and freeze the Phase 3 abstention policy from development reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from newsqa_rag.evaluation.benchmark_io import atomic_write_json, stable_hash, utc_now
from newsqa_rag.evaluation.testset import sha256_file


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _summary(name: str, report: dict) -> dict:
    return {
        "name": name,
        "false_answer_rate": report["overall"]["false_answer_rate"],
        "false_abstention_rate": report["overall"]["false_abstention_rate"],
        "abstention_f1": report["overall"]["abstention_f1"],
        "generation_success_rate": report["execution"]["success_rate"],
        "answerable_f1": report["answerable_quality"]["qa"]["f1"],
        "citation_validity": report["answerable_quality"]["citations"]["citation_validity"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--prompt-report", required=True)
    parser.add_argument("--gated-report", required=True)
    parser.add_argument("--threshold-decision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "baseline": args.baseline_report,
        "prompt_only": args.prompt_report,
        "prompt_plus_score_gate": args.gated_report,
    }
    reports = {name: _load(path) for name, path in paths.items()}
    summaries = {name: _summary(name, report) for name, report in reports.items()}
    baseline = summaries["baseline"]
    eligible = []
    reasons = {}
    for name, summary in summaries.items():
        failures = []
        if summary["generation_success_rate"] < 0.98:
            failures.append("generation_success_rate_below_0.98")
        if summary["false_abstention_rate"] > 0.10:
            failures.append("false_abstention_rate_above_0.10")
        if summary["answerable_f1"] < baseline["answerable_f1"] - 0.02:
            failures.append("answerable_f1_drop_above_0.02")
        if summary["citation_validity"] < baseline["citation_validity"] - 0.01:
            failures.append("citation_validity_drop_above_0.01")
        reasons[name] = failures
        if not failures:
            eligible.append(name)
    complexity = {"baseline": 0, "prompt_only": 1, "prompt_plus_score_gate": 2}
    winner = None
    if eligible:
        best_far = min(summaries[name]["false_answer_rate"] for name in eligible)
        practical_ties = [
            name
            for name in eligible
            if summaries[name]["false_answer_rate"] <= best_far + 0.02
        ]
        winner = min(practical_ties, key=lambda name: complexity[name])
    threshold = _load(args.threshold_decision)
    decision = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "selected" if winner else "infeasible",
        "winner": winner,
        "summaries": summaries,
        "ineligibility_reasons": reasons,
        "guardrails": {
            "minimum_generation_success_rate": 0.98,
            "maximum_false_abstention_rate": 0.10,
            "maximum_answerable_f1_drop": 0.02,
            "maximum_citation_validity_drop": 0.01,
            "practical_false_answer_tie": 0.02,
        },
        "threshold_decision": threshold if winner == "prompt_plus_score_gate" else None,
        "inputs": {name: {"path": path, "sha256": sha256_file(path)} for name, path in paths.items()},
    }
    decision["fingerprint"] = stable_hash(decision)
    atomic_write_json(args.output, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if winner else 2


if __name__ == "__main__":
    raise SystemExit(main())
