#!/usr/bin/env python3
"""Run the resumable Phase 3 development or locked final protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))
sys.path.insert(0, str(PROJECT_ROOT))

from newsqa_rag.evaluation.abstention import load_jsonl, save_jsonl
from newsqa_rag.evaluation.benchmark_io import atomic_write_json, stable_hash, utc_now
from newsqa_rag.evaluation.testset import sha256_file
from scripts.calibrate_abstention_threshold import _apply_threshold


def _run(arguments: list[object]) -> None:
    command = [sys.executable, "-u", *map(str, arguments)]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _collect(
    cases: Path,
    retrievals: Path,
    args: argparse.Namespace,
    policy: str,
    run_dir: Path,
) -> Path:
    command: list[object] = [
        "scripts/collect_abstention_predictions.py",
        "--cases", cases,
        "--chunks", args.chunks,
        "--config", args.config,
        "--retrievals", retrievals,
        "--run-dir", run_dir,
        "--policy", policy,
        "--context-depth", args.context_depth,
        "--max-attempts", args.max_attempts,
        "--generation-min-interval-seconds", args.generation_min_interval_seconds,
    ]
    if policy == "phase2_baseline":
        command.extend(["--system-prompt-file", args.phase2_prompt])
    if args.n_eval is not None:
        command.extend(["--n-eval", args.n_eval])
    _run(command)
    return run_dir / "predictions.jsonl"


def _score(cases: Path, predictions: Path, output: Path, args: argparse.Namespace) -> None:
    _run(
        [
            "scripts/score_abstention_predictions.py",
            "--cases", cases,
            "--predictions", predictions,
            "--output", output,
            "--bootstrap-repetitions", args.bootstrap_repetitions,
            "--seed", args.seed,
        ]
    )


def _retrieval(cases: Path, args: argparse.Namespace, run_dir: Path) -> Path:
    _run(
        [
            "scripts/collect_abstention_retrievals.py",
            "--cases", cases,
            "--chunks", args.chunks,
            "--sparse-index", args.sparse_index,
            "--config", args.config,
            "--run-dir", run_dir,
            "--top-k", args.top_k,
            "--rerank-top-n", args.context_depth,
            "--max-attempts", args.max_attempts,
            "--progress",
        ]
    )
    return run_dir / "case_retrievals.jsonl"


def _active_development_cases(source: Path, output: Path, n_eval: int | None) -> Path:
    if n_eval is None:
        return source
    rows = load_jsonl(source)
    selected = []
    seen_types = set()
    for row in rows:
        if row["case_type"] not in seen_types:
            selected.append(row)
            seen_types.add(row["case_type"])
        if len(selected) == n_eval:
            break
    selected_ids = {row["case_id"] for row in selected}
    selected.extend(
        row for row in rows if row["case_id"] not in selected_ids
    )
    selected = selected[:n_eval]
    target = output / "smoke_cases.jsonl"
    save_jsonl(selected, target)
    return target


def _write_stage_manifest(
    stage: str,
    cases: Path,
    output: Path,
    args: argparse.Namespace,
    extra: dict | None = None,
) -> None:
    record = {
        "schema_version": 1,
        "created_at": utc_now(),
        "stage": stage,
        "inputs": {
            "cases_sha256": sha256_file(cases),
            "chunks_sha256": sha256_file(args.chunks),
            "sparse_index_sha256": sha256_file(args.sparse_index),
            "config_sha256": sha256_file(args.config),
            "phase2_prompt_sha256": sha256_file(args.phase2_prompt),
        },
        "settings": {
            "top_k": args.top_k,
            "context_depth": args.context_depth,
            "max_attempts": args.max_attempts,
            "generation_min_interval_seconds": args.generation_min_interval_seconds,
            "seed": args.seed,
            "n_eval": args.n_eval,
        },
        **(extra or {}),
    }
    record["fingerprint"] = stable_hash(record)
    atomic_write_json(output / "stage_manifest.json", record)


def _development(args: argparse.Namespace, output: Path) -> None:
    cases = _active_development_cases(
        Path(args.cases_root) / "development_cases.jsonl", output, args.n_eval
    )
    retrievals = _retrieval(cases, args, output / "retrieval_development")
    baseline_predictions = _collect(
        cases, retrievals, args, "phase2_baseline", output / "development_baseline"
    )
    prompt_predictions = _collect(
        cases, retrievals, args, "structured_abstention", output / "development_prompt"
    )
    baseline_report = output / "development_baseline_report.json"
    prompt_report = output / "development_prompt_report.json"
    _score(cases, baseline_predictions, baseline_report, args)
    _score(cases, prompt_predictions, prompt_report, args)
    calibration = output / "calibration"
    _run(
        [
            "scripts/calibrate_abstention_threshold.py",
            "--development-cases", cases,
            "--development-predictions", prompt_predictions,
            "--output-dir", calibration,
            "--max-false-abstention-rate", args.max_false_abstention_rate,
        ]
    )
    gated_predictions = calibration / "development_gated_predictions.jsonl"
    gated_report = output / "development_gated_report.json"
    _score(cases, gated_predictions, gated_report, args)
    _run(
        [
            "scripts/select_abstention_policy.py",
            "--baseline-report", baseline_report,
            "--prompt-report", prompt_report,
            "--gated-report", gated_report,
            "--threshold-decision", calibration / "threshold_decision.json",
            "--output", output / "winner_decision.json",
        ]
    )
    _write_stage_manifest(
        "development", cases, output, args,
        {"winner_decision_sha256": sha256_file(output / "winner_decision.json")},
    )


def _final(args: argparse.Namespace, output: Path) -> None:
    decision_path = Path(args.winner_decision or "")
    if not decision_path.is_file():
        raise SystemExit("--winner-decision is required for final stage")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "selected" or decision.get("winner") not in {
        "baseline", "prompt_only", "prompt_plus_score_gate"
    }:
        raise SystemExit("Final stage requires a valid locked development winner")
    cases = Path(args.cases_root) / "final_test_cases.jsonl"
    retrievals = _retrieval(cases, args, output / "retrieval_final")
    baseline_predictions = _collect(
        cases, retrievals, args, "phase2_baseline", output / "final_baseline"
    )
    prompt_predictions = _collect(
        cases, retrievals, args, "structured_abstention", output / "final_prompt"
    )
    threshold_record = decision.get("threshold_decision") or {}
    selected_threshold = (threshold_record.get("selected") or {}).get("threshold")
    if selected_threshold is None:
        source = Path(args.threshold_decision or "")
        if source.is_file():
            selected_threshold = (
                json.loads(source.read_text(encoding="utf-8")).get("selected") or {}
            ).get("threshold")
    if selected_threshold is None:
        raise SystemExit("Final protocol requires the locked development threshold")
    gated_predictions = output / "final_gated_predictions.jsonl"
    save_jsonl(
        _apply_threshold(
            load_jsonl(cases), load_jsonl(prompt_predictions), float(selected_threshold)
        ),
        gated_predictions,
    )
    reports = {
        "baseline": (baseline_predictions, output / "final_baseline_report.json"),
        "prompt_only": (prompt_predictions, output / "final_prompt_report.json"),
        "prompt_plus_score_gate": (gated_predictions, output / "final_gated_report.json"),
    }
    for predictions, report in reports.values():
        _score(cases, predictions, report, args)
    locked = {
        "schema_version": 1,
        "winner": decision["winner"],
        "development_decision": str(decision_path.resolve()),
        "final_report": str(reports[decision["winner"]][1].resolve()),
        "threshold": selected_threshold,
        "no_final_reselection": True,
    }
    atomic_write_json(output / "final_protocol_record.json", locked)
    _write_stage_manifest(
        "final", cases, output, args,
        {
            "development_winner_decision_sha256": sha256_file(decision_path),
            "locked_winner": decision["winner"],
            "no_final_reselection": True,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["development", "final"], required=True)
    parser.add_argument("--cases-root", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--sparse-index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase2-prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--winner-decision")
    parser.add_argument("--threshold-decision")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--context-depth", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--generation-min-interval-seconds", type=float, default=4.2)
    parser.add_argument("--max-false-abstention-rate", type=float, default=0.10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, help="Smoke-only prefix; never use for reported runs")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "development":
        _development(args, output)
    else:
        if args.n_eval is not None:
            raise SystemExit("--n-eval is forbidden for the locked final stage")
        _final(args, output)
    print(f"Phase 3 {args.stage} outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
