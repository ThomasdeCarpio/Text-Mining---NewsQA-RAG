#!/usr/bin/env python3
"""Build per-question and aggregate API usage reports for a RAGAS judge run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def _distribution(values: list[float], digits: int = 2) -> dict:
    return {
        "min": round(min(values), digits),
        "mean": round(statistics.mean(values), digits),
        "median": round(statistics.median(values), digits),
        "p95": round(_percentile(values, 0.95) or 0.0, digits),
        "max": round(max(values), digits),
    }


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def build_report(run_dir: Path) -> tuple[dict, list[dict]]:
    results_path = run_dir / "judge_results.jsonl"
    attempts_path = run_dir / "attempts.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    records = _load_jsonl(results_path)
    attempts = [row for row in _load_jsonl(attempts_path) if row.get("stage") == "judge"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    successful = [row for row in records if row.get("status") == "success"]
    if len(successful) != len({row["question_id"] for row in successful}):
        raise ValueError("judge_results.jsonl contains duplicate successful question records")

    by_batch: dict[str, list[dict]] = defaultdict(list)
    for row in successful:
        by_batch[str(row["batch_id"])].append(row)
    batch_sizes = Counter(len(rows) for rows in by_batch.values())
    exact_per_question = set(batch_sizes) == {1}
    if not exact_per_question:
        raise ValueError(
            "Exact per-question token usage is unavailable because at least one judge batch "
            "contains multiple questions"
        )

    attempts_by_batch = {}
    for attempt in attempts:
        batch_id = str(attempt.get("question_id") or "").rsplit(":", 1)[-1]
        attempts_by_batch.setdefault(batch_id, []).append(attempt)

    per_question = []
    for row in successful:
        batch_id = str(row["batch_id"])
        usage = row.get("batch_usage") or {}
        batch_attempts = attempts_by_batch.get(batch_id, [])
        successful_attempt = next(
            (item for item in reversed(batch_attempts) if item.get("status") == "success"),
            None,
        )
        elapsed_ms = float(row["batch_elapsed_ms"])
        elapsed_minutes = elapsed_ms / 60_000
        requests = int(usage.get("successful_requests", 0))
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
        per_question.append(
            {
                "question_id": row["question_id"],
                "batch_id": batch_id,
                "status": row["status"],
                "outer_attempt_count": len(batch_attempts),
                "successful_api_requests": requests,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "elapsed_seconds": round(elapsed_ms / 1000, 3),
                "effective_requests_per_minute": round(requests / elapsed_minutes, 3),
                "effective_input_tokens_per_minute": round(input_tokens / elapsed_minutes, 1),
                "effective_output_tokens_per_minute": round(output_tokens / elapsed_minutes, 1),
                "effective_total_tokens_per_minute": round(total_tokens / elapsed_minutes, 1),
                "started_at": successful_attempt.get("started_at") if successful_attempt else None,
                "finished_at": successful_attempt.get("finished_at") if successful_attempt else row.get("finished_at"),
            }
        )
    per_question.sort(key=lambda row: (row["started_at"] or "", row["question_id"]))

    unique_batch_records = [rows[0] for rows in by_batch.values()]
    total_requests = sum(int((row.get("batch_usage") or {}).get("successful_requests", 0)) for row in unique_batch_records)
    total_input = sum(int((row.get("batch_usage") or {}).get("input_tokens", 0)) for row in unique_batch_records)
    total_output = sum(int((row.get("batch_usage") or {}).get("output_tokens", 0)) for row in unique_batch_records)
    total_tokens = sum(int((row.get("batch_usage") or {}).get("total_tokens", 0)) for row in unique_batch_records)
    total_active_seconds = sum(float(row["batch_elapsed_ms"]) for row in unique_batch_records) / 1000
    starts = [_iso(row["started_at"]) for row in attempts if row.get("started_at")]
    finishes = [_iso(row["finished_at"]) for row in attempts if row.get("finished_at")]
    wall_seconds = (max(finishes) - min(starts)).total_seconds()

    requests_per_question = [float(row["successful_api_requests"]) for row in per_question]
    input_per_question = [float(row["input_tokens"]) for row in per_question]
    output_per_question = [float(row["output_tokens"]) for row in per_question]
    total_per_question = [float(row["total_tokens"]) for row in per_question]
    elapsed_per_question = [float(row["elapsed_seconds"]) for row in per_question]
    effective_rpm = [float(row["effective_requests_per_minute"]) for row in per_question]
    input_tpm = [float(row["effective_input_tokens_per_minute"]) for row in per_question]
    output_tpm = [float(row["effective_output_tokens_per_minute"]) for row in per_question]
    total_tpm = [float(row["effective_total_tokens_per_minute"]) for row in per_question]

    outer_failures = sum(row.get("status") != "success" for row in attempts)
    pricing = ((manifest.get("experiment") or {}).get("pricing") or {}).get("judge") or {}
    estimated_cost = total_input / 1_000_000 * float(pricing.get("input_per_million", 0))
    estimated_cost += total_output / 1_000_000 * float(pricing.get("output_per_million", 0))
    report = {
        "schema_version": 1,
        "source": {
            "run_directory": str(run_dir.resolve()),
            "judge_results_sha256": _sha256(results_path),
            "attempts_sha256": _sha256(attempts_path),
            "run_manifest_sha256": _sha256(manifest_path),
        },
        "configuration": {
            "provider": successful[0].get("judge_provider") if successful else None,
            "model": successful[0].get("judge_model") if successful else None,
            "reasoning_effort": successful[0].get("reasoning_effort") if successful else None,
            "judge_max_tokens": successful[0].get("judge_max_tokens") if successful else None,
            "metrics_per_question": successful[0].get("metrics") if successful else [],
            "batch_size": 1,
            "max_workers": 1,
        },
        "measurement_notes": {
            "per_question_usage_is_exact": exact_per_question,
            "successful_api_requests_source": "LangChain get_openai_callback.successful_requests",
            "total_http_attempts_known": False,
            "total_http_attempts_limitation": (
                "The callback counts successful requests. Provider-internal or RAGAS-internal "
                "failed HTTP attempts are not timestamped or counted in these artifacts."
            ),
            "exact_peak_rolling_60_second_rpm_known": False,
            "rpm_limitation": (
                "Only batch start/end timestamps are available. Effective RPM is requests divided "
                "by elapsed batch or run time, not an exact rolling-window peak."
            ),
        },
        "whole_run": {
            "questions": len(successful),
            "successful_questions": len(successful),
            "partial_or_failed_questions": len(records) - len(successful),
            "judge_batches": len(by_batch),
            "outer_batch_attempts": len(attempts),
            "outer_failed_batch_attempts": outer_failures,
            "successful_api_requests": total_requests,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "active_elapsed_seconds": round(total_active_seconds, 3),
            "wall_clock_seconds": round(wall_seconds, 3),
            "wall_clock_started_at": min(starts).isoformat(),
            "wall_clock_finished_at": max(finishes).isoformat(),
            "active_average_requests_per_minute": round(total_requests / (total_active_seconds / 60), 3),
            "wall_clock_average_requests_per_minute": round(total_requests / (wall_seconds / 60), 3),
            "wall_clock_average_input_tokens_per_minute": round(total_input / (wall_seconds / 60), 1),
            "wall_clock_average_output_tokens_per_minute": round(total_output / (wall_seconds / 60), 1),
            "wall_clock_average_total_tokens_per_minute": round(total_tokens / (wall_seconds / 60), 1),
            "average_input_tokens_per_successful_request": round(total_input / total_requests, 2),
            "average_output_tokens_per_successful_request": round(total_output / total_requests, 2),
            "average_total_tokens_per_successful_request": round(total_tokens / total_requests, 2),
            "estimated_cost_usd": round(estimated_cost, 6),
        },
        "per_question_distributions": {
            "successful_api_requests": _distribution(requests_per_question, 2),
            "input_tokens": _distribution(input_per_question, 1),
            "output_tokens": _distribution(output_per_question, 1),
            "total_tokens": _distribution(total_per_question, 1),
            "elapsed_seconds": _distribution(elapsed_per_question, 3),
            "effective_requests_per_minute": _distribution(effective_rpm, 3),
            "effective_input_tokens_per_minute": _distribution(input_tpm, 1),
            "effective_output_tokens_per_minute": _distribution(output_tpm, 1),
            "effective_total_tokens_per_minute": _distribution(total_tpm, 1),
        },
        "capacity_guidance": {
            "minimum_successful_requests_for_281_questions": total_requests,
            "minimum_requests_per_day": 4000,
            "preferred_requests_per_day": 5000,
            "recommended_requests_per_minute_minimum": 20,
            "recommended_requests_per_minute_preferred": 30,
            "recommended_total_tokens_per_minute_minimum": 30000,
            "reason": (
                "The run averaged about 10.6 RPM, while the fastest per-question batch averaged "
                f"{max(effective_rpm):.1f} RPM. The recommendation adds retry and timing headroom."
            ),
        },
        "per_question": per_question,
    }
    return report, per_question


def _write_markdown(path: Path, report: dict, json_path: Path, csv_path: Path) -> None:
    whole = report["whole_run"]
    distributions = report["per_question_distributions"]
    guidance = report["capacity_guidance"]
    lines = [
        "# Báo cáo mức sử dụng Fireworks AI - Phase 2 baseline",
        "",
        "## Cấu hình đo",
        "",
        f"- Judge: `{report['configuration']['model']}`; reasoning `{report['configuration']['reasoning_effort']}`.",
        "- RAGAS: 5 metrics, `batch_size=1`, `max_workers=1`.",
        f"- Số câu hoàn tất: **{whole['successful_questions']}/{whole['questions']}**.",
        "- Mỗi câu là một batch riêng, vì vậy token và successful request theo câu là số đo trực tiếp.",
        "",
        "## Toàn bộ 281 câu",
        "",
        "| Chỉ số | Giá trị |",
        "|---|---:|",
        f"| Successful API requests | {whole['successful_api_requests']:,} |",
        f"| Input tokens | {whole['input_tokens']:,} |",
        f"| Output tokens | {whole['output_tokens']:,} |",
        f"| Tổng tokens | {whole['total_tokens']:,} |",
        f"| Thời gian wall-clock | {whole['wall_clock_seconds']/3600:.2f} giờ |",
        f"| RPM trung bình wall-clock | {whole['wall_clock_average_requests_per_minute']:.3f} |",
        f"| Tổng TPM trung bình wall-clock | {whole['wall_clock_average_total_tokens_per_minute']:,.1f} |",
        f"| Chi phí ước tính theo manifest | ${whole['estimated_cost_usd']:.6f} |",
        "",
        "## Theo mỗi câu",
        "",
        "| Chỉ số | Min | Mean | Median | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Successful requests", "successful_api_requests"),
        ("Input tokens", "input_tokens"),
        ("Output tokens", "output_tokens"),
        ("Tổng tokens", "total_tokens"),
        ("Thời gian (giây)", "elapsed_seconds"),
        ("Effective RPM", "effective_requests_per_minute"),
        ("Effective tổng TPM", "effective_total_tokens_per_minute"),
    ):
        row = distributions[key]
        lines.append(
            f"| {label} | {row['min']:,} | {row['mean']:,} | {row['median']:,} | {row['p95']:,} | {row['max']:,} |"
        )
    lines.extend(
        [
            "",
            "## Yêu cầu dịch vụ thay thế",
            "",
            f"- Ít nhất **{guidance['minimum_successful_requests_for_281_questions']:,} successful requests** cho một lượt 281 câu.",
            f"- Tối thiểu **{guidance['minimum_requests_per_day']:,} RPD**; ưu tiên **{guidance['preferred_requests_per_day']:,} RPD** để đủ retry headroom.",
            f"- Tối thiểu **{guidance['recommended_requests_per_minute_minimum']} RPM**, ưu tiên **{guidance['recommended_requests_per_minute_preferred']} RPM** để có dư địa retry.",
            f"- Nên hỗ trợ ít nhất **{guidance['recommended_total_tokens_per_minute_minimum']:,} tổng TPM**.",
            "- Phải hỗ trợ GLM-5.3 Flash, reasoning `low`, output tối đa tối thiểu 2,048 tokens/request và OpenAI-compatible usage metadata.",
            "- Năm RAGAS metrics được tách thành nhiều judge prompt. Log không ánh xạ từng request về từng metric, nên không thể giảm quota bằng cách chia đều 12 request cho 5 metrics.",
            "",
            "## Giới hạn của phép đo",
            "",
            "`successful_requests` đến từ callback của LangChain. Artifact không ghi từng HTTP timestamp hoặc request thất bại nội bộ, nên không thể tính exact peak rolling-60-second RPM hay tổng HTTP attempts tuyệt đối. Con số 3,372 là nhu cầu successful request tối thiểu; quota nên có headroom.",
            "",
            f"Chi tiết từng câu: `{csv_path.name}`. Báo cáo máy đọc: `{json_path.name}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--json-output")
    parser.add_argument("--csv-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    json_path = Path(args.json_output) if args.json_output else run_dir / "Judge_API_Calls_stat.json"
    csv_path = Path(args.csv_output) if args.csv_output else run_dir / "Judge_API_Calls_per_question.csv"
    markdown_path = Path(args.markdown_output) if args.markdown_output else run_dir / "Judge_API_Calls_report.md"
    report, per_question = build_report(run_dir)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_question[0]))
        writer.writeheader()
        writer.writerows(per_question)
    _write_markdown(markdown_path, report, json_path, csv_path)
    print(json_path)
    print(csv_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
