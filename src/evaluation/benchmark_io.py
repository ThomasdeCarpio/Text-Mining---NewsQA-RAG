"""Durable artifacts and retry helpers for resumable benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from src.evaluation.testset import canonical_json

T = TypeVar("T")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def append_jsonl(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: str | Path, *, recover_final_line: bool = False) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    lines = target.read_text(encoding="utf-8").splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            is_final = index == len(lines) - 1
            if not recover_final_line or not is_final:
                raise ValueError(f"Malformed JSONL at {target}:{index + 1}") from exc
            quarantine = target.with_suffix(target.suffix + f".partial-{int(time.time())}")
            quarantine.write_text(line + "\n", encoding="utf-8")
            target.write_text(
                "\n".join(lines[:index]) + ("\n" if index else ""),
                encoding="utf-8",
            )
    return records


def latest_by_question(records: list[dict]) -> dict[str, dict]:
    """Return latest state while rejecting duplicate successful outputs."""
    latest: dict[str, dict] = {}
    successful: set[str] = set()
    for record in records:
        question_id = str(record.get("question_id") or "")
        if not question_id:
            raise ValueError("Benchmark record is missing question_id")
        if record.get("status") == "success":
            if question_id in successful:
                raise ValueError(f"Duplicate successful record for {question_id}")
            successful.add(question_id)
        latest[question_id] = record
    return latest


def status_code_from_exception(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return int(status) if isinstance(status, int) else None


def sanitized_error_message(exc: Exception) -> str:
    message = str(exc)[:1000]
    for name in (
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "COHERE_API_KEY",
        "GEMINI_API_KEY",
    ):
        secret = os.getenv(name, "")
        if secret:
            message = message.replace(secret, f"<{name} redacted>")
    return message


def is_retryable_exception(exc: Exception) -> bool:
    status = status_code_from_exception(exc)
    if status is not None:
        return status == 429 or status >= 500
    return isinstance(exc, (TimeoutError, ConnectionError)) or any(
        token in exc.__class__.__name__.lower()
        for token in ("timeout", "connection", "ratelimit")
    )


def is_global_fatal_exception(exc: Exception) -> bool:
    status = status_code_from_exception(exc)
    return status in {401, 403, 404}


def retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    raw_value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(raw_value) if raw_value is not None else None
    except (TypeError, ValueError):
        return None


def run_with_retries(
    operation: Callable[[], T],
    *,
    stage: str,
    question_id: str,
    attempts_path: str | Path,
    max_attempts: int = 3,
    base_delay_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T | None, dict | None, int]:
    """Run one stage with bounded transient retries and an append-only audit log."""
    for attempt in range(1, max_attempts + 1):
        started_at = utc_now()
        try:
            result = operation()
            append_jsonl(
                attempts_path,
                {
                    "question_id": question_id,
                    "stage": stage,
                    "attempt": attempt,
                    "status": "success",
                    "started_at": started_at,
                    "finished_at": utc_now(),
                },
            )
            return result, None, attempt
        except Exception as exc:
            retryable = is_retryable_exception(exc)
            error = {
                "type": exc.__class__.__name__,
                "message": sanitized_error_message(exc),
                "status_code": status_code_from_exception(exc),
                "retryable": retryable,
            }
            append_jsonl(
                attempts_path,
                {
                    "question_id": question_id,
                    "stage": stage,
                    "attempt": attempt,
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "error": error,
                },
            )
            if is_global_fatal_exception(exc):
                raise
            if not retryable or attempt == max_attempts:
                return None, error, attempt
            delay = retry_after_seconds(exc) or base_delay_seconds * (2 ** (attempt - 1))
            sleep(delay + random.uniform(0, delay * 0.25))
    raise AssertionError("retry loop exited unexpectedly")
