"""Validate XAH model gateway contracts without printing sensitive content.

Examples:
    python scripts/check_model_gateway.py
    python scripts/check_model_gateway.py --chat-model MODEL --embedding-model MODEL
    python scripts/check_model_gateway.py --responses-model MODEL --stream
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.xah.io/v1"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one model gateway contract check.

    Args:
        name: Human-readable check name.
        passed: Whether the endpoint returned the expected contract.
        status_code: HTTP response status, or zero for a transport failure.
        latency_ms: End-to-end request latency in milliseconds.
        detail: Non-sensitive result summary.
    """

    name: str
    passed: bool
    status_code: int
    latency_ms: float
    detail: str


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the gateway smoke-test command-line parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(
        description="Check model gateway response contracts without printing generated content."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL ending in /v1.",
    )
    parser.add_argument("--chat-model", help="Model ID for /v1/chat/completions.")
    parser.add_argument("--responses-model", help="Model ID for /v1/responses.")
    parser.add_argument("--messages-model", help="Model ID for /v1/messages.")
    parser.add_argument("--embedding-model", help="Model ID for /v1/embeddings.")
    parser.add_argument("--embedding-dimensions", type=int, help="Expected embedding vector dimensions.")
    parser.add_argument("--gemini-model", help="Model ID for Gemini generateContent.")
    parser.add_argument("--ollama-model", help="Model ID for /api/chat.")
    parser.add_argument("--stream", action="store_true", help="Validate SSE for core generation endpoints.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds. Default: 60")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run configured gateway checks and return a process exit code.

    Args:
        argv: Optional CLI arguments. Uses ``sys.argv`` when omitted.

    Returns:
        Zero when every selected check passes, otherwise one or two.
    """

    load_dotenv(PROJECT_ROOT / ".env")
    args = build_arg_parser().parse_args(argv)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY is required.", file=sys.stderr)
        return 2

    openai_base = args.base_url.rstrip("/")
    gateway_root = _gateway_root(openai_base)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    results: list[CheckResult] = []

    with httpx.Client(timeout=args.timeout, follow_redirects=True) as client:
        results.append(
            _check_json(
                client,
                "models",
                "GET",
                f"{openai_base}/models",
                headers,
                None,
                lambda body: isinstance(body.get("data"), list),
                "model list",
            )
        )

        generation_checks = (
            (
                "chat",
                args.chat_model,
                f"{openai_base}/chat/completions",
                {"messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 8},
                lambda body: isinstance(body.get("choices"), list),
            ),
            (
                "responses",
                args.responses_model,
                f"{openai_base}/responses",
                {"input": "Reply with OK.", "max_output_tokens": 8},
                lambda body: "output" in body or "output_text" in body,
            ),
            (
                "messages",
                args.messages_model,
                f"{openai_base}/messages",
                {"messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 8},
                lambda body: isinstance(body.get("content"), list),
            ),
        )
        for name, model, url, payload, validator in generation_checks:
            if not model:
                continue
            payload["model"] = model
            request_headers = dict(headers)
            if name == "messages":
                request_headers.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
            if args.stream:
                payload["stream"] = True
                results.append(_check_sse(client, name, url, request_headers, payload))
            else:
                results.append(
                    _check_json(client, name, "POST", url, request_headers, payload, validator, "response shape")
                )

        if args.embedding_model:
            payload = {"model": args.embedding_model, "input": ["gateway health check"]}
            if args.embedding_dimensions:
                payload["dimensions"] = args.embedding_dimensions
            results.append(
                _check_embedding(
                    client,
                    f"{openai_base}/embeddings",
                    headers,
                    payload,
                    args.embedding_dimensions,
                )
            )

        if args.gemini_model:
            model_path = quote(args.gemini_model, safe="")
            results.append(
                _check_json(
                    client,
                    "gemini",
                    "POST",
                    f"{gateway_root}/v1beta/models/{model_path}:generateContent",
                    headers,
                    {"contents": [{"parts": [{"text": "Reply with OK."}]}]},
                    lambda body: isinstance(body.get("candidates"), list),
                    "candidate list",
                )
            )

        if args.ollama_model:
            results.append(
                _check_json(
                    client,
                    "ollama",
                    "POST",
                    f"{gateway_root}/api/chat",
                    headers,
                    {
                        "model": args.ollama_model,
                        "messages": [{"role": "user", "content": "Reply with OK."}],
                        "stream": False,
                    },
                    lambda body: isinstance(body.get("message"), dict),
                    "message object",
                )
            )

    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        print(
            f"[{marker}] {result.name}: status={result.status_code} "
            f"latency_ms={result.latency_ms:.1f} detail={result.detail}"
        )
    return 0 if all(result.passed for result in results) else 1


def _check_json(
    client: httpx.Client,
    name: str,
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict | None,
    validator: Callable[[dict], bool],
    success_detail: str,
) -> CheckResult:
    """Call a JSON endpoint and validate its top-level response shape.

    Args:
        client: Configured synchronous HTTP client.
        name: Human-readable check name.
        method: HTTP method to send.
        url: Absolute endpoint URL.
        headers: Authentication and content headers.
        payload: Optional JSON request body.
        validator: Function that validates the decoded JSON object.
        success_detail: Non-sensitive summary for a successful check.

    Returns:
        CheckResult containing status, latency, and validation outcome.
    """

    started = time.perf_counter()
    try:
        response = client.request(method, url, headers=headers, json=payload)
        latency_ms = (time.perf_counter() - started) * 1000
        if not response.is_success:
            return CheckResult(name, False, response.status_code, latency_ms, "HTTP error")
        body = response.json()
        passed = isinstance(body, dict) and validator(body)
        return CheckResult(
            name,
            passed,
            response.status_code,
            latency_ms,
            success_detail if passed else "unexpected JSON shape",
        )
    except (httpx.HTTPError, ValueError) as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return CheckResult(name, False, 0, latency_ms, type(exc).__name__)


def _check_sse(
    client: httpx.Client,
    name: str,
    url: str,
    headers: dict[str, str],
    payload: dict,
) -> CheckResult:
    """Call a streaming endpoint and verify that it emits SSE frames.

    Args:
        client: Configured synchronous HTTP client.
        name: Human-readable check name.
        url: Absolute streaming endpoint URL.
        headers: Authentication and content headers.
        payload: JSON request body with streaming enabled.

    Returns:
        CheckResult reporting whether at least one SSE frame was received.
    """

    started = time.perf_counter()
    try:
        with client.stream("POST", url, headers=headers, json=payload) as response:
            if not response.is_success:
                latency_ms = (time.perf_counter() - started) * 1000
                return CheckResult(name, False, response.status_code, latency_ms, "HTTP error")
            saw_frame = any(
                line.startswith("data:") or line.startswith("event:")
                for line in response.iter_lines()
                if line
            )
            latency_ms = (time.perf_counter() - started) * 1000
            return CheckResult(
                name,
                saw_frame,
                response.status_code,
                latency_ms,
                "SSE frames" if saw_frame else "no SSE frames",
            )
    except httpx.HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return CheckResult(name, False, 0, latency_ms, type(exc).__name__)


def _check_embedding(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict,
    expected_dimensions: int | None,
) -> CheckResult:
    """Call the embedding endpoint and validate vector count and dimensions.

    Args:
        client: Configured synchronous HTTP client.
        url: Absolute embedding endpoint URL.
        headers: Authentication and content headers.
        payload: OpenAI-compatible embedding request body.
        expected_dimensions: Optional expected vector length.

    Returns:
        CheckResult with the returned vector dimension in its detail.
    """

    started = time.perf_counter()
    try:
        response = client.post(url, headers=headers, json=payload)
        latency_ms = (time.perf_counter() - started) * 1000
        if not response.is_success:
            return CheckResult("embeddings", False, response.status_code, latency_ms, "HTTP error")
        data = response.json().get("data", [])
        vector = data[0].get("embedding", []) if data and isinstance(data[0], dict) else []
        dimensions = len(vector)
        passed = dimensions > 0 and (expected_dimensions is None or dimensions == expected_dimensions)
        return CheckResult(
            "embeddings",
            passed,
            response.status_code,
            latency_ms,
            f"dimensions={dimensions}",
        )
    except (httpx.HTTPError, ValueError) as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return CheckResult("embeddings", False, 0, latency_ms, type(exc).__name__)


def _gateway_root(openai_base: str) -> str:
    """Remove the terminal ``/v1`` segment from an OpenAI base URL.

    Args:
        openai_base: OpenAI-compatible base URL.

    Returns:
        Gateway root used by Anthropic, Gemini, and Ollama routes.
    """

    normalized = openai_base.rstrip("/")
    return normalized[:-3] if normalized.endswith("/v1") else normalized


if __name__ == "__main__":
    raise SystemExit(main())
