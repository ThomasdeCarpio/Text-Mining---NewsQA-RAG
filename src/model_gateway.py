"""Shared client configuration for OpenAI-compatible model gateways."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OpenAIClientSettings:
    """Runtime settings used to create an OpenAI-compatible client.

    Args:
        api_key: Provider API key, or None when the SDK should report a missing key.
        base_url: Optional OpenAI-compatible API base URL ending in ``/v1``.
    """

    api_key: str | None
    base_url: str | None


def load_openai_client_settings(
    environ: Mapping[str, str] | None = None,
) -> OpenAIClientSettings:
    """Read OpenAI-compatible client settings from environment variables.

    Args:
        environ: Optional environment mapping used instead of ``os.environ``.

    Returns:
        Settings populated from ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL``.
    """

    source = environ if environ is not None else os.environ
    api_key = source.get("OPENAI_API_KEY", "").strip() or None
    base_url = source.get("OPENAI_BASE_URL", "").strip() or None
    return OpenAIClientSettings(api_key=api_key, base_url=base_url)


def create_openai_client(environ: Mapping[str, str] | None = None) -> Any:
    """Create an OpenAI SDK client for OpenAI or a compatible gateway.

    Args:
        environ: Optional environment mapping used to resolve client settings.

    Returns:
        Configured ``openai.OpenAI`` client instance.
    """

    import openai

    if environ is None:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)

    settings = load_openai_client_settings(environ)
    return openai.OpenAI(api_key=settings.api_key, base_url=settings.base_url)
