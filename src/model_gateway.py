"""Shared client configuration for OpenAI-compatible model gateways."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class OpenAIClientSettings:
    """Runtime settings used to create an OpenAI-compatible client.

    Args:
        api_key: Provider API key, or None when the SDK should report a missing key.
        base_url: Optional OpenAI-compatible API base URL ending in ``/v1``.
    """

    api_key: str | None
    base_url: str | None


@dataclass(frozen=True)
class GenerationClientSettings:
    """Provider settings for text generation.

    Args:
        api_key: API key sent to the selected OpenAI-compatible provider.
        base_url: API base URL for the selected provider.
        model: Effective model name sent to the provider.
    """

    api_key: str | None
    base_url: str | None
    model: str


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


def load_generation_client_settings(
    model: str,
    environ: Mapping[str, str] | None = None,
) -> GenerationClientSettings:
    """Resolve generation settings for DeepSeek or the shared model gateway.

    Args:
        model: Model requested by the caller.
        environ: Optional environment mapping used instead of ``os.environ``.

    Returns:
        DeepSeek settings when ``DEEPSEEK_API_KEY`` is set; otherwise settings
        derived from ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL``.
    """

    if environ is None:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)

    source = environ if environ is not None else os.environ
    deepseek_api_key = source.get("DEEPSEEK_API_KEY", "").strip()
    if deepseek_api_key:
        effective_model = model if model.startswith("deepseek") else "deepseek-chat"
        return GenerationClientSettings(
            api_key=deepseek_api_key,
            base_url=DEEPSEEK_BASE_URL,
            model=effective_model,
        )

    openai_settings = load_openai_client_settings(source)
    return GenerationClientSettings(
        api_key=openai_settings.api_key,
        base_url=openai_settings.base_url,
        model=model,
    )


def create_generation_client(
    model: str,
    environ: Mapping[str, str] | None = None,
) -> tuple[Any, str]:
    """Create the generation client and return its effective model name.

    Args:
        model: Model requested by the caller.
        environ: Optional environment mapping used instead of ``os.environ``.

    Returns:
        Tuple containing the configured OpenAI SDK client and effective model.
    """

    import openai

    settings = load_generation_client_settings(model, environ)
    client = openai.OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    return client, settings.model
