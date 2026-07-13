"""Tests for model gateway settings and OpenAI-compatible clients."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.check_model_gateway import _gateway_root

try:
    import chromadb  # noqa: F401
except ModuleNotFoundError:
    sys.modules["chromadb"] = SimpleNamespace(
        Documents=list,
        EmbeddingFunction=object,
        Embeddings=list,
    )

from src.indexing.embeddings import OpenAIEmbeddingFunction
from src.llm import OpenAILLM
from src.model_gateway import create_openai_client, load_openai_client_settings


class ModelGatewaySettingsTests(unittest.TestCase):
    """Verify environment parsing and OpenAI SDK client construction."""

    def test_load_settings_normalizes_empty_values(self):
        """Convert blank environment values to None for SDK compatibility."""

        settings = load_openai_client_settings(
            {"OPENAI_API_KEY": "  ", "OPENAI_BASE_URL": ""}
        )
        self.assertIsNone(settings.api_key)
        self.assertIsNone(settings.base_url)

    @patch("openai.OpenAI")
    def test_create_client_passes_gateway_key_and_base_url(self, openai_constructor):
        """Pass XAH-compatible environment values directly to the OpenAI SDK."""

        environment = {
            "OPENAI_API_KEY": "test-secret",
            "OPENAI_BASE_URL": "https://api.xah.io/v1",
        }
        expected_client = Mock()
        openai_constructor.return_value = expected_client

        actual_client = create_openai_client(environment)

        self.assertIs(actual_client, expected_client)
        openai_constructor.assert_called_once_with(
            api_key="test-secret",
            base_url="https://api.xah.io/v1",
        )

    def test_gateway_root_removes_only_terminal_v1(self):
        """Build Anthropic, Gemini, and Ollama URLs without damaging other paths."""

        self.assertEqual(_gateway_root("https://api.xah.io/v1"), "https://api.xah.io")
        self.assertEqual(_gateway_root("https://example.com/proxy"), "https://example.com/proxy")


class ModelClientIntegrationTests(unittest.TestCase):
    """Verify LLM and embedding wrappers use the shared gateway factory."""

    @patch("src.llm.create_openai_client")
    def test_llm_uses_shared_client_factory(self, client_factory):
        """Create the chat client lazily and reuse it across calls."""

        expected_client = Mock()
        client_factory.return_value = expected_client
        llm = OpenAILLM()

        self.assertIs(llm._get_client(), expected_client)
        self.assertIs(llm._get_client(), expected_client)
        client_factory.assert_called_once_with()

    @patch("src.llm.create_openai_client")
    def test_llm_generation_preserves_model_configuration(self, client_factory):
        """Send configured model controls through the shared gateway client."""

        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )
        client_factory.return_value = client
        llm = OpenAILLM(model="remote-chat", temperature=0.2, max_tokens=321)

        answer = llm.generate("system", "question")

        self.assertEqual(answer, "answer")
        client.chat.completions.create.assert_called_once_with(
            model="remote-chat",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
            temperature=0.2,
            max_tokens=321,
        )

    @patch("src.llm.create_openai_client")
    def test_llm_can_defer_output_limit_to_reasoning_model(self, client_factory):
        """Omit max_tokens when the gateway should choose a model-appropriate limit."""

        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )
        client_factory.return_value = client
        llm = OpenAILLM(model="reasoning-model", max_tokens=None)

        answer = llm.generate_messages([{"role": "user", "content": "question"}])

        self.assertEqual(answer, "answer")
        client.chat.completions.create.assert_called_once_with(
            model="reasoning-model",
            messages=[{"role": "user", "content": "question"}],
            temperature=0.0,
        )

    @patch("src.indexing.embeddings.create_openai_client")
    def test_embedding_batches_preserve_order_and_request_dimensions(self, client_factory):
        """Batch large embedding inputs without reordering returned vectors."""

        client = Mock()

        def create_embeddings(**kwargs):
            """Return deterministic vectors derived from each test input."""

            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[float(value)]) for value in kwargs["input"]]
            )

        client.embeddings.create.side_effect = create_embeddings
        client_factory.return_value = client
        embedding = OpenAIEmbeddingFunction(model_name="remote-embedding", dimensions=1)
        inputs = [str(index) for index in range(2049)]

        vectors = embedding(inputs)

        self.assertEqual(vectors[0], [0.0])
        self.assertEqual(vectors[-1], [2048.0])
        self.assertEqual(client.embeddings.create.call_count, 2)
        first_call = client.embeddings.create.call_args_list[0].kwargs
        second_call = client.embeddings.create.call_args_list[1].kwargs
        self.assertEqual(len(first_call["input"]), 2048)
        self.assertEqual(len(second_call["input"]), 1)
        self.assertEqual(first_call["model"], "remote-embedding")
        self.assertEqual(first_call["dimensions"], 1)

    def test_embedding_config_does_not_serialize_gateway_credentials(self):
        """Keep API keys and base URLs out of Chroma embedding metadata."""

        embedding = OpenAIEmbeddingFunction(model_name="remote-embedding", dimensions=1024)
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "must-not-be-serialized",
                "OPENAI_BASE_URL": "https://api.xah.io/v1",
            },
            clear=False,
        ):
            config = embedding.get_config()

        self.assertEqual(config, {"model_name": "remote-embedding", "dimensions": 1024})
        self.assertNotIn("api_key", config)
        self.assertNotIn("base_url", config)


if __name__ == "__main__":
    unittest.main()
