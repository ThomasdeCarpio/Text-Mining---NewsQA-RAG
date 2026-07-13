"""API tests proving that chat remains usable without local RAG."""

from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api.main import app
from src.services import chat_service, retrieval_service


class OptionalRAGApiTests(unittest.TestCase):
    """Verify API startup and direct SSE chat when retrieval is unavailable."""

    def setUp(self):
        """Create an isolated API client and clear its deterministic test session."""

        self.client = TestClient(app)
        self.session_id = "optional-rag-api-test"
        chat_service.get_session_store().clear(self.session_id)

    def tearDown(self):
        """Remove test messages from the shared in-memory session store."""

        chat_service.get_session_store().clear(self.session_id)

    def test_health_does_not_require_chromadb(self):
        """Start the FastAPI application even when optional RAG is not installed."""

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_chat_cors_accepts_both_local_vite_addresses(self):
        """Allow the UI to call chat when Vite is opened by hostname or IP address."""

        for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
            with self.subTest(origin=origin):
                response = self.client.options(
                    "/chat/ask",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    },
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.headers["access-control-allow-origin"], origin
                )

    def test_direct_chat_streams_and_persists_answer_without_rag(self):
        """Complete the chat route through the model gateway with RAG disabled."""

        settings = chat_service.ChatSettings(
            mode="auto",
            model="test-chat-model",
            max_history=20,
            max_tokens=128,
            temperature=0.0,
            rag_top_k=3,
        )
        llm = Mock()
        llm.generate_messages.return_value = "API direct answer"

        with (
            patch.object(chat_service, "load_chat_settings", return_value=settings),
            patch.object(chat_service, "_rag_is_candidate", return_value=False),
            patch.object(chat_service, "_create_llm", return_value=llm),
        ):
            response = self.client.post(
                "/chat/ask",
                json={"session_id": self.session_id, "question": "Hello"},
            )

        self.assertEqual(response.status_code, 200)
        frames = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(frames[-1]["type"], "final_answer")
        self.assertEqual(frames[-1]["content"], "API direct answer")

        history = self.client.get(f"/chat/history/{self.session_id}").json()
        self.assertEqual([message["role"] for message in history], ["user", "assistant"])
        self.assertEqual(history[-1]["content"], "API direct answer")

    def test_retrieval_search_reports_optional_dependency_failure(self):
        """Return a controlled 503 only for retrieval-specific requests."""

        with patch.object(
            retrieval_service,
            "_get_store",
            side_effect=retrieval_service.RetrievalUnavailableError("RAG unavailable"),
        ):
            response = self.client.post(
                "/retrieval/search",
                json={"query": "test", "algorithm": "dense", "top_k": 1},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "RAG unavailable")


if __name__ == "__main__":
    unittest.main()
