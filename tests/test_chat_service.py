"""Tests for direct chat and optional RAG fallback behavior."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.services import chat_service
from src.services.types import ChatMessage


def _settings(mode: chat_service.ChatMode = "auto") -> chat_service.ChatSettings:
    """Create deterministic chat settings for service tests.

    Args:
        mode: Chat strategy exercised by the test.

    Returns:
        Settings that never depend on the developer's environment.
    """

    return chat_service.ChatSettings(
        mode=mode,
        model="test-chat-model",
        max_history=20,
        max_tokens=128,
        temperature=0.0,
        rag_top_k=3,
    )


class ChatSettingsTests(unittest.TestCase):
    """Verify chat environment parsing and validation."""

    def test_defaults_keep_rag_optional(self):
        """Use automatic fallback behavior when chat variables are absent."""

        settings = chat_service.load_chat_settings({})

        self.assertEqual(settings.mode, "auto")
        self.assertEqual(settings.model, "gpt-4o-mini")
        self.assertEqual(settings.max_history, 20)
        self.assertIsNone(settings.max_tokens)

    def test_invalid_mode_is_rejected(self):
        """Reject misspelled modes before a gateway request is attempted."""

        with self.assertRaisesRegex(ValueError, "CHAT_MODE"):
            chat_service.load_chat_settings({"CHAT_MODE": "sometimes"})


class ChatServiceTests(unittest.IsolatedAsyncioTestCase):
    """Verify direct responses, RAG use, and graceful degradation."""

    def setUp(self):
        """Clear the shared in-memory session before every test."""

        self.session_id = "chat-service-test"
        self.store = chat_service.get_session_store()
        self.store.clear(self.session_id)

    async def asyncTearDown(self):
        """Remove test history from the shared session store."""

        self.store.clear(self.session_id)

    async def _collect_events(self, question: str):
        """Collect all streamed events for one service request.

        Args:
            question: Current user question passed to the chat service.

        Returns:
            Ordered list of events emitted by the async generator.
        """

        return [event async for event in chat_service.ask(self.session_id, question)]

    async def test_direct_chat_sends_bounded_conversation_history(self):
        """Send stored user and assistant turns to the gateway in their original order."""

        self.store.append(
            self.session_id,
            ChatMessage(role="user", content="Previous question"),
        )
        self.store.append(
            self.session_id,
            ChatMessage(role="assistant", content="Previous answer"),
        )
        self.store.append(
            self.session_id,
            ChatMessage(role="user", content="Current question"),
        )
        llm = Mock()
        llm.generate_messages.return_value = "Direct answer"

        with (
            patch.object(chat_service, "load_chat_settings", return_value=_settings()),
            patch.object(chat_service, "_rag_is_candidate", return_value=False),
            patch.object(chat_service, "_create_llm", return_value=llm),
        ):
            events = await self._collect_events("Current question")

        self.assertEqual([event.type for event in events], ["thought", "final_answer"])
        self.assertEqual(events[-1].content, "Direct answer")
        self.assertEqual(events[-1].citations, [])
        sent_messages = llm.generate_messages.call_args.args[0]
        self.assertEqual(
            sent_messages[1:],
            [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": "Current question"},
            ],
        )

    async def test_unavailable_rag_falls_back_to_direct_chat(self):
        """Continue through the gateway when local retrieval cannot initialize."""

        llm = Mock()
        llm.generate_messages.return_value = "Fallback answer"

        with (
            patch.object(chat_service, "load_chat_settings", return_value=_settings("rag")),
            patch.object(chat_service, "_rag_is_candidate", return_value=True),
            patch.object(
                chat_service,
                "_retrieve_rag_results",
                side_effect=chat_service.RAGUnavailableError("missing index"),
            ),
            patch.object(chat_service, "_create_llm", return_value=llm),
            self.assertLogs(chat_service.logger, level="WARNING"),
        ):
            events = await self._collect_events("Question without an index")

        self.assertEqual(events[-1].type, "final_answer")
        self.assertEqual(events[-1].content, "Fallback answer")
        llm.generate_messages.assert_called_once()
        llm.generate_rag_answer.assert_not_called()

    async def test_available_rag_returns_context_citations(self):
        """Return retrieved article metadata when the local RAG path succeeds."""

        llm = Mock()
        llm.generate_rag_answer.return_value = "RAG answer"
        results = [
            {
                "text": "Relevant Reuters article text.",
                "metadata": {
                    "publisher": "Reuters",
                    "title": "Test article",
                    "publish_date": "2026-07-13",
                    "url": "https://example.com/article",
                },
            }
        ]

        with (
            patch.object(chat_service, "load_chat_settings", return_value=_settings("rag")),
            patch.object(chat_service, "_rag_is_candidate", return_value=True),
            patch.object(chat_service, "_retrieve_rag_results", return_value=results),
            patch.object(chat_service, "_create_llm", return_value=llm),
        ):
            events = await self._collect_events("Question with an index")

        final = events[-1]
        self.assertEqual(final.content, "RAG answer")
        self.assertEqual(final.citations[0].source, "Reuters")
        self.assertEqual(final.citations[0].title, "Test article")
        llm.generate_messages.assert_not_called()

    async def test_gateway_error_is_sanitized_and_keeps_stream_valid(self):
        """Emit a final answer event without exposing provider exception details."""

        llm = Mock()
        llm.generate_messages.side_effect = RuntimeError("secret provider response")

        with (
            patch.object(chat_service, "load_chat_settings", return_value=_settings()),
            patch.object(chat_service, "_rag_is_candidate", return_value=False),
            patch.object(chat_service, "_create_llm", return_value=llm),
            self.assertLogs(chat_service.logger, level="ERROR"),
        ):
            events = await self._collect_events("Trigger an error")

        final = events[-1]
        self.assertEqual(final.type, "final_answer")
        self.assertIn("OPENAI_API_KEY", final.content)
        self.assertNotIn("secret provider response", final.content)


if __name__ == "__main__":
    unittest.main()
