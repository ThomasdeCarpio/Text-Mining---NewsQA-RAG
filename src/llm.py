from collections.abc import Sequence
from typing import Any, TypedDict

from src.model_gateway import create_generation_client


class ChatMessageInput(TypedDict):
    """Message accepted by the OpenAI-compatible chat completions endpoint."""

    role: str
    content: str


class OpenAILLM:
    """
    Thin wrapper around the OpenAI chat completions API.

    Configuration via environment variables:
      OPENAI_API_KEY  — required
      OPENAI_BASE_URL — optional; set to point at Ollama, Azure OpenAI, or any
                        OpenAI-compatible endpoint (e.g. http://localhost:11434/v1)
      DEEPSEEK_API_KEY — optional; when set, generation uses DeepSeek directly
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant answering questions based on provided context. "
        "Answer concisely and only based on the given context. "
        "Cite supporting context using its bracketed number, for example [1]. "
        "Every factual claim must have a citation, and you must not cite context that "
        "does not support the claim. "
        "If the answer is not in the context, say: "
        "'I cannot find this information in the provided context.'"
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int | None = 1024,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._effective_model = model

    def _get_client(self):
        if self._client is None:
            self._client, self._effective_model = create_generation_client(self.model)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate one response from a system prompt and a user prompt.

        Args:
            system_prompt: Instructions that define the assistant's behavior.
            user_prompt: Current user request or a formatted RAG prompt.

        Returns:
            Text returned by the configured chat model.
        """

        return self.generate_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    def generate_messages(self, messages: Sequence[ChatMessageInput]) -> str:
        """Generate one response from an ordered multi-turn conversation.

        Args:
            messages: Ordered chat messages containing ``role`` and ``content``.

        Returns:
            Text returned by the configured chat model.
        """

        client = self._get_client()
        request: dict[str, Any] = {
            "model": self._effective_model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens

        response = client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    def generate_rag_answer(self, question: str, contexts: list[str]) -> str:
        """Format contexts into a numbered block, then call the default RAG prompt."""
        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
        return self.generate(self.DEFAULT_SYSTEM_PROMPT, user_prompt)


def get_llm(config: dict) -> OpenAILLM:
    """
    Factory. Reads config["llm"].
    Provider credentials and base URLs are always read from environment variables.
    """
    llm_cfg = config.get("llm", {})
    return OpenAILLM(
        model=llm_cfg.get("model", "gpt-4o-mini"),
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 1024),
    )
