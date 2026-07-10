import os


class OpenAILLM:
    """
    Thin wrapper around the OpenAI chat completions API.

    Configuration via environment variables:
      OPENAI_API_KEY  — required
      OPENAI_BASE_URL — optional; set to point at Ollama, Azure OpenAI, or any
                        OpenAI-compatible endpoint (e.g. http://localhost:11434/v1)
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant answering questions based on provided context. "
        "Answer concisely and only based on the given context. "
        "If the answer is not in the context, say: "
        "'I cannot find this information in the provided context.'"
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL") or None,
            )
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_rag_answer(self, question: str, contexts: list[str]) -> str:
        """Format contexts into a numbered block, then call the default RAG prompt."""
        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
        return self.generate(self.DEFAULT_SYSTEM_PROMPT, user_prompt)


def get_llm(config: dict) -> OpenAILLM:
    """
    Factory. Reads config["llm"].
    API key and base URL are always read from env (OPENAI_API_KEY, OPENAI_BASE_URL).
    """
    llm_cfg = config.get("llm", {})
    return OpenAILLM(
        model=llm_cfg.get("model", "gpt-4o-mini"),
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 1024),
    )
