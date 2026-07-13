"""Chat orchestration with optional retrieval and direct-model fallback."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import yaml

from src.llm import ChatMessageInput, OpenAILLM
from src.model_gateway import PROJECT_ROOT
from src.services.session_store import SessionStore, get_session_store
from src.services.types import AgentEvent, ChatMessage, Citation

logger = logging.getLogger(__name__)

ChatMode = Literal["auto", "direct", "rag"]

_DIRECT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's request accurately and concisely. "
    "Do not claim that you searched a news database or cite retrieved sources unless "
    "retrieval context is explicitly provided."
)
_RAG_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
_RAG_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
_VALID_CHAT_MODES = {"auto", "direct", "rag"}
_rag_agent = None
_rag_agent_key: tuple[str, float, int | None, int] | None = None


class RAGUnavailableError(RuntimeError):
    """Raised when the optional local retrieval pipeline cannot serve a query."""


@dataclass(frozen=True)
class ChatSettings:
    """Runtime controls for direct and retrieval-augmented chat.

    Args:
        mode: ``auto`` for optional RAG, ``direct`` to skip RAG, or ``rag`` to
            request RAG while retaining direct fallback.
        model: Model identifier accepted by the configured model gateway.
        max_history: Maximum number of stored conversation messages sent to the model.
        max_tokens: Optional maximum number of tokens requested for each answer.
            ``None`` lets the gateway select a model-appropriate limit.
        temperature: Sampling temperature passed to the chat completions endpoint.
        rag_top_k: Maximum number of local chunks included in a RAG response.
    """

    mode: ChatMode
    model: str
    max_history: int
    max_tokens: int | None
    temperature: float
    rag_top_k: int


def _read_positive_int(source: Mapping[str, str], name: str, default: int) -> int:
    """Read a strictly positive integer from an environment mapping.

    Args:
        source: Environment-style key/value mapping.
        name: Variable name to parse.
        default: Value used when the variable is absent or blank.

    Returns:
        Parsed positive integer.

    Raises:
        ValueError: If the configured value is not a positive integer.
    """

    raw_value = source.get(name, "").strip()
    if not raw_value:
        return default

    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _read_optional_positive_int(
    source: Mapping[str, str], name: str
) -> int | None:
    """Read an optional positive integer from an environment mapping.

    Args:
        source: Environment-style key/value mapping.
        name: Variable name to parse.

    Returns:
        Parsed positive integer, or ``None`` when the variable is absent or blank.

    Raises:
        ValueError: If the configured value is not a positive integer.
    """

    raw_value = source.get(name, "").strip()
    if not raw_value:
        return None

    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def load_chat_settings(environ: Mapping[str, str] | None = None) -> ChatSettings:
    """Load chat behavior from environment variables and the project ``.env`` file.

    Args:
        environ: Optional environment mapping used instead of ``os.environ``.

    Returns:
        Validated settings for one chat request.

    Raises:
        ValueError: If a chat mode or numeric setting is invalid.
    """

    if environ is None:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)

    source = environ if environ is not None else os.environ
    mode = source.get("CHAT_MODE", "auto").strip().lower() or "auto"
    if mode not in _VALID_CHAT_MODES:
        supported = ", ".join(sorted(_VALID_CHAT_MODES))
        raise ValueError(f"CHAT_MODE must be one of: {supported}.")

    model = source.get("CHAT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    temperature = float(source.get("CHAT_TEMPERATURE", "0.2").strip() or "0.2")
    return ChatSettings(
        mode=cast(ChatMode, mode),
        model=model,
        max_history=_read_positive_int(source, "CHAT_MAX_HISTORY", 20),
        max_tokens=_read_optional_positive_int(source, "CHAT_MAX_TOKENS"),
        temperature=temperature,
        rag_top_k=_read_positive_int(source, "RAG_TOP_K", 5),
    )


def _create_llm(settings: ChatSettings) -> OpenAILLM:
    """Create an LLM wrapper from validated chat settings.

    Args:
        settings: Runtime chat settings.

    Returns:
        Lazily connected OpenAI-compatible LLM wrapper.
    """

    return OpenAILLM(
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def _rag_is_candidate(mode: ChatMode) -> bool:
    """Determine whether a request should attempt the optional local RAG path.

    Args:
        mode: Configured chat mode.

    Returns:
        ``True`` when RAG was explicitly requested or an installed local index
        makes RAG a viable automatic choice.
    """

    if mode == "direct":
        return False
    if mode == "rag":
        return True

    try:
        has_chromadb = importlib.util.find_spec("chromadb") is not None
    except (ImportError, ValueError):
        has_chromadb = False
    return has_chromadb and _RAG_DB_PATH.is_dir()


def _get_rag_agent(settings: ChatSettings):
    """Build and cache the real RAG pipeline without adding startup dependencies.

    Args:
        settings: Runtime chat settings used for retrieval and generation.

    Returns:
        Configured ``RAGAgent`` backed by the production news collection.

    Raises:
        RAGUnavailableError: If optional RAG dependencies or configuration fail.
    """

    global _rag_agent, _rag_agent_key
    cache_key = (
        settings.model,
        settings.temperature,
        settings.max_tokens,
        settings.rag_top_k,
    )
    if _rag_agent is not None and _rag_agent_key == cache_key:
        return _rag_agent

    try:
        from src.agents.rag_agent import RAGAgent
        from src.indexing.chroma_store import ChromaStore
        from src.indexing.embeddings import get_embedding_function
        from src.retrieval.reranker import get_reranker
        from src.retrieval.retriever_factory import get_retriever

        with _RAG_CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}

        store = ChromaStore(
            str(_RAG_DB_PATH),
            get_embedding_function(config),
        )
        retriever = get_retriever(
            "dense",
            config,
            store,
            "newsqa_cnn",
        )
        rerank_top_n = min(
            settings.rag_top_k,
            int(
                config.get("retrieval", {})
                .get("reranker", {})
                .get("top_n", settings.rag_top_k)
            ),
        )
        _rag_agent = RAGAgent(
            retriever=retriever,
            reranker=get_reranker(config),
            llm=_create_llm(settings),
            top_k=settings.rag_top_k,
            rerank_top_n=rerank_top_n,
        )
        _rag_agent_key = cache_key
    except Exception as exc:
        raise RAGUnavailableError("The local RAG pipeline could not initialize.") from exc

    return _rag_agent


def _run_rag_pipeline(question: str, settings: ChatSettings) -> dict:
    """Run retrieval, reranking, and generation for one question.

    Args:
        question: User question passed to the RAG pipeline.
        settings: Runtime chat settings used to build the pipeline.

    Returns:
        Full ``RAGAgent.run`` result containing the answer and ranked chunks.

    Raises:
        RAGUnavailableError: If the collection is absent or any RAG stage fails.
    """

    try:
        from src.services import retrieval_service

        stats = retrieval_service.get_collection_stats()
        if not stats.get("exists") or int(stats.get("count", 0)) <= 0:
            raise RAGUnavailableError("The local news collection is empty or missing.")

        result = _get_rag_agent(settings).run(question)
    except RAGUnavailableError:
        raise
    except Exception as exc:
        raise RAGUnavailableError("The local RAG pipeline failed.") from exc

    if not result.get("reranked_chunks") or not str(result.get("answer") or "").strip():
        raise RAGUnavailableError("The local RAG pipeline returned no answer context.")
    return result


def _build_citations(results: Sequence[dict]) -> list[Citation]:
    """Convert retrieved chunks into the citation contract used by the UI.

    Args:
        results: Retrieved chunk dictionaries containing text and metadata.

    Returns:
        Citations in the same order as the retrieved chunks.
    """

    citations: list[Citation] = []
    for result in results:
        metadata = result.get("metadata") or {}
        citations.append(
            Citation(
                source=str(
                    metadata.get("publisher")
                    or metadata.get("source")
                    or "Unknown source"
                ),
                title=str(metadata.get("title") or "Untitled article"),
                date=str(
                    metadata.get("publish_date")
                    or metadata.get("published_date")
                    or metadata.get("date")
                    or "Unknown date"
                ),
                url=str(metadata.get("url") or ""),
                chunk_text=str(result.get("text") or ""),
            )
        )
    return citations


def _build_direct_messages(
    history: Sequence[ChatMessage],
    question: str,
    max_history: int,
) -> list[ChatMessageInput]:
    """Build a bounded multi-turn conversation for direct model chat.

    Args:
        history: Stored user and assistant messages for the current session.
        question: Current question, added only when the router has not stored it yet.
        max_history: Maximum number of conversation messages to include.

    Returns:
        Ordered system and conversation messages accepted by ``OpenAILLM``.
    """

    conversation = list(history)
    if (
        not conversation
        or conversation[-1].role != "user"
        or conversation[-1].content != question
    ):
        conversation.append(ChatMessage(role="user", content=question))

    messages: list[ChatMessageInput] = [
        {"role": "system", "content": _DIRECT_SYSTEM_PROMPT}
    ]
    messages.extend(
        {"role": message.role, "content": message.content}
        for message in conversation[-max_history:]
    )
    return messages


def _record(store: SessionStore, event: AgentEvent) -> AgentEvent:
    """Record an event in the admin trace log before streaming it.

    Args:
        store: Session store that owns the shared trace log.
        event: Event to record and return.

    Returns:
        The unchanged event, allowing concise ``yield`` statements.
    """

    store.record_trace(event)
    return event


def _gateway_failure_event() -> AgentEvent:
    """Create a credential-safe final event for model gateway failures."""

    return AgentEvent(
        type="final_answer",
        content=(
            "The model gateway request failed. Check OPENAI_API_KEY, "
            "OPENAI_BASE_URL, and CHAT_MODEL, then try again."
        ),
        citations=[],
    )


async def ask(session_id: str, question: str) -> AsyncIterator[AgentEvent]:
    """Answer a chat request with optional RAG and reliable direct fallback.

    Args:
        session_id: Identifier used to load the conversation history.
        question: Current user question.

    Yields:
        Trace events followed by exactly one final answer event.

    Notes:
        RAG dependencies are imported only when RAG is a viable candidate. Any
        retrieval initialization failure falls back to normal model chat.
    """

    store = get_session_store()
    try:
        settings = load_chat_settings()
    except (TypeError, ValueError):
        logger.exception("Invalid chat configuration")
        yield _record(
            store,
            AgentEvent(
                type="final_answer",
                content="Chat configuration is invalid. Check the CHAT_* variables in .env.",
                citations=[],
            ),
        )
        return

    if _rag_is_candidate(settings.mode):
        yield _record(
            store,
            AgentEvent(
                type="tool_call",
                tool_name="rag_pipeline",
                content="Retrieving and reranking the local news collection.",
            ),
        )
        try:
            rag_result = await asyncio.to_thread(_run_rag_pipeline, question, settings)
        except RAGUnavailableError:
            logger.warning(
                "RAG is unavailable; falling back to direct chat",
                exc_info=True,
            )
            yield _record(
                store,
                AgentEvent(
                    type="thought",
                    content="Local retrieval is unavailable. Continuing without RAG.",
                ),
            )
        else:
            results = rag_result["reranked_chunks"]
            yield _record(
                store,
                AgentEvent(
                    type="tool_result",
                    tool_name="rag_pipeline",
                    content=f"Found {len(results)} relevant news chunks.",
                ),
            )
            yield _record(
                store,
                AgentEvent(
                    type="final_answer",
                    content=str(rag_result["answer"]),
                    citations=_build_citations(results),
                ),
            )
            return

    llm = _create_llm(settings)
    yield _record(
        store,
        AgentEvent(
            type="thought",
            content="Using direct model chat without local retrieval.",
        ),
    )
    messages = _build_direct_messages(
        store.get_history(session_id), question, settings.max_history
    )
    try:
        answer = await asyncio.to_thread(llm.generate_messages, messages)
        if not answer.strip():
            raise RuntimeError("The model gateway returned an empty answer.")
    except Exception:
        logger.exception("Direct model gateway request failed")
        yield _record(store, _gateway_failure_event())
        return

    yield _record(
        store,
        AgentEvent(type="final_answer", content=answer, citations=[]),
    )
