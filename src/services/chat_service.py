import asyncio
import os
from collections.abc import AsyncIterator

import yaml

from src.services.session_store import get_session_store
from src.services.types import AgentEvent, Citation

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CHROMA_DB_DIR = os.path.join(_PROJECT_ROOT, "data", "chroma_db")
_COLLECTION_NAME = "newsqa_cnn"  # the app's production corpus (from scripts/ingest.py)

_agent = None


def _get_agent():
    """Build the RAG pipeline once (loads the embedding model + collection)."""
    global _agent
    if _agent is None:
        from src.indexing.chroma_store import ChromaStore
        from src.indexing.embeddings import get_embedding_function
        from src.retrieval.retriever_factory import get_retriever
        from src.retrieval.reranker import get_reranker
        from src.llm import get_llm
        from src.agents.rag_agent import RAGAgent

        config = yaml.safe_load(open(os.path.join(_PROJECT_ROOT, "configs", "config.yaml"), encoding="utf-8"))
        store = ChromaStore(_CHROMA_DB_DIR, get_embedding_function(config))
        retriever = get_retriever("dense", config, store, _COLLECTION_NAME)
        retrieval_cfg = config.get("retrieval", {})
        _agent = RAGAgent(
            retriever=retriever,
            reranker=get_reranker(config),
            llm=get_llm(config),
            top_k=retrieval_cfg.get("top_k", 10),
            rerank_top_n=retrieval_cfg.get("reranker", {}).get("top_n", 5),
        )
    return _agent


def _to_citations(chunks: list[dict]) -> list[Citation]:
    out = []
    for c in chunks:
        m = c.get("metadata", {}) or {}
        out.append(Citation(
            source=m.get("publisher", "") or "Unknown",
            title=m.get("title", "") or "(untitled)",
            date=m.get("publish_date", "") or "",
            url=m.get("url", "") or "",
            chunk_text=(c.get("text", "") or "")[:300],
        ))
    return out


async def ask(_session_id: str, question: str) -> AsyncIterator[AgentEvent]:
    """
    Run the real RAG pipeline (retrieve → rerank → generate) and stream events:
    thought → tool_call → tool_result → final_answer (with citations).
    """
    store = get_session_store()

    def emit(event: AgentEvent) -> AgentEvent:
        store.record_trace(event)
        return event

    yield emit(AgentEvent(type="thought", content=f"Analyzing question: '{question}'"))
    yield emit(AgentEvent(type="tool_call", tool_name="retrieval",
                          content=f"Searching corpus for: {question}"))

    try:
        result = await asyncio.to_thread(_get_agent().run, question)
    except Exception as e:
        yield emit(AgentEvent(type="final_answer",
                              content=f"Sorry, I couldn't answer that (pipeline error: {e}).",
                              citations=[]))
        return

    reranked = result["reranked_chunks"]
    yield emit(AgentEvent(type="tool_result", tool_name="retrieval",
                          content=f"Found {len(reranked)} relevant chunks."))
    yield emit(AgentEvent(type="final_answer",
                          content=result["answer"],
                          citations=_to_citations(reranked)))
