import os
import time
from typing import Any

# Must match scripts/ingest.py's PIPELINE_CONFIG/paths exactly, since that's
# what actually populates this collection. Centralize via configs/setting.py
# once that exists (see README roadmap Milestone 5) — not worth building for
# a single caller today.
_EMBEDDING_CONFIG = {
    "embedding": {
        "provider": "sentence-transformers",
        "model_name": "all-MiniLM-L6-v2",
    }
}
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CHROMA_DB_DIR = os.path.join(_PROJECT_ROOT, "data", "chroma_db")
_COLLECTION_NAME = "newsqa_cnn"

_ALGORITHMS = [
    {"id": "dense", "label": "Dense (ChromaDB vector search)", "available": True},
    {"id": "hybrid", "label": "Hybrid (dense + BM25)", "available": False},
    {"id": "reranked", "label": "Hybrid + Reranker", "available": False},
]

_store: Any | None = None


class RetrievalUnavailableError(RuntimeError):
    """Raised when optional local retrieval dependencies are unavailable."""


def _get_store() -> Any:
    """Create the local ChromaDB store only when a retrieval route needs it.

    Returns:
        Cached ``ChromaStore`` instance.

    Raises:
        RetrievalUnavailableError: If optional retrieval packages are not installed.
    """

    global _store
    if _store is None:
        try:
            from src.indexing.chroma_store import ChromaStore
            from src.indexing.embeddings import get_embedding_function
        except ImportError as exc:
            raise RetrievalUnavailableError(
                "Local retrieval dependencies are not installed."
            ) from exc

        _store = ChromaStore(
            db_path=_CHROMA_DB_DIR,
            embedding_function=get_embedding_function(_EMBEDDING_CONFIG),
        )
    return _store


def list_algorithms() -> list[dict]:
    return _ALGORITHMS


def get_collection_stats() -> dict:
    """Return collection stats without making RAG a startup requirement.

    Returns:
        Collection metadata, or an absent collection response when optional
        retrieval dependencies are not installed.
    """

    try:
        return _get_store().get_collection_stats(_COLLECTION_NAME)
    except RetrievalUnavailableError:
        return {
            "exists": False,
            "name": _COLLECTION_NAME,
            "count": 0,
            "sample": [],
            "metadata": {},
            "embedding_info": {},
        }


def search(query: str, algorithm: str, top_k: int) -> tuple[list[dict], dict]:
    """Search the local collection with the selected retrieval algorithm.

    Args:
        query: Natural-language search query.
        algorithm: Retrieval algorithm identifier.
        top_k: Maximum number of chunks to return.

    Returns:
        Ranked results and a timing breakdown.

    Raises:
        RetrievalUnavailableError: If local retrieval dependencies are absent.
        ValueError: If the algorithm name is unknown.
        NotImplementedError: If the algorithm is known but not implemented.
    """

    if algorithm == "dense":
        store = _get_store()
        from src.retrieval.dense import dense_search

        t0 = time.perf_counter()
        results, timing_ms = dense_search(store, _COLLECTION_NAME, query, top_k)
        timing_ms["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return results, timing_ms

    known = {a["id"] for a in _ALGORITHMS}
    if algorithm not in known:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Known: {sorted(known)}")
    raise NotImplementedError(f"Algorithm '{algorithm}' is not implemented yet.")
