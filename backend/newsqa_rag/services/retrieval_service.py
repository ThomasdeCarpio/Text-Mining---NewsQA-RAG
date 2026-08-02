import importlib.util
import os
import time
from pathlib import Path
from typing import Any

import yaml

from newsqa_rag.model_gateway import PROJECT_ROOT

_DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
_DEFAULT_COLLECTION = "newsqa_cnn"
_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

_ALGORITHMS = [
    {"id": "dense", "label": "Dense (ChromaDB vector search)", "available": True},
    {"id": "hybrid", "label": "Hybrid (dense + BM25)", "available": False},
    {"id": "reranked", "label": "Hybrid + Reranker", "available": False},
]

_store: Any | None = None


class RetrievalUnavailableError(RuntimeError):
    """Raised when optional local retrieval dependencies are unavailable."""


def get_database_path() -> Path:
    """Return the configured ChromaDB directory, relative to the project root."""

    configured = os.getenv("RAG_DB_PATH", "").strip()
    path = Path(configured) if configured else _DEFAULT_DB_PATH
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_collection_name() -> str:
    """Return the collection shared by every API retrieval flow."""

    return os.getenv("RAG_COLLECTION", "").strip() or _DEFAULT_COLLECTION


def is_available() -> bool:
    """Return whether the optional local retrieval runtime can be used."""

    try:
        has_chromadb = importlib.util.find_spec("chromadb") is not None
    except (ImportError, ValueError):
        has_chromadb = False
    return has_chromadb and get_database_path().is_dir()


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
            from newsqa_rag.indexing.chroma_store import ChromaStore
            from newsqa_rag.indexing.embeddings import get_embedding_function
        except ImportError as exc:
            raise RetrievalUnavailableError(
                "Local retrieval dependencies are not installed."
            ) from exc

        with _CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        _store = ChromaStore(
            db_path=str(get_database_path()),
            embedding_function=get_embedding_function(config),
        )
    return _store


def get_dense_retriever() -> Any:
    """Return a dense retriever backed by the shared API data configuration."""

    from newsqa_rag.retrieval.dense import DenseRetriever

    return DenseRetriever(_get_store(), get_collection_name())


def list_algorithms() -> list[dict]:
    return _ALGORITHMS


def get_collection_stats() -> dict:
    """Return collection stats without making RAG a startup requirement.

    Returns:
        Collection metadata, or an absent collection response when optional
        retrieval dependencies are not installed.
    """

    try:
        return _get_store().get_collection_stats(get_collection_name())
    except RetrievalUnavailableError:
        return {
            "exists": False,
            "name": get_collection_name(),
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
        t0 = time.perf_counter()
        results, timing_ms = get_dense_retriever().retrieve_with_timing(query, top_k)
        timing_ms["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return results, timing_ms

    known = {a["id"] for a in _ALGORITHMS}
    if algorithm not in known:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Known: {sorted(known)}")
    raise NotImplementedError(f"Algorithm '{algorithm}' is not implemented yet.")
