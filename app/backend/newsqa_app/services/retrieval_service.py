import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from newsqa_rag.model_gateway import PROJECT_ROOT

_DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
_DEFAULT_COLLECTION = "newsqa_cnn"
_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

_store: Any | None = None
_locked: Any | None = None


class RetrievalUnavailableError(RuntimeError):
    """Raised when optional local retrieval dependencies are unavailable."""


def load_config() -> dict:
    """Return the pipeline configuration the locked retrieval path is built from."""

    with _CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def get_database_path() -> Path:
    """Return the configured ChromaDB directory, relative to the project root."""

    configured = os.getenv("RAG_DB_PATH", "").strip()
    path = Path(configured) if configured else _DEFAULT_DB_PATH
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_collection_name() -> str:
    """Return the collection shared by every API retrieval flow."""

    return os.getenv("RAG_COLLECTION", "").strip() or _DEFAULT_COLLECTION


def get_locked_index_dir() -> Path:
    """Return the directory holding the locked chunk and sparse-index artifacts."""

    configured = os.getenv("RAG_LOCKED_INDEX_DIR", "").strip()
    if not configured:
        configured = load_config().get("artifacts", {}).get("locked_index_dir", "data/locked_index")
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def locked_artifacts() -> dict[str, Path]:
    """Return the two files the locked retriever loads, present or not."""

    root = get_locked_index_dir()
    return {"chunks": root / "chunks.jsonl", "sparse_index": root / "bge_m3_sparse.pkl"}


def locked_is_available() -> bool:
    """Return whether the exported locked index is on disk."""

    return all(path.exists() for path in locked_artifacts().values())


def is_available() -> bool:
    """Return whether any local retrieval runtime can serve a query."""

    if locked_is_available():
        return True
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


def _get_locked() -> tuple[Any, Any, dict]:
    """Build the tournament-locked retriever and reranker once, on first query.

    Both are expensive to construct - BGE-M3 and bge-reranker-large each load
    model weights - so the result is cached for the process lifetime.

    Returns:
        The retriever, its reranker, and the config they were built from.

    Raises:
        RetrievalUnavailableError: If the exported artifacts or their optional
            dependencies are missing.
    """

    global _locked
    if _locked is not None:
        return _locked

    artifacts = locked_artifacts()
    missing = [name for name, path in artifacts.items() if not path.exists()]
    if missing:
        raise RetrievalUnavailableError(
            f"Locked index artifacts not found in {get_locked_index_dir()}: {missing}. "
            "Run notebooks/14_export_locked_index_kaggle.ipynb and unpack its bundle there."
        )

    try:
        from newsqa_rag.retrieval.reranker import get_reranker
        from newsqa_rag.retrieval.retriever_factory import get_retriever
    except ImportError as exc:
        raise RetrievalUnavailableError(
            "Local retrieval dependencies are not installed."
        ) from exc

    config = load_config()
    with artifacts["chunks"].open(encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]

    retriever = get_retriever(
        config.get("retrieval", {}).get("retriever", "sparse"),
        config,
        store=None,
        collection_name=None,
        chunks=chunks,
        bm25_path=str(artifacts["sparse_index"]),
    )
    _locked = (retriever, get_reranker(config), config)
    return _locked


def get_locked_retriever() -> Any:
    """Return the retriever the Phase 1 tournament locked in."""

    return _get_locked()[0]


def list_algorithms() -> list[dict]:
    """Return the retrieval algorithms, flagged by what is actually on disk."""

    reranker_model = load_config().get("retrieval", {}).get("reranker", {}).get("model", "reranker")
    has_locked = locked_is_available()
    try:
        has_chroma = importlib.util.find_spec("chromadb") is not None and get_database_path().is_dir()
    except (ImportError, ValueError):
        has_chroma = False
    return [
        {
            "id": "locked",
            "label": f"Locked (BGE-M3 sparse + {str(reranker_model).split('/')[-1]})",
            "available": has_locked,
        },
        {"id": "sparse", "label": "BGE-M3 sparse, no reranker", "available": has_locked},
        {"id": "dense", "label": "Dense (ChromaDB vector search)", "available": has_chroma},
        {"id": "hybrid", "label": "Hybrid (dense + sparse)", "available": False},
    ]


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

    if algorithm in {"locked", "sparse"}:
        retriever, reranker, config = _get_locked()
        # A reranker can only reorder what it is handed, so retrieve the
        # configured candidate pool (20) and let it cut down to top_k. This is
        # the same two-stage shape the tournament measured.
        candidates = max(top_k, int(config.get("retrieval", {}).get("top_k", 20)))

        t0 = time.perf_counter()
        results = retriever.retrieve(query, candidates)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if algorithm == "locked":
            results = reranker.rerank(query, results, top_k)
        else:
            results = results[:top_k]
        rerank_ms = (time.perf_counter() - t0) * 1000

        # Sparse scores are similarities, not distances, but the response
        # schema is shaped around Chroma. Report the complement so that a
        # smaller number still means a closer match.
        for result in results:
            result.setdefault("distance", round(1.0 - float(result.get("score", 0.0)), 6))
        return results, {
            "model_cold_start": False,
            "embed_ms": round(retrieve_ms, 1),
            "db_query_ms": round(rerank_ms, 1),
            "total_ms": round(retrieve_ms + rerank_ms, 1),
        }

    known = {a["id"] for a in list_algorithms()}
    if algorithm not in known:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Known: {sorted(known)}")
    raise NotImplementedError(f"Algorithm '{algorithm}' is not implemented yet.")
