import os
import time

from src.indexing.chroma_store import ChromaStore
from src.indexing.embeddings import get_embedding_function
from src.retrieval.dense import dense_search

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

_store: ChromaStore | None = None


def _get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore(db_path=_CHROMA_DB_DIR, embedding_function=get_embedding_function(_EMBEDDING_CONFIG))
    return _store


def list_algorithms() -> list[dict]:
    return _ALGORITHMS


def get_collection_stats() -> dict:
    return _get_store().get_collection_stats(_COLLECTION_NAME)


def search(query: str, algorithm: str, top_k: int) -> tuple[list[dict], dict]:
    if algorithm == "dense":
        t0 = time.perf_counter()
        results, timing_ms = dense_search(_get_store(), _COLLECTION_NAME, query, top_k)
        timing_ms["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return results, timing_ms

    known = {a["id"] for a in _ALGORITHMS}
    if algorithm not in known:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Known: {sorted(known)}")
    raise NotImplementedError(f"Algorithm '{algorithm}' is not implemented yet.")
