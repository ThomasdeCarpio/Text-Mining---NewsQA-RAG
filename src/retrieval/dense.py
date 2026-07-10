import time

from src.indexing.chroma_store import ChromaStore
from src.retrieval import BaseRetriever


def dense_search(store: ChromaStore, collection_name: str, query: str, top_k: int) -> tuple[list[dict], dict]:
    """
    Dense vector retrieval against a ChromaDB collection.

    Embeds the query separately from the ChromaDB search so the two phases
    can be timed independently — this is the only way to tell whether a slow
    request is the embedding model or the DB search.

    Returns (results, timing_ms) where results is a list of
    {id, text, score, distance, metadata} sorted by score descending, and
    timing_ms is {"model_cold_start", "embed_ms", "db_query_ms"}.
    """
    ef = store.ef
    model_cold_start = getattr(ef, "_model", "n/a") is None

    t0 = time.perf_counter()
    query_embedding = ef([query])[0]
    embed_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    result = store.query(
        collection_name,
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    db_query_ms = (time.perf_counter() - t0) * 1000

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    results = [
        {
            "id": chunk_id,
            "text": text,
            "distance": distance,
            "score": 1.0 - distance,  # cosine similarity: higher = more similar
            "metadata": metadata,
        }
        for chunk_id, text, distance, metadata in zip(ids, documents, distances, metadatas)
    ]
    timing_ms = {
        "model_cold_start": model_cold_start,
        "embed_ms": round(embed_ms, 1),
        "db_query_ms": round(db_query_ms, 1),
    }
    return results, timing_ms


class DenseRetriever(BaseRetriever):
    """Retriever backed by ChromaDB dense vector search."""

    def __init__(self, store: ChromaStore, collection_name: str):
        self.store = store
        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        results, _ = dense_search(self.store, self.collection_name, query, top_k)
        return results

    def retrieve_with_timing(self, query: str, top_k: int) -> tuple[list[dict], dict]:
        return dense_search(self.store, self.collection_name, query, top_k)
