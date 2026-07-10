from src.retrieval import BaseRetriever
from src.indexing.bm25_index import BM25Index


class BM25Retriever(BaseRetriever):
    """
    Retriever backed by a BM25 sparse index.
    Keeps an in-memory chunk lookup so it can return full {id, text, metadata} results.
    """

    def __init__(self, bm25_index: BM25Index, chunk_lookup: dict[str, dict]):
        self._index = bm25_index
        self._lookup = chunk_lookup  # id → {id, text, metadata}

    @classmethod
    def from_chunks(cls, chunks: list[dict]) -> "BM25Retriever":
        """Build a BM25Retriever directly from a list of chunk dicts."""
        index = BM25Index()
        index.build(chunks)
        lookup = {c["id"]: c for c in chunks}
        return cls(index, lookup)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        raw = self._index.query(query, top_k)
        results = []
        for r in raw:
            chunk = self._lookup.get(r["id"], {})
            results.append({
                "id": r["id"],
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "score": r["score"],
            })
        return results


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining dense and BM25 via Reciprocal Rank Fusion (RRF).

    RRF formula:  score(id) = Σ weight_i * 1 / (k + rank_i)
    where k=60 is a stability constant (standard in RRF literature).

    dense_weight and sparse_weight scale the contribution of each retriever.
    """

    def __init__(
        self,
        dense: BaseRetriever,
        bm25: BM25Retriever,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        # Fetch more candidates from each retriever before merging
        fetch_k = min(top_k * 3, top_k + 30)

        dense_results = self.dense.retrieve(query, fetch_k)
        bm25_results = self.bm25.retrieve(query, fetch_k)

        rrf_scores: dict[str, float] = {}
        id_to_data: dict[str, dict] = {}

        for rank, r in enumerate(dense_results):
            rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0.0) + self.dense_weight * (
                1.0 / (self.rrf_k + rank + 1)
            )
            id_to_data[r["id"]] = r

        for rank, r in enumerate(bm25_results):
            rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0.0) + self.sparse_weight * (
                1.0 / (self.rrf_k + rank + 1)
            )
            if r["id"] not in id_to_data:
                id_to_data[r["id"]] = r

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for chunk_id in sorted_ids[:top_k]:
            entry = id_to_data[chunk_id].copy()
            entry["score"] = round(rrf_scores[chunk_id], 6)
            results.append(entry)

        return results
