import os
import pickle

from rank_bm25 import BM25Okapi


class BM25Index:
    """
    BM25 sparse index over chunk texts.
    Stores a position→chunk_id mapping so query results can be joined
    with ChromaDB results in hybrid retrieval.
    """

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._id_map: list[str] = []

    def build(self, chunks: list[dict]) -> None:
        """
        Build BM25 index from chunk dicts.

        Args:
            chunks: list of {id, text, metadata} — same format as chunker output.
        """
        self._id_map = [c["id"] for c in chunks]
        tokenized = [c["text"].lower().split() for c in chunks]
        self._index = BM25Okapi(tokenized)

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        """
        Search BM25 index.

        Returns:
            list of {id, score} sorted by score descending, only includes results with score > 0.
        """
        if self._index is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokens = query_text.lower().split()
        scores = self._index.get_scores(tokens)

        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {"id": self._id_map[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]

    def save(self, path: str) -> None:
        """Persist index to disk as pickle."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"index": self._index, "id_map": self._id_map}, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load persisted index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._index = data["index"]
        obj._id_map = data["id_map"]
        return obj

    @property
    def size(self) -> int:
        return len(self._id_map)
