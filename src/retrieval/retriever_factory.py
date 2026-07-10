from src.retrieval import BaseRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import BM25Retriever, HybridRetriever
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index


def get_retriever(
    retriever_type: str,
    config: dict,
    store: ChromaStore,
    collection_name: str,
    chunks: list[dict] | None = None,
    bm25_path: str | None = None,
) -> BaseRetriever:
    """
    Factory for retriever instances.

    Args:
        retriever_type: "dense" | "bm25" | "hybrid"
        config: Full config dict (reads retrieval.hybrid weights).
        store: ChromaStore instance (used by dense + hybrid).
        collection_name: ChromaDB collection name.
        chunks: Chunk list needed to build BM25 index (bm25/hybrid only).
               If None and bm25_path exists, loads from disk.
        bm25_path: Path to persisted BM25 pickle. If provided and exists, loads
                   from disk instead of rebuilding from chunks.

    Returns:
        A BaseRetriever instance.
    """
    if retriever_type == "dense":
        return DenseRetriever(store, collection_name)

    if retriever_type in ("bm25", "hybrid"):
        bm25_index, chunk_lookup = _resolve_bm25(chunks, bm25_path)

        bm25_retriever = BM25Retriever(bm25_index, chunk_lookup)

        if retriever_type == "bm25":
            return bm25_retriever

        # hybrid
        dense_retriever = DenseRetriever(store, collection_name)
        hybrid_cfg = config.get("retrieval", {}).get("hybrid", {})
        return HybridRetriever(
            dense=dense_retriever,
            bm25=bm25_retriever,
            dense_weight=hybrid_cfg.get("dense_weight", 0.7),
            sparse_weight=hybrid_cfg.get("sparse_weight", 0.3),
        )

    raise ValueError(
        f"Unknown retriever type: '{retriever_type}'. Supported: 'dense', 'bm25', 'hybrid'."
    )


def _resolve_bm25(
    chunks: list[dict] | None, bm25_path: str | None
) -> tuple[BM25Index, dict[str, dict]]:
    """Load BM25 from disk if available, otherwise build from chunks."""
    import os

    if bm25_path and os.path.exists(bm25_path):
        print(f"Loading BM25 index from {bm25_path}...")
        bm25_index = BM25Index.load(bm25_path)
        if chunks is None:
            raise ValueError(
                "chunks must be provided to build the chunk lookup even when loading BM25 from disk."
            )
        chunk_lookup = {c["id"]: c for c in chunks}
        return bm25_index, chunk_lookup

    if chunks is None:
        raise ValueError(
            "chunks must be provided to build BM25 index (no bm25_path found on disk)."
        )

    print(f"Building BM25 index from {len(chunks)} chunks...")
    bm25_index = BM25Index()
    bm25_index.build(chunks)
    chunk_lookup = {c["id"]: c for c in chunks}

    if bm25_path:
        bm25_index.save(bm25_path)
        print(f"BM25 index saved to {bm25_path}")

    return bm25_index, chunk_lookup
