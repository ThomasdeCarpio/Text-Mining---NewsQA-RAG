import os

from newsqa_rag.retrieval import BaseRetriever
from newsqa_rag.retrieval.dense import DenseRetriever
from newsqa_rag.retrieval.hybrid import BM25Retriever, HybridRetriever, SparseIndexRetriever
from newsqa_rag.indexing.chroma_store import ChromaStore
from newsqa_rag.indexing.bm25_index import BM25Index
from newsqa_rag.indexing.learned_sparse_index import LearnedSparseIndex


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
        retriever_type: "dense" | "bm25" | "sparse" | "hybrid"
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

    if retriever_type in ("bm25", "sparse", "hybrid"):
        sparse_cfg = config.get("retrieval", {}).get("sparse", {})
        sparse_retriever = _resolve_sparse(chunks, bm25_path, sparse_cfg)

        if retriever_type in ("bm25", "sparse"):
            return sparse_retriever

        # hybrid
        dense_retriever = DenseRetriever(store, collection_name)
        hybrid_cfg = config.get("retrieval", {}).get("hybrid", {})
        return HybridRetriever(
            dense=dense_retriever,
            sparse=sparse_retriever,
            dense_weight=hybrid_cfg.get("dense_weight", 0.7),
            sparse_weight=hybrid_cfg.get("sparse_weight", 0.3),
            rrf_k=hybrid_cfg.get("rrf_k", 60),
        )

    raise ValueError(
        f"Unknown retriever type: '{retriever_type}'. Supported: dense, bm25, sparse, hybrid."
    )


def _resolve_sparse(
    chunks: list[dict] | None,
    index_path: str | None,
    sparse_config: dict,
) -> BaseRetriever:
    method = str(sparse_config.get("method", "bm25")).lower()
    if method in {"bge-m3", "bge_m3", "learned"}:
        if chunks is None:
            raise ValueError("chunks must be provided for learned sparse retrieval")
        if not index_path or not os.path.exists(index_path):
            raise ValueError("BGE-M3 sparse retrieval requires an existing sparse index artifact")
        index = LearnedSparseIndex.load(index_path, device=sparse_config.get("device"))
        return SparseIndexRetriever(index, {chunk["id"]: chunk for chunk in chunks})
    index, lookup = _resolve_bm25(chunks, index_path, sparse_config=sparse_config)
    return BM25Retriever(index, lookup)


def _resolve_bm25(
    chunks: list[dict] | None,
    bm25_path: str | None,
    sparse_config: dict | None = None,
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

    sparse_cfg = sparse_config or {}
    variant = sparse_cfg.get("variant", "okapi")
    tokenizer_mode = sparse_cfg.get("tokenizer_mode", sparse_cfg.get("tokenizer", "simple"))

    print(f"Building BM25 index ({variant=}, {tokenizer_mode=}) from {len(chunks)} chunks...")
    bm25_index = BM25Index(variant=variant, tokenizer_mode=tokenizer_mode)
    bm25_index.build(chunks)
    chunk_lookup = {c["id"]: c for c in chunks}

    if bm25_path:
        bm25_index.save(bm25_path)
        print(f"BM25 index saved to {bm25_path}")

    return bm25_index, chunk_lookup
