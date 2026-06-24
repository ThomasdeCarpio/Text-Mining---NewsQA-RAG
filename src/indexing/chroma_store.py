from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings, Collection
from datetime import datetime
import chromadb

CHROMA_BATCH_SIZE = 500


class ChromaStore:
    def __init__(self, db_path: str, embedding_function: EmbeddingFunction):
        """
        Initialize ChromaDB client with persistent storage.

        Args:
            db_path: Path to ChromaDB persistent storage directory.
            embedding_function: EmbeddingFunction instance from embeddings.py.
        """
        self.client = chromadb.PersistentClient(db_path)
        self.ef = embedding_function

    def get_or_create_collection(
        self,
        name: str,
        hnsw_config: dict | None = None
    ) -> Collection:
        """
        Get existing collection or create new one with given HNSW config.
        If collection exists, ignores hnsw_config (HNSW params are immutable after creation).

        Returns:
            chromadb.Collection object.
        """
        metadata = {"hnsw:space": hnsw_config.get("space", "l2")} if hnsw_config else None

        try:
            col = self.client.get_collection(
                name=name,
                embedding_function=self.ef,
            )
            if hnsw_config:
                col_metadata = col.metadata or {}
                current_space = col_metadata.get("hnsw:space", "l2")
                requested_space = hnsw_config.get("space", "l2")
                if current_space != requested_space:
                    print(f"⚠️ Collection '{name}' exists with space='{current_space}', requested='{requested_space}'. HNSW params are immutable.")
                else:
                    print(f"✅ Connected to existing collection '{name}'.")
            else:
                print(f"✅ Connected to existing collection '{name}'.")
            return col

        except Exception:
            print(f"✨ Collection '{name}' not found. Creating...")
            configuration = None
            if hnsw_config:
                configuration = {"hnsw": hnsw_config}

            col = self.client.create_collection(
                name=name,
                embedding_function=self.ef,
                metadata=metadata,
                configuration=configuration,
            )
            print(f"✅ Created collection '{name}'.")
            return col

    def get_collection_stats(self, collection_name: str) -> dict:
        """
        Return collection statistics: existence, count, sample entries, metadata summary,
        and embedding function info.
        """
        ef_info = self.ef.get_info() if hasattr(self.ef, "get_info") else {}

        try:
            col = self.client.get_collection(
                name=collection_name,
                embedding_function=self.ef,
            )
        except Exception:
            return {
                "exists": False,
                "name": collection_name,
                "count": 0,
                "sample": [],
                "metadata": {},
                "embedding_info": ef_info,
            }

        count = col.count()
        sample = {}
        if count > 0:
            sample = col.peek(limit=min(5, count))

        return {
            "exists": True,
            "name": collection_name,
            "count": count,
            "sample": sample,
            "metadata": col.metadata or {},
            "embedding_info": ef_info,
        }

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[dict]
    ) -> dict:
        """
        Upsert a list of chunk dicts into the collection.
        Uses upsert (not add) so re-running the pipeline is idempotent.
        Batches internally to respect ChromaDB limits.

        Returns:
            Dict with "total", "batches" keys.
        """
        if not chunks:
            return {"total": 0, "batches": 0}

        try:
            col = self.client.get_collection(
                name=collection_name,
                embedding_function=self.ef,
            )
        except Exception:
            raise ValueError(f"Collection '{collection_name}' does not exist. Create it first with get_or_create_collection().")

        batch_count = 0
        for i in range(0, len(chunks), CHROMA_BATCH_SIZE):
            batch = chunks[i:i + CHROMA_BATCH_SIZE]
            col.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )
            batch_count += 1

        return {
            "total": len(chunks),
            "batches": batch_count,
        }

    def query(self, collection_name: str, **kwargs) -> dict:
        """
        Query the collection. Passes all kwargs directly to ChromaDB Collection.query().

        Supported kwargs (mirrors ChromaDB API):
            query_texts: str | list[str] — text query (embedded automatically)
            query_embeddings: list[list[float]] — raw embedding vectors
            n_results: int — number of results (default 10)
            where: dict — metadata filter
            where_document: dict — document content filter
            include: list[str] — fields to include (default ["metadatas", "documents", "distances"])
            ids: str | list[str] — filter by IDs

        Returns:
            ChromaDB QueryResult with keys: "ids", "documents", "metadatas", "distances", etc.
        """
        try:
            col = self.client.get_collection(
                name=collection_name,
                embedding_function=self.ef,
            )
        except Exception:
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        return col.query(**kwargs)

    def get(self, collection_name: str, **kwargs) -> dict:
        """
        Get data from a collection. Passes all kwargs directly to ChromaDB Collection.get().

        Supported kwargs (mirrors ChromaDB API):
            ids: str | list[str] — specific IDs to fetch
            where: dict — metadata filter
            where_document: dict — document content filter
            limit: int — max results
            offset: int — skip first N results
            include: list[str] — fields to include (default ["metadatas", "documents"])

        Returns:
            ChromaDB GetResult with keys: "ids", "documents", "metadatas", etc.
        """
        try:
            col = self.client.get_collection(
                name=collection_name,
                embedding_function=self.ef,
            )
        except Exception:
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        return col.get(**kwargs)

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection entirely."""
        try:
            self.client.delete_collection(name=collection_name)
            print(f"🗑️ Deleted collection '{collection_name}'.")
        except Exception:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
