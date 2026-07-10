# Indexing Pipeline Implementation Guide

**Module:** `src/indexing/`

This guide specifies what each file in the indexing module must implement. The indexing module takes chunked documents (output of `src/ingestion/`) and stores them in vector/sparse indexes for retrieval.

---

## Module Overview

```
embeddings.py ──── Embedding function (provider-agnostic)
chroma_store.py ── ChromaDB collection management (CRUD)
bm25_index.py ──── BM25 sparse index (for hybrid retrieval)
```

These modules are independent of each other — `chroma_store.py` uses `embeddings.py`, but `bm25_index.py` operates on raw text without embeddings.

---

## 1. `embeddings.py` — Embedding Function

**Purpose:** Provide a unified embedding interface that supports multiple providers (OpenAI API, Sentence-Transformers local) selectable via config.

### Public API

```python
def get_embedding_function(config: dict) -> EmbeddingFunction:
    """
    Factory function. Returns a ChromaDB-compatible EmbeddingFunction
    based on config["embedding"]["provider"].

    Args:
        config: Full config dict (from config.yaml).
                Reads config["embedding"]["provider"], 
                config["embedding"]["model_name"],
                config["embedding"]["dimensions"].

    Returns:
        An object implementing chromadb.EmbeddingFunction interface.
    """
```

### Provider: OpenAI

```python
# Use chromadb's built-in OpenAI function or wrap langchain_openai.OpenAIEmbeddings
# API key from environment variable OPENAI_API_KEY

class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str, dimensions: int): ...
    def __call__(self, input: Documents) -> Embeddings: ...
```

### Provider: Sentence-Transformers

```python
# Use sentence-transformers library for local inference
# No API key needed, runs on CPU/GPU

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str): ...
    def __call__(self, input: Documents) -> Embeddings: ...
```

### `get_info()` Method

Both implementations expose a `get_info()` method returning model metadata:

```python
def get_info(self) -> Dict[str, Any]:
    # Returns: {provider, model_name, output_dimensions, max_input_tokens, use_cases}
```

Used by `ChromaStore.get_collection_stats()` to report embedding details alongside collection statistics.

> **Note:** The `@register_embedding_function` decorator is intentionally **not** used. It imports `chromadb.utils.embedding_functions` which eagerly loads ChromaDB's built-in ONNX module, causing a native DLL conflict with PyTorch on Windows (exit 0xC0000005). Since we always pass `ef` manually to `ChromaStore`, the ChromaDB serialization registry is not needed.

### Contract

- Input: `Documents` = `List[str]` (list of text strings)
- Output: `Embeddings` = `List[List[float]]` (list of float vectors)
- Output dimension must match `config["embedding"]["dimensions"]`
- Must handle batching internally if the provider has request limits (e.g., OpenAI max 2048 texts per request)

---

## 2. `chroma_store.py` — ChromaDB Collection Manager

**Purpose:** Manage ChromaDB collections — create, upsert chunks, query, delete. This is the main interface between our pipeline and ChromaDB.

### Public API

```python
class ChromaStore:
    def __init__(self, db_path: str, embedding_function: EmbeddingFunction):
        """
        Initialize ChromaDB client with persistent storage.

        Args:
            db_path: Path to ChromaDB persistent storage directory.
            embedding_function: EmbeddingFunction instance from embeddings.py.
        """

    def get_or_create_collection(
        self,
        name: str,
        hnsw_config: dict | None = None
    ) -> Collection:
        """
        Get existing collection or create new one with given HNSW config.
        If collection exists, ignores hnsw_config (HNSW params are immutable after creation).

        Args:
            name: Collection name (e.g., "basic_collection").
            hnsw_config: HNSW parameters dict (see docs/database.md).

        Returns:
            chromadb.Collection object.
        """

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[dict]
    ) -> int:
        """
        Upsert a list of chunk dicts into the collection.
        Uses upsert (not add) so re-running the pipeline is idempotent.

        Args:
            collection_name: Target collection.
            chunks: List of chunk dicts from chunker.py output.
                    Each has "id", "text", "metadata" keys.

        Returns:
            Number of chunks upserted.
        """

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 10,
        where: dict | None = None
    ) -> dict:
        """
        Query the collection with a text string.

        Args:
            collection_name: Collection to search.
            query_text: Natural language query.
            n_results: Number of results to return.
            where: Optional metadata filter (ChromaDB where clause).

        Returns:
            ChromaDB query result dict with keys:
            "ids", "documents", "metadatas", "distances"
        """

    def get_collection_stats(self, collection_name: str) -> dict:
        """
        Return collection statistics: count, sample entries, metadata summary.
        Used by scripts/inspect_collection.py.
        """

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection entirely."""
```

### Upsert Batching

ChromaDB has a default batch limit. `upsert_chunks` must batch the input:

```python
CHROMA_BATCH_SIZE = 500  # safe default

for i in range(0, len(chunks), CHROMA_BATCH_SIZE):
    batch = chunks[i:i + CHROMA_BATCH_SIZE]
    collection.upsert(
        ids=[c["id"] for c in batch],
        documents=[c["text"] for c in batch],
        metadatas=[c["metadata"] for c in batch],
    )
```

### Contract

- `upsert_chunks` expects chunks in the schema defined in `docs/database.md`.
- Embedding is handled by ChromaDB internally via the `embedding_function` passed at collection creation — do NOT embed manually before upserting.
- `query` returns raw ChromaDB result format. Downstream modules (`src/retrieval/`) handle formatting.

---

## 3. `bm25_index.py` — BM25 Sparse Index

**Purpose:** Build and maintain a BM25 index over chunk texts for keyword-based retrieval. Used alongside ChromaDB dense retrieval for hybrid search.

### Public API

```python
class BM25Index:
    def __init__(self):
        """Initialize empty BM25 index."""

    def build(self, chunks: list[dict]) -> None:
        """
        Build BM25 index from chunk dicts.

        Args:
            chunks: List of chunk dicts (same format as chunker output).
                    Uses "id" and "text" fields.
        """

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        """
        Search BM25 index.

        Args:
            query_text: Natural language query.
            top_k: Number of results.

        Returns:
            List of {"id": str, "score": float} sorted by score descending.
        """

    def save(self, path: str) -> None:
        """Persist index to disk (pickle or JSON)."""

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load persisted index from disk."""
```

### Implementation Notes

- Use `rank_bm25.BM25Okapi` from the `rank_bm25` package.
- Tokenization: simple whitespace + lowercase. Do NOT use heavy NLP tokenizers.
- The index must store a mapping from internal index position → chunk ID, so results can be joined with ChromaDB results in `src/retrieval/hybrid.py`.
- `save`/`load` enables rebuilding the index only when data changes, not on every query.

### Storage

BM25 index files are stored alongside the ChromaDB database:

```
database/
    |- chroma/           # ChromaDB internal storage
    |__ bm25/
        |__ basic_collection.pkl   # Serialized BM25 index
```

---

## Integration: How Scripts Use These Modules

### `scripts/build_chroma_collection.py` (pipeline)

```python
# Pseudocode — shows how indexing modules are called

from src.ingestion.loader import load_articles
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.indexing.embeddings import get_embedding_function
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index

# 1. Ingest
docs = load_articles(source_dir)
docs = clean_documents(docs)
chunks = chunk_documents(docs, config["chunking"])

# 2. Index (dense)
ef = get_embedding_function(config)
store = ChromaStore(db_path, ef)
store.get_or_create_collection(collection_name, config["database"]["hnsw"])
store.upsert_chunks(collection_name, chunks)

# 3. Index (sparse)
bm25 = BM25Index()
bm25.build(chunks)
bm25.save(f"{db_path}/bm25/{collection_name}.pkl")
```

---

## Config Dependencies

The indexing module reads from `configs/config.yaml`:

```yaml
embedding:
  provider: "sentence-transformers"   # "openai" | "sentence-transformers"
  model_name: "all-MiniLM-L6-v2"
  dimensions: 384

database:
  hnsw:
    space: "cosine"
    ef_construction: 200
    max_neighbors: 16    # ChromaDB API key (not "M")
    ef_search: 50
```

---

## Testing Checklist

- [ ] `embeddings.py` factory returns correct provider based on config
- [ ] OpenAI embedding function produces vectors of correct dimension
- [ ] Sentence-Transformers embedding function works without API key
- [ ] Both embedding functions expose `get_info()` returning provider/dimensions/use_cases
- [ ] Importing `embeddings.py` does NOT trigger `onnxruntime` or `torch` loading
- [ ] `chroma_store.py` creates collection with correct HNSW config
- [ ] `chroma_store.py` upsert is idempotent (re-running doesn't duplicate)
- [ ] `chroma_store.py` batches large upserts correctly
- [ ] `chroma_store.py` query returns results in expected format
- [ ] `bm25_index.py` build + query returns relevant results
- [ ] `bm25_index.py` save/load roundtrip preserves index
- [ ] End-to-end: chunks from ingestion → upsert → query returns them
