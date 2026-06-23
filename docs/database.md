# Database Usage and Data Object Contracts

This is the guide for building the data ingestion and chunking pipeline. All code relating to ingesting and indexing data must follow these contracts to maintain consistency.

**Tech stack:** ChromaDB, Custom Embedding Function (OpenAI / Sentence-Transformers via config).

The resulting database is used for a Question & Answer RAG system. Each data point is a chunk of a news article crawled from CNN, Reuters, etc.

---

## ID Format

```
<article_id>_chunk_<chunk_index>
```

- `article_id`: filename without extension (e.g., file `001.html` → `article_id = "001"`)
- `chunk_index`: zero-padded 3-digit index within the article (e.g., `000`, `001`, `012`)
- Example: `001_chunk_003` = 4th chunk of article `001.html`

---

## Metadata Contract

Every chunk added to the database **must** include this metadata schema.

```python
from typing import TypedDict, Optional

class NewsChunkMetadata(TypedDict):
    source: str            # News outlet: "CNN", "Reuters", "BBC", etc.
    article_id: str        # Filename-based ID (e.g., "001")
    title: str             # Article title (extracted from content or filename)
    url: str               # Original article URL (empty string if unavailable)
    published_date: str    # ISO 8601 format: "YYYY-MM-DDTHH:MM:SSZ" (empty string if unavailable)
    author: str            # Author name (empty string if unavailable)
    category: str          # Topic category: "Politics", "Tech", "Business", "World", etc.
    chunk_index: int       # Position of this chunk within the article (0-based)
    total_chunks: int      # Total number of chunks for this article
```

**Rules:**
- Fields that cannot be extracted should be set to `""` (empty string), never `None`.
- `source` should be normalized to a canonical name (e.g., always "CNN" not "cnn.com" or "CNN News").
- `published_date` must be ISO 8601 or empty string. No other date formats.
- `category` should use a controlled vocabulary defined in `configs/config.yaml`.

---

## Chunking Strategies

The chunking strategy is **configurable** via `configs/config.yaml`. The `src/ingestion/chunker.py` module must support swapping strategies without changing downstream code.

### Supported Strategies

| Strategy | Config Key | Description |
|---|---|---|
| Recursive Character | `recursive` | LangChain's `RecursiveCharacterTextSplitter`. Splits by paragraph → sentence → character. Best general-purpose option. |
| Semantic | `semantic` | Uses embedding similarity to find natural breakpoints between sections. Better coherence per chunk but slower. |
| Sentence-based | `sentence` | Groups N consecutive sentences into a chunk with sentence-level overlap. Preserves sentence boundaries. |

### Chunking Parameters (in `config.yaml`)

```yaml
chunking:
  strategy: "recursive"      # "recursive" | "semantic" | "sentence"
  chunk_size: 512             # Target chunk size in tokens
  chunk_overlap: 64           # Overlap between consecutive chunks in tokens
  # Sentence strategy only:
  sentences_per_chunk: 8
  sentence_overlap: 2
```

### Contract

The chunker must return a list of dictionaries, each containing:

```python
{
    "id": "001_chunk_003",           # ID format as defined above
    "text": "The chunked text...",   # The actual chunk content
    "metadata": { ... }             # NewsChunkMetadata as defined above
}
```

---

## Database Configuration

When initializing a new collection, follow this configuration strictly.

**Docs:** https://docs.trychroma.com/docs/collections/configure#what-is-an-hnsw-index

### HNSW Parameters Reference

| Parameter | Description | Default |
|---|---|---|
| `space` | Distance metric: `"cosine"`, `"l2"`, `"ip"` | `"l2"` |
| `ef_construction` | Candidate list size during index build. Higher = better quality, slower build. | 100 |
| `ef_search` | Candidate list size during search. Higher = better recall, slower query. Modifiable after creation. | 100 |
| `M` (max_neighbors) | Max connections per node. Higher = denser graph, better recall, more memory. | 16 |
| `num_threads` | Threads for index operations. Modifiable after creation. | CPU count |
| `batch_size` | Vectors per batch during index ops. Modifiable after creation. | 100 |
| `sync_threshold` | When to sync with persistent storage. Modifiable after creation. | 1000 |
| `resize_factor` | Index growth factor on resize. Modifiable after creation. | 1.2 |

### Collection: `basic_collection`

First collection, optimized for cosine similarity RAG.

```python
configuration = {
    "hnsw": {
        "space": "cosine",
        "ef_construction": 200,
        "M": 16,
        "ef_search": 50,
    }
}
```

**Note:** `ef_search` is set conservatively at 50 for faster queries. Increase to 100-150 if recall is insufficient during evaluation.

---

## Embedding Model

**Docs:** https://docs.trychroma.com/docs/embeddings/embedding-functions

**Implementation:** `src/indexing/embeddings.py`

The embedding function is configurable between OpenAI API and local Sentence-Transformers models via `configs/config.yaml`.

### Supported Providers

| Provider | Model Example | Dimensions | Notes |
|---|---|---|---|
| OpenAI | `text-embedding-3-small` | 1536 | API-based, fast, costs per token |
| OpenAI | `text-embedding-3-large` | 3072 | Higher accuracy, higher cost |
| Sentence-Transformers | `all-MiniLM-L6-v2` | 384 | Free, local, needs compute |
| Sentence-Transformers | `all-mpnet-base-v2` | 768 | Better accuracy than MiniLM |

### Config (in `config.yaml`)

```yaml
embedding:
  provider: "openai"                    # "openai" | "sentence-transformers"
  model_name: "text-embedding-3-small"  # Model identifier
  dimensions: 1536                      # Output dimensions (must match HNSW space)
```

### Contract

The embedding function must implement ChromaDB's `EmbeddingFunction` interface:

```python
class EmbeddingFunction:
    def __call__(self, input: Documents) -> Embeddings: ...
```

Where `Documents = List[str]` and `Embeddings = List[List[float]]`.

---

## Data Flow Summary

```
Raw Files (HTML/TXT/MD)
    │
    ▼
loader.py ──── Load into Document objects
    │
    ▼
cleaner.py ─── Normalize text, strip HTML, fix encoding ───> bm25_index.py ──── Build/update BM25 sparse index (parallel)
    │
    ▼
chunker.py ─── Split into chunks (strategy from config)
    │
    ▼
embeddings.py ─ Generate embeddings (model from config)
    │
    ▼
chroma_store.py ── Upsert into ChromaDB collection
```
