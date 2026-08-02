# Database and chunk contract

Chroma stores dense vectors; BM25 uses the same chunk IDs for sparse retrieval.
`configs/config.yaml` is the source of truth for chunking, embeddings, and HNSW.

## Chunk record

```python
{
    "id": "<article_id>_chunk_<zero-padded-index>",
    "text": "...",
    "metadata": {
        "source": "CNN",
        "article_id": "001",
        "title": "...",
        "url": "...",
        "published_date": "2026-01-01T00:00:00Z",
        "author": "...",
        "category": "World",
        "chunk_index": 0,
        "total_chunks": 4,
    },
}
```

Use empty strings for unknown text metadata, not `None`. Normalize source names,
use ISO-8601 dates, and keep category values inside the vocabulary in config.

## Data flow

```text
HTML/files -> loader -> cleaner -> chunker -> embeddings -> Chroma
                                 \-> same chunks -> BM25
```

Build and inspect with:

```bash
python scripts/build_chroma_collection.py --help
python scripts/inspect_collection.py
```

The API reads the database and collection from `RAG_DB_PATH` and
`RAG_COLLECTION`; defaults live in `.env.example`. A benchmark instead reads
both from its locked variant manifest so testset and index cannot drift.

Changing embedding model/dimensions or chunking requires a new collection and
variant manifest. Never mix vector dimensions or chunk ID schemes inside an
existing collection.
