
Core task:
- Evaluate full pipeline, consists of:
    + Ingestion (load, clean, chunking)
    + Indexing (embed, storing) -> database + process input promts
    + Retrieval (query -> retrieve)
    + Reranker (ranking the retrieved chunks)
    + generator (LLM calls)
    + agent loop