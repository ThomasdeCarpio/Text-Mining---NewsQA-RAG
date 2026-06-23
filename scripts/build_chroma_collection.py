"""
End-to-end pipeline: Load raw articles → Clean → Chunk → Embed → Index into ChromaDB + BM25.

Usage:
    python scripts/build_chroma_collection.py \
        --source data/articles/ \
        --db-path database/ \
        --collection basic_collection \
        --config configs/config.yaml

Args:
    --source        Path to folder containing raw article files (.html, .txt, .md)
    --db-path       Path to ChromaDB persistent storage directory
    --collection    Name of the ChromaDB collection to create or update
    --config        Path to config.yaml (default: configs/config.yaml)

Pipeline steps:
    1. Load articles from source directory        (src.ingestion.loader)
    2. Clean and normalize text                   (src.ingestion.cleaner)
    3. Chunk documents according to config         (src.ingestion.chunker)
    4. Initialize embedding function from config   (src.indexing.embeddings)
    5. Upsert chunks into ChromaDB collection      (src.indexing.chroma_store)
    6. Build and save BM25 sparse index            (src.indexing.bm25_index)

Output:
    - ChromaDB collection at <db-path>/
    - BM25 index at <db-path>/bm25/<collection>.pkl
"""
