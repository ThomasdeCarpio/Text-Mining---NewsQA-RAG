"""
CLI tool for ad-hoc RAG queries. Useful for testing and debugging the retrieval pipeline.

Usage:
    # Single question mode:
    python scripts/query.py \
        --db-path database/ \
        --collection basic_collection \
        --question "What happened in the 2024 US election?"

    # Interactive mode (omit --question):
    python scripts/query.py \
        --db-path database/ \
        --collection basic_collection

Args:
    --db-path       Path to ChromaDB persistent storage
    --collection    Collection name to query
    --question      Single question string (omit for interactive REPL mode)
    --top-k         Number of retrieval results (default: from config)
    --no-rerank     Disable reranker even if config enables it
    --config        Path to config.yaml (default: configs/config.yaml)

Output:
    Prints the generated answer followed by retrieved source chunks with metadata.

    Example:
        Answer: The 2024 US election resulted in...

        Sources:
        [1] (CNN, 2024-11-06) Title of article...
            "...relevant chunk text..."
        [2] (Reuters, 2024-11-06) Another article...
            "...relevant chunk text..."
"""
