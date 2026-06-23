"""
Utility to inspect and debug a ChromaDB collection.

Usage:
    # Show collection statistics:
    python scripts/inspect_collection.py \
        --db-path database/ \
        --collection basic_collection \
        --action stats

    # Sample random entries:
    python scripts/inspect_collection.py \
        --db-path database/ \
        --collection basic_collection \
        --action sample --n 5

    # Search for a specific article's chunks:
    python scripts/inspect_collection.py \
        --db-path database/ \
        --collection basic_collection \
        --action search --filter-article 001

    # Delete a collection:
    python scripts/inspect_collection.py \
        --db-path database/ \
        --collection basic_collection \
        --action delete

Args:
    --db-path           Path to ChromaDB persistent storage
    --collection        Collection name
    --action            One of: stats, sample, search, delete (default: stats)
    --n                 Number of samples for 'sample' action (default: 5)
    --filter-article    Article ID to filter by for 'search' action

Actions:
    stats   — Print count, metadata field summary, embedding dimension, HNSW config
    sample  — Print N random entries with full metadata
    search  — List all chunks for a given article_id
    delete  — Delete the entire collection (asks for confirmation)
"""
