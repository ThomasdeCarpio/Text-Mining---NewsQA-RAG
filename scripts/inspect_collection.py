"""
Utility to inspect and manage a ChromaDB collection.

Usage:
    # Collection statistics + embedding info:
    python scripts/inspect_collection.py --action stats

    # Sample N random entries:
    python scripts/inspect_collection.py --action sample --n 5

    # Semantic search:
    python scripts/inspect_collection.py --action search --query "interest rate hike"

    # Filter chunks by article_id metadata:
    python scripts/inspect_collection.py --action search --filter-article abc123

    # Delete the collection (asks for confirmation):
    python scripts/inspect_collection.py --action delete

Args:
    --db-path       ChromaDB storage path (default: database/chroma/)
    --collection    Collection name (default: basic_collection)
    --config        Config file path (default: configs/config.yaml)
    --action        stats | sample | search | delete (default: stats)
    --n             Number of results for sample/search (default: 5)
    --query         Query string for semantic search
    --filter-article  article_id to filter chunks by
"""

import argparse
import json
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexing.embeddings import get_embedding_function
from src.indexing.chroma_store import ChromaStore


def main():
    parser = argparse.ArgumentParser(description="Inspect a ChromaDB collection.")
    parser.add_argument("--db-path", default="database/chroma/")
    parser.add_argument("--collection", default="basic_collection")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--action", choices=["stats", "sample", "search", "delete"],
                        default="stats")
    parser.add_argument("--n", type=int, default=5, help="Number of results")
    parser.add_argument("--query", default=None, help="Query for semantic search")
    parser.add_argument("--filter-article", default=None,
                        help="Filter chunks by article_id metadata")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ef = get_embedding_function(config)
    store = ChromaStore(args.db_path, ef)

    if args.action == "stats":
        _action_stats(store, args.collection)

    elif args.action == "sample":
        _action_sample(store, args.collection, args.n)

    elif args.action == "search":
        _action_search(store, args.collection, args.query, args.filter_article, args.n)

    elif args.action == "delete":
        _action_delete(store, args.collection)


def _action_stats(store: ChromaStore, collection_name: str) -> None:
    stats = store.get_collection_stats(collection_name)
    if not stats["exists"]:
        print(f"Collection '{collection_name}' does not exist.")
        return

    print(f"\n=== Collection: {collection_name} ===")
    print(f"Count       : {stats['count']}")
    print(f"Metadata    : {json.dumps(stats['metadata'], indent=2)}")

    ei = stats.get("embedding_info", {})
    if ei:
        print(f"\n--- Embedding Function ---")
        print(f"  Provider   : {ei.get('provider')}")
        print(f"  Model      : {ei.get('model_name')}")
        print(f"  Dimensions : {ei.get('output_dimensions')}")
        print(f"  Max tokens : {ei.get('max_input_tokens')}")
        print(f"  Use cases  : {ei.get('use_cases')}")

    sample = stats.get("sample", {})
    if sample and sample.get("ids"):
        print(f"\n--- Sample Entries (first {len(sample['ids'])}) ---")
        for i, (sid, doc, meta) in enumerate(zip(
            sample["ids"], sample.get("documents", []), sample.get("metadatas", [])
        )):
            print(f"  [{i+1}] id={sid}")
            print(f"       text={doc[:100]}...")
            print(f"       meta={meta}")


def _action_sample(store: ChromaStore, collection_name: str, n: int) -> None:
    try:
        result = store.get(collection_name, limit=n)
    except ValueError as e:
        print(e)
        return

    print(f"\n=== {len(result['ids'])} sample entries from '{collection_name}' ===")
    for i, (sid, doc, meta) in enumerate(zip(
        result["ids"], result.get("documents", []), result.get("metadatas", [])
    )):
        print(f"\n[{i+1}] ID: {sid}")
        print(f"  Text    : {doc[:200]}...")
        print(f"  Metadata: {json.dumps(meta, indent=4)}")


def _action_search(
    store: ChromaStore,
    collection_name: str,
    query: str | None,
    filter_article: str | None,
    n: int,
) -> None:
    if query:
        print(f"\n=== Semantic search: '{query}' ===")
        where = {"article_id": filter_article} if filter_article else None
        try:
            result = store.query(
                collection_name,
                query_texts=query,
                n_results=n,
                where=where,
            )
        except ValueError as e:
            print(e)
            return

        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        for i, (sid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
            print(f"\n[{i+1}] ID: {sid}  distance: {dist:.4f}")
            print(f"  Text    : {doc[:200]}...")
            print(f"  Metadata: {json.dumps(meta, indent=4)}")

    elif filter_article:
        print(f"\n=== Chunks for article_id='{filter_article}' ===")
        try:
            result = store.get(
                collection_name,
                where={"article_id": filter_article},
            )
        except ValueError as e:
            print(e)
            return

        print(f"Found {len(result['ids'])} chunk(s).")
        for i, (sid, doc, meta) in enumerate(zip(
            result["ids"], result.get("documents", []), result.get("metadatas", [])
        )):
            print(f"\n[{i+1}] ID: {sid}")
            print(f"  Text    : {doc[:200]}...")
            print(f"  Metadata: {json.dumps(meta, indent=4)}")
    else:
        print("Provide --query for semantic search or --filter-article to list chunks by article.")


def _action_delete(store: ChromaStore, collection_name: str) -> None:
    stats = store.get_collection_stats(collection_name)
    if not stats["exists"]:
        print(f"Collection '{collection_name}' does not exist.")
        return

    confirm = input(
        f"Delete collection '{collection_name}' ({stats['count']} chunks)? [y/N]: "
    ).strip().lower()
    if confirm == "y":
        store.delete_collection(collection_name)
    else:
        print("Aborted.")


if __name__ == "__main__":
    main()
