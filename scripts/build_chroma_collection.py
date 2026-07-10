"""
End-to-end pipeline: Load raw articles → Clean → Chunk → Embed → Index into ChromaDB + BM25.

Usage:
    python scripts/build_chroma_collection.py \\
        --source data/articles/ \\
        --db-path database/ \\
        --collection basic_collection \\
        --config configs/config.yaml

Cache behaviour (skip steps when intermediate outputs already exist):
    --force-clean   Re-clean all HTML files even if *_clean.json files exist
    --force-chunk   Re-chunk even if chunks JSONL exists
    --force-index   Drop and re-index the ChromaDB collection
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.loader import DataLoader
from src.ingestion.cleaner import NewsCleaner
from src.ingestion.chunker import get_chunker, save_chunks, load_chunks
from src.indexing.embeddings import get_embedding_function
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB collection from raw articles.")
    parser.add_argument("--source", required=True, help="Raw articles directory (.html files)")
    parser.add_argument("--db-path", default="database/chroma/", help="ChromaDB storage path")
    parser.add_argument("--collection", default="basic_collection", help="Collection name")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--cleaned-dir", default=None,
                        help="Directory for cleaned JSON files (default: <source>/../cleaned/)")
    parser.add_argument("--chunks-path", default=None,
                        help="JSONL path for cached chunks (default: database/chunks/<collection>.jsonl)")
    parser.add_argument("--bm25-path", default=None,
                        help="Pickle path for BM25 index (default: database/bm25/<collection>.pkl)")
    parser.add_argument("--force-clean", action="store_true", help="Re-run cleaning even if cache exists")
    parser.add_argument("--force-chunk", action="store_true", help="Re-chunk even if cache exists")
    parser.add_argument("--force-index", action="store_true", help="Drop and re-index collection")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve default paths
    db_path = args.db_path
    cleaned_dir = args.cleaned_dir or os.path.join(os.path.dirname(db_path), "cleaned")
    chunks_path = args.chunks_path or os.path.join(db_path, "chunks", f"{args.collection}.jsonl")
    bm25_path = args.bm25_path or os.path.join(db_path, "bm25", f"{args.collection}.pkl")

    # ------------------------------------------------------------------
    # Step 1: Clean
    # ------------------------------------------------------------------
    if args.force_clean or not _has_cleaned_files(cleaned_dir):
        print("\n--- Step 1: Cleaning HTML articles ---")
        cleaner = NewsCleaner()
        cleaner.process_directory(args.source, cleaned_dir)
    else:
        n = len([f for f in os.listdir(cleaned_dir) if f.endswith("_clean.json")])
        print(f"\n--- Step 1: Skipping clean (found {n} cached files in {cleaned_dir}) ---")

    # ------------------------------------------------------------------
    # Step 2: Chunk
    # ------------------------------------------------------------------
    if args.force_chunk or not os.path.exists(chunks_path):
        print("\n--- Step 2: Chunking ---")
        chunker = get_chunker(config)
        chunks = chunker.chunk_directory(cleaned_dir)
        save_chunks(chunks, chunks_path)
    else:
        print(f"\n--- Step 2: Loading cached chunks from {chunks_path} ---")
        chunks = load_chunks(chunks_path)
        print(f"Loaded {len(chunks)} chunks.")

    if not chunks:
        print("No chunks produced — aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Embed + Index into ChromaDB
    # ------------------------------------------------------------------
    ef = get_embedding_function(config)
    store = ChromaStore(db_path, ef)
    hnsw_config = config.get("database", {}).get("hnsw", {})

    if args.force_index:
        try:
            store.delete_collection(args.collection)
        except ValueError:
            pass  # didn't exist

    print("\n--- Step 3: Indexing into ChromaDB ---")
    col = store.get_or_create_collection(args.collection, hnsw_config)

    stats = store.get_collection_stats(args.collection)
    if stats["count"] == 0 or args.force_index:
        result = store.upsert_chunks(args.collection, chunks)
        print(f"Upserted {result['total']} chunks in {result['batches']} batches.")
    else:
        print(f"Collection already has {stats['count']} chunks — skipping upsert (use --force-index to redo).")

    # ------------------------------------------------------------------
    # Step 4: Build BM25 index
    # ------------------------------------------------------------------
    print("\n--- Step 4: Building BM25 index ---")
    if args.force_index or not os.path.exists(bm25_path):
        bm25 = BM25Index()
        bm25.build(chunks)
        bm25.save(bm25_path)
        print(f"BM25 index saved to {bm25_path} ({bm25.size} documents).")
    else:
        print(f"BM25 index already exists at {bm25_path} — skipping (use --force-index to redo).")

    final_stats = store.get_collection_stats(args.collection)
    print(f"\nDone. Collection '{args.collection}' has {final_stats['count']} chunks.")
    _print_embedding_info(final_stats)


def _has_cleaned_files(directory: str) -> bool:
    if not os.path.isdir(directory):
        return False
    return any(f.endswith("_clean.json") for f in os.listdir(directory))


def _print_embedding_info(stats: dict) -> None:
    ei = stats.get("embedding_info", {})
    if ei:
        print(
            f"Embedding: {ei.get('provider')} / {ei.get('model_name')} "
            f"(dims={ei.get('output_dimensions')})"
        )


if __name__ == "__main__":
    main()
