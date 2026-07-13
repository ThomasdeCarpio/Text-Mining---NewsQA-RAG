"""
Build a small article-grouped NewsQA test set in one command, so later evaluation is trivial.

Groups NewsQA by article (not per-question), maps each answer's evidence span to chunk IDs, and
writes a JSONL test set (schema: docs/evaluation.md section 6.1). With --build-collection it also
ingests the same chunks into a Chroma collection, so the set is immediately runnable end to end.

Usage:
    # dataset only
    python scripts/build_mini_testset.py --n-articles 15 --output data/testset_mini.jsonl

    # dataset + matching collection, then it's ready to score:
    python scripts/build_mini_testset.py --n-articles 15 --output data/testset_mini.jsonl \\
        --build-collection --collection newsqa_mini --db-path data/chroma_db
    python scripts/run_benchmark.py --retriever dense --testset data/testset_mini.jsonl \\
        --collection newsqa_mini --db-path data/chroma_db --report-dir reports/mini

Args:
    --n-articles        Number of articles to select (default: 15)
    --max-scan          NewsQA rows to stream while grouping (default: 800)
    --output            Output JSONL path (default: data/testset_mini.jsonl)
    --config            Config file (default: configs/config.yaml)
    --split             HuggingFace split: train/validation/test (default: train)
    --build-collection  Also ingest the chunks into a Chroma collection
    --collection        Collection name for --build-collection (default: newsqa_mini)
    --db-path           ChromaDB path for --build-collection (default: data/chroma_db)
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.chunker import get_chunker
from src.evaluation.testset import build_article_testset, save_testset


def main():
    p = argparse.ArgumentParser(description="Build a mini article-grouped NewsQA test set.")
    p.add_argument("--n-articles", type=int, default=15)
    p.add_argument("--max-scan", type=int, default=800)
    p.add_argument("--output", default="data/testset_mini.jsonl")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--build-collection", action="store_true",
                   help="Also ingest the chunks into a Chroma collection")
    p.add_argument("--collection", default="newsqa_mini")
    p.add_argument("--db-path", default="data/chroma_db")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    chunker = get_chunker(config)

    print(f"Streaming NewsQA ({args.split}) and grouping into {args.n_articles} articles ...")
    entries, all_chunks = build_article_testset(
        chunker, n_articles=args.n_articles, max_scan=args.max_scan, split=args.split,
    )

    mapped = sum(1 for e in entries if e["relevant_chunk_ids"])
    n_articles = len({e["article_key"] for e in entries})
    save_testset(entries, args.output)
    print(f"Wrote {args.output}")
    print(f"  Articles  : {n_articles}")
    print(f"  Questions : {len(entries)}")
    print(f"  Mapped    : {mapped}/{len(entries)} have relevant_chunk_ids")
    if mapped < len(entries):
        print(f"  WARNING   : {len(entries) - mapped} questions did not map to any chunk.")

    if args.build_collection:
        import json

        from src.indexing.embeddings import get_embedding_function
        from src.indexing.chroma_store import ChromaStore

        print(f"\nIngesting {len(all_chunks)} chunks into collection '{args.collection}' ...")
        store = ChromaStore(args.db_path, get_embedding_function(config))
        store.get_or_create_collection(args.collection, hnsw_config=config.get("database", {}).get("hnsw"))
        print("Upserted:", store.upsert_chunks(args.collection, all_chunks))

        # Also dump chunks JSONL (mirrors the newsqa_cnn layout) so bm25/hybrid retrieval works.
        chunks_path = os.path.join(args.db_path, "chunks", f"{args.collection}.jsonl")
        os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Chunks JSONL: {chunks_path}")
        print(f"\nReady. Score it with:\n"
              f"  python scripts/run_benchmark.py --retriever dense "
              f"--testset {args.output} --collection {args.collection} "
              f"--db-path {args.db_path} --report-dir reports/mini")


if __name__ == "__main__":
    main()
