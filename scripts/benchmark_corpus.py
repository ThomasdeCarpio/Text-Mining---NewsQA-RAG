#!/usr/bin/env python3
"""Evaluate finalized chunking and indexing artifacts independently of RAG QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import chromadb
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_io import atomic_write_json, utc_now
from src.evaluation.metrics import (
    count_chunk_tokens,
    deduplication_rate,
    evaluate_chunking,
    semantic_integrity,
)
from src.evaluation.testset import canonical_json, sha256_file
from src.indexing.chroma_store import ChromaStore
from src.indexing.embeddings import get_embedding_function
from src.ingestion.chunker import load_chunks

REQUIRED_METADATA = {
    "article_id",
    "canonical_article_id",
    "chunk_index",
    "corpus_role",
    "dataset_split",
    "publisher",
    "title",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark chunk and index integrity.")
    parser.add_argument("--variant-manifest", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--self-retrieval-sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-self-retrieval", action="store_true")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.variant_manifest).read_text(encoding="utf-8"))
    config = yaml.safe_load(_resolve(args.config).read_text(encoding="utf-8")) or {}
    config_hash = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    if config_hash != manifest.get("pipeline", {}).get("config_sha256"):
        raise SystemExit("Config hash does not match the finalized variant manifest")

    chunks_record = manifest["artifacts"]["chunks"]
    chunks_path = _resolve(chunks_record["path"])
    if sha256_file(chunks_path) != chunks_record["sha256"]:
        raise SystemExit("Chunk artifact hash does not match the variant manifest")
    chunks = load_chunks(str(chunks_path))
    texts = [chunk["text"] for chunk in chunks]

    complete = sum(
        REQUIRED_METADATA <= set(chunk.get("metadata", {})) for chunk in chunks
    )
    chunk_size = int(config.get("chunking", {}).get("chunk_size", 512))
    token_counts, tokenizer = count_chunk_tokens(texts)

    database = manifest["database"]
    database_path = _resolve(database["path"])
    collection_name = database["collection"]
    collection = chromadb.PersistentClient(str(database_path)).get_collection(collection_name)
    indexed_count = collection.count()
    expected_count = int(database["chunk_count"])

    indexing = {
        "collection": collection_name,
        "expected_chunks": expected_count,
        "indexed_chunks": indexed_count,
        "count_matches_manifest": indexed_count == expected_count == len(chunks),
        "self_retrieval_recall@1": None,
        "self_retrieval_samples": 0,
        "write_latency_ms": None,
        "write_latency_note": "Measure during a fresh index build; the finalized collection is read-only for evaluation.",
    }
    if not args.skip_self_retrieval:
        sample = random.Random(args.seed).sample(
            chunks, min(args.self_retrieval_sample, len(chunks))
        )
        store = ChromaStore(str(database_path), get_embedding_function(config))
        hits = 0
        for chunk in sample:
            result = store.query(
                collection_name,
                query_texts=[chunk["text"]],
                n_results=1,
            )
            hits += int(bool(result["ids"][0]) and result["ids"][0][0] == chunk["id"])
        indexing["self_retrieval_recall@1"] = round(hits / len(sample), 4) if sample else 0.0
        indexing["self_retrieval_samples"] = len(sample)

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "variant_manifest": str(Path(args.variant_manifest)),
        "chunking": {
            **evaluate_chunking(chunks, token_counts, tokenizer),
            "chunk_size_limit": chunk_size,
            "chunk_size_compliance": round(
                sum(0 < count <= chunk_size for count in token_counts) / len(token_counts), 4
            ),
            "metadata_completeness": round(complete / len(chunks), 4),
            "required_metadata_fields": sorted(REQUIRED_METADATA),
            "deduplication_rate": deduplication_rate(texts),
            "semantic_integrity": semantic_integrity(texts),
        },
        "indexing": indexing,
    }
    atomic_write_json(args.output, report)
    print(f"Corpus report: {args.output}")
    print(f"Index count matches manifest: {indexing['count_matches_manifest']}")


if __name__ == "__main__":
    main()
