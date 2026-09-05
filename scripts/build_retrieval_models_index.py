#!/usr/bin/env python3
"""Build and index multiple Dense Embedding Models and Sparse BM25 Variants for Phase 1 Retrieval Tournament."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from newsqa_rag.indexing.bm25_index import BM25Index
from newsqa_rag.indexing.chroma_store import ChromaStore
from newsqa_rag.indexing.embeddings import get_embedding_function
from newsqa_rag.indexing.learned_sparse_index import LearnedSparseIndex
from newsqa_rag.evaluation.benchmark_io import stable_hash
from newsqa_rag.evaluation.testset import sha256_file


DENSE_MODELS = [
    {"name": "all-MiniLM-L6-v2", "provider": "sentence-transformers", "dims": 384},
    {"name": "BAAI/bge-small-en-v1.5", "provider": "sentence-transformers", "dims": 384},
    {"name": "intfloat/e5-base-v2", "provider": "sentence-transformers", "dims": 768},
    {"name": "BAAI/bge-large-en-v1.5", "provider": "sentence-transformers", "dims": 1024},
]

SPARSE_CONFIGS = [
    {"id": "bm25_okapi_simple", "variant": "okapi", "tokenizer_mode": "simple"},
    {"id": "bm25_plus_simple", "variant": "plus", "tokenizer_mode": "simple"},
    {"id": "bm25_okapi_stemmed", "variant": "okapi", "tokenizer_mode": "stem"},
    {"id": "bge_m3_sparse", "method": "bge-m3", "model_name": "BAAI/bge-m3"},
]


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks-path",
        default=str(PROJECT_ROOT / "data/evaluation/newsqa_200_11064/final_deduplicated/chunks.jsonl"),
        help="Path to chunks.jsonl to index.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data/evaluation/retrieval_models_index"),
        help="Directory to save generated indexes and configs.",
    )
    parser.add_argument(
        "--base-config",
        default=str(PROJECT_ROOT / "configs/config.yaml"),
        help="Base config YAML.",
    )
    parser.add_argument(
        "--base-variant-manifest",
        default=str(PROJECT_ROOT / "evaluation/manifests/newsqa_200_11064.deduplicated.variant.json"),
        help="Canonical manifest cloned for each retrieval profile.",
    )
    parser.add_argument(
        "--dense-models",
        nargs="+",
        default=[m["name"] for m in DENSE_MODELS],
        help="List of dense embedding models to index.",
    )
    parser.add_argument(
        "--sparse-ids",
        nargs="+",
        default=[s["id"] for s in SPARSE_CONFIGS],
        help="List of sparse configurations to build.",
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Skip building dense vector indexes.",
    )
    parser.add_argument(
        "--skip-sparse",
        action="store_true",
        help="Skip building sparse BM25 indexes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for Chroma embedding ingestion.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing model indexes.")
    parser.add_argument("--device", default=None, help="Torch device for dense or BGE-M3 encoding.")
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def write_profile_manifest(base_path, output_path, config_path, database=None, sparse_path=None):
    value = json.loads(Path(base_path).read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    value.setdefault("pipeline", {}).update({
        "config_path": str(config_path),
        "config_sha256": stable_hash(config),
        "embedding": config.get("embedding", {}),
    })
    if database:
        value.setdefault("database", {}).update(database)
    if sparse_path:
        value.setdefault("artifacts", {})["bm25"] = {
            "path": str(sparse_path), "sha256": sha256_file(sparse_path)
        }
    Path(output_path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_dense_index(
    chunks: list[dict],
    model_name: str,
    output_dir: Path,
    base_config: dict,
    batch_size: int = 512,
    overwrite: bool = False,
    device: str | None = None,
) -> dict:
    slug = model_name.replace("/", "_").replace("-", "_").lower()
    db_path = output_dir / f"chroma_{slug}"
    db_path.mkdir(parents=True, exist_ok=True)
    collection_name = f"chunks_{slug}"

    cfg = yaml.safe_load(yaml.dump(base_config))
    cfg.setdefault("embedding", {})
    cfg["embedding"]["provider"] = "sentence-transformers"
    cfg["embedding"]["model_name"] = model_name
    spec = next(item for item in DENSE_MODELS if item["name"] == model_name)
    cfg["embedding"]["dimensions"] = spec["dims"]
    if device:
        cfg["embedding"]["device"] = device

    config_path = output_dir / f"config_dense_{slug}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"\n[DENSE] Indexing {len(chunks)} chunks with '{model_name}' into {db_path}...")
    start_time = time.perf_counter()

    embedding_fn = get_embedding_function(cfg)
    store = ChromaStore(str(db_path), embedding_fn)
    stats = store.get_collection_stats(collection_name)
    if stats["exists"] and stats["count"]:
        if not overwrite:
            raise RuntimeError(f"Collection {collection_name!r} already exists; pass --overwrite")
        store.delete_collection(collection_name)
    store.get_or_create_collection(
        collection_name,
        hnsw_config=cfg.get("database", {}).get("hnsw"),
    )

    # Ingest chunks in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        documents = [c.get("text", "") for c in batch]
        embeddings = embedding_fn.embed_documents(documents)
        store.upsert_chunks(collection_name, batch, embeddings=embeddings)
        print(f"  Ingested {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...", end="\r")

    elapsed = time.perf_counter() - start_time
    print(f"\n[DENSE] Done indexing '{model_name}' in {elapsed:.2f} seconds.")

    return {
        "model_name": model_name,
        "slug": slug,
        "db_path": str(db_path),
        "collection_name": collection_name,
        "config_path": str(config_path),
        "elapsed_seconds": round(elapsed, 2),
        "dimensions": spec["dims"],
        "chunk_count": store.get_collection_stats(collection_name)["count"],
    }


def build_sparse_index(
    chunks: list[dict],
    sparse_spec: dict,
    output_dir: Path,
    base_config: dict,
    device: str | None = None,
) -> dict:
    sparse_id = sparse_spec["id"]
    method = sparse_spec.get("method", "bm25")
    variant = sparse_spec.get("variant")
    tokenizer_mode = sparse_spec.get("tokenizer_mode")
    index_path = output_dir / f"{sparse_id}.pkl"

    cfg = yaml.safe_load(yaml.dump(base_config))
    cfg.setdefault("retrieval", {}).setdefault("sparse", {})
    cfg["retrieval"]["sparse"].update({"method": method})
    if device:
        cfg["retrieval"]["sparse"]["device"] = device
    if variant:
        cfg["retrieval"]["sparse"].update({"variant": variant, "tokenizer_mode": tokenizer_mode})
    else:
        cfg["retrieval"]["sparse"]["model_name"] = sparse_spec["model_name"]

    config_path = output_dir / f"config_sparse_{sparse_id}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"\n[SPARSE] Building '{sparse_id}' ({variant=}, {tokenizer_mode=}) over {len(chunks)} chunks...")
    start_time = time.perf_counter()

    if method == "bge-m3":
        sparse_index = LearnedSparseIndex(model_name=sparse_spec["model_name"], device=device)
        sparse_index.build(chunks)
        sparse_index.save(str(index_path))
    else:
        sparse_index = BM25Index(variant=variant, tokenizer_mode=tokenizer_mode)
        sparse_index.build(chunks)
        sparse_index.save(str(index_path))

    elapsed = time.perf_counter() - start_time
    print(f"[SPARSE] Saved '{sparse_id}' to {index_path} in {elapsed:.2f} seconds.")

    return {
        "sparse_id": sparse_id,
        "variant": variant,
        "tokenizer_mode": tokenizer_mode,
        "method": method,
        "model_name": sparse_spec.get("model_name"),
        "index_path": str(index_path),
        "config_path": str(config_path),
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> None:
    args = parse_args()
    chunks_path = Path(args.chunks_path).resolve()
    if not chunks_path.exists():
        raise SystemExit(f"Chunks file not found at: {chunks_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.base_config).resolve().open(encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    base_manifest = Path(args.base_variant_manifest).resolve()
    if not base_manifest.exists():
        raise SystemExit(f"Base variant manifest not found at: {base_manifest}")

    print(f"Loading chunks from {chunks_path}...")
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks.")

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunks_path": str(chunks_path),
        "total_chunks": len(chunks),
        "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "dense_indexes": {},
        "sparse_indexes": {},
    }

    if not args.skip_dense:
        dense_to_build = [m for m in DENSE_MODELS if m["name"] in args.dense_models]
        for spec in dense_to_build:
            res = build_dense_index(
                chunks,
                spec["name"],
                output_dir,
                base_config,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                device=args.device,
            )
            manifest["dense_indexes"][spec["name"]] = res
            profile = output_dir / f"variant_dense_{res['slug']}.json"
            write_profile_manifest(base_manifest, profile, res["config_path"], database={
                "path": res["db_path"], "collection": res["collection_name"],
                "chunk_count": res["chunk_count"], "indexed": True,
            })
            res["variant_manifest"] = str(profile)

    if not args.skip_sparse:
        sparse_to_build = [s for s in SPARSE_CONFIGS if s["id"] in args.sparse_ids]
        for spec in sparse_to_build:
            res = build_sparse_index(chunks, spec, output_dir, base_config, device=args.device)
            manifest["sparse_indexes"][spec["id"]] = res
            profile = output_dir / f"variant_sparse_{spec['id']}.json"
            write_profile_manifest(base_manifest, profile, res["config_path"], sparse_path=res["index_path"])
            res["variant_manifest"] = str(profile)

    manifest_path = output_dir / "index_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=======================================================")
    print(f"  All retrieval models indexed successfully!")
    print(f"  Manifest written to: {manifest_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
