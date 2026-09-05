#!/usr/bin/env python3
"""Standalone, high-performance evaluator for Phase 1 Retrieval Tournament."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from newsqa_rag.indexing.bm25_index import BM25Index
from newsqa_rag.indexing.chroma_store import ChromaStore
from newsqa_rag.indexing.embeddings import get_embedding_function
from newsqa_rag.retrieval.dense import DenseRetriever
from newsqa_rag.retrieval.hybrid import BM25Retriever, HybridRetriever
from newsqa_rag.retrieval.reranker import CrossEncoderReranker, NoOpReranker


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--testset-path",
        default=str(PROJECT_ROOT / "data/evaluation/newsqa_200_11064/final_deduplicated/testset_reviewed_original.jsonl"),
        help="Path to reviewed testset JSONL.",
    )
    parser.add_argument(
        "--chunks-path",
        default=str(PROJECT_ROOT / "data/evaluation/newsqa_200_11064/final_deduplicated/chunks.jsonl"),
        help="Path to chunks JSONL.",
    )
    parser.add_argument(
        "--retriever-type",
        choices=["dense", "bm25", "hybrid"],
        required=True,
        help="Retrieval strategy: 'dense', 'bm25', 'hybrid'.",
    )
    parser.add_argument(
        "--bm25-path",
        default=None,
        help="Path to BM25 pickle file.",
    )
    parser.add_argument(
        "--db-path",
        default=str(PROJECT_ROOT / "data/chroma_db"),
        help="ChromaDB directory path.",
    )
    parser.add_argument(
        "--collection-name",
        default="newsqa_chunks_canonical_512_64",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name for dense search.",
    )
    parser.add_argument(
        "--reranker-type",
        choices=["noop", "cross-encoder"],
        default="noop",
        help="Reranker: 'noop' or 'cross-encoder'.",
    )
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Model name for cross-encoder reranker.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Initial retrieval top-K.")
    parser.add_argument("--top-n", type=int, default=5, help="Post-reranking top-N.")
    parser.add_argument("--output-json", default=None, help="Path to save evaluation summary JSON.")
    parser.add_argument("--n-eval", type=int, default=None, help="Limit number of evaluation questions.")
    parser.add_argument("--progress", action="store_true", help="Display progress bar.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def compute_metrics(
    evaluations: list[dict],
    retrieve_times: list[float],
    rerank_times: list[float],
    total_times: list[float],
) -> dict:
    """Compute Hit@K, Recall@K, MRR@5, NDCG@5, and P50/P95 latencies."""
    n_queries = len(evaluations)
    if n_queries == 0:
        return {}

    hits_at_1 = sum(1 for e in evaluations if e["hit@1"]) / n_queries
    hits_at_3 = sum(1 for e in evaluations if e["hit@3"]) / n_queries
    hits_at_5 = sum(1 for e in evaluations if e["hit@5"]) / n_queries
    hits_at_10 = sum(1 for e in evaluations if e["hit@10"]) / n_queries

    recalls_at_5 = np.mean([e["recall@5"] for e in evaluations])
    recalls_at_10 = np.mean([e["recall@10"] for e in evaluations])
    mrr_at_5 = np.mean([e["mrr@5"] for e in evaluations])
    ndcg_at_5 = np.mean([e["ndcg@5"] for e in evaluations])

    return {
        "n_queries": n_queries,
        "hit@1": round(float(hits_at_1), 4),
        "hit@3": round(float(hits_at_3), 4),
        "hit@5": round(float(hits_at_5), 4),
        "hit@10": round(float(hits_at_10), 4),
        "recall@5": round(float(recalls_at_5), 4),
        "recall@10": round(float(recalls_at_10), 4),
        "mrr@5": round(float(mrr_at_5), 4),
        "ndcg@5": round(float(ndcg_at_5), 4),
        "latency_p50_ms": round(float(np.percentile(total_times, 50)), 2),
        "latency_p90_ms": round(float(np.percentile(total_times, 90)), 2),
        "latency_p95_ms": round(float(np.percentile(total_times, 95)), 2),
        "mean_retrieve_ms": round(float(np.mean(retrieve_times)), 2),
        "mean_rerank_ms": round(float(np.mean(rerank_times)), 2),
        "mean_total_ms": round(float(np.mean(total_times)), 2),
    }


def score_single_query(
    ranked_chunks: list[dict],
    ground_truth_chunk_ids: set[str],
) -> dict:
    """Calculate ranking scores for a single query's retrieved items."""
    retrieved_ids = [c["id"] for c in ranked_chunks]

    hit_1 = bool(retrieved_ids[:1] and set(retrieved_ids[:1]) & ground_truth_chunk_ids)
    hit_3 = bool(retrieved_ids[:3] and set(retrieved_ids[:3]) & ground_truth_chunk_ids)
    hit_5 = bool(retrieved_ids[:5] and set(retrieved_ids[:5]) & ground_truth_chunk_ids)
    hit_10 = bool(retrieved_ids[:10] and set(retrieved_ids[:10]) & ground_truth_chunk_ids)

    # Recall
    found_5 = len(set(retrieved_ids[:5]) & ground_truth_chunk_ids)
    found_10 = len(set(retrieved_ids[:10]) & ground_truth_chunk_ids)
    total_gt = max(1, len(ground_truth_chunk_ids))
    recall_5 = found_5 / total_gt
    recall_10 = found_10 / total_gt

    # MRR@5
    mrr_5 = 0.0
    for rank, chunk_id in enumerate(retrieved_ids[:5], start=1):
        if chunk_id in ground_truth_chunk_ids:
            mrr_5 = 1.0 / rank
            break

    # NDCG@5
    dcg_5 = 0.0
    for rank, chunk_id in enumerate(retrieved_ids[:5], start=1):
        if chunk_id in ground_truth_chunk_ids:
            dcg_5 += 1.0 / math.log2(rank + 1)

    idcg_5 = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(ground_truth_chunk_ids), 5) + 1))
    ndcg_5 = (dcg_5 / idcg_5) if idcg_5 > 0 else 0.0

    return {
        "hit@1": hit_1,
        "hit@3": hit_3,
        "hit@5": hit_5,
        "hit@10": hit_10,
        "recall@5": recall_5,
        "recall@10": recall_10,
        "mrr@5": mrr_5,
        "ndcg@5": ndcg_5,
    }


def main() -> None:
    args = parse_args()

    testset_path = Path(args.testset_path).resolve()
    chunks_path = Path(args.chunks_path).resolve()

    testset = load_jsonl(testset_path)
    testset = [item for item in testset if item.get("relevant_chunk_ids")]
    if args.n_eval:
        testset = testset[: args.n_eval]

    chunks = load_jsonl(chunks_path)
    chunk_lookup = {c["id"]: c for c in chunks}

    # Initialize Embedding / Store
    emb_cfg = {"embedding": {"provider": "sentence-transformers", "model_name": args.embedding_model}}
    emb_fn = get_embedding_function(emb_cfg)
    store = ChromaStore(args.db_path, emb_fn)

    # Initialize Retriever
    if args.retriever_type == "dense":
        retriever = DenseRetriever(store, args.collection_name)
    elif args.retriever_type == "bm25":
        if args.bm25_path and Path(args.bm25_path).exists():
            bm25_idx = BM25Index.load(args.bm25_path)
        else:
            bm25_idx = BM25Index()
            bm25_idx.build(chunks)
        retriever = BM25Retriever(bm25_idx, chunk_lookup)
    elif args.retriever_type == "hybrid":
        dense = DenseRetriever(store, args.collection_name)
        if args.bm25_path and Path(args.bm25_path).exists():
            bm25_idx = BM25Index.load(args.bm25_path)
        else:
            bm25_idx = BM25Index()
            bm25_idx.build(chunks)
        bm25 = BM25Retriever(bm25_idx, chunk_lookup)
        retriever = HybridRetriever(dense=dense, sparse=bm25, dense_weight=0.7, sparse_weight=0.3)

    # Initialize Reranker
    if args.reranker_type == "cross-encoder":
        reranker = CrossEncoderReranker(model_name=args.reranker_model)
    else:
        reranker = NoOpReranker()

    evaluations = []
    retrieve_times = []
    rerank_times = []
    total_times = []

    iterable = testset
    if args.progress:
        from tqdm import tqdm
        iterable = tqdm(testset, desc="Evaluating", unit="q")

    for item in iterable:
        question = item["question"]
        gt_chunk_ids = set(item["relevant_chunk_ids"])

        t0 = time.perf_counter()
        retrieved = retriever.retrieve(question, top_k=args.top_k)
        t1 = time.perf_counter()

        reranked = reranker.rerank(question, retrieved, top_n=args.top_n)
        t2 = time.perf_counter()

        ret_ms = (t1 - t0) * 1000
        rerank_ms = (t2 - t1) * 1000
        tot_ms = (t2 - t0) * 1000

        retrieve_times.append(ret_ms)
        rerank_times.append(rerank_ms)
        total_times.append(tot_ms)

        query_score = score_single_query(reranked, gt_chunk_ids)
        evaluations.append(query_score)

    metrics = compute_metrics(evaluations, retrieve_times, rerank_times, total_times)
    metrics["retriever_type"] = args.retriever_type
    metrics["embedding_model"] = args.embedding_model if args.retriever_type in ("dense", "hybrid") else "N/A"
    metrics["bm25_path"] = args.bm25_path if args.retriever_type in ("bm25", "hybrid") else "N/A"
    metrics["reranker_type"] = args.reranker_type
    metrics["reranker_model"] = args.reranker_model if args.reranker_type == "cross-encoder" else "noop"
    metrics["top_k"] = args.top_k
    metrics["top_n"] = args.top_n

    print("\n=======================================================")
    print(f"  EVALUATION SUMMARY: {args.retriever_type.upper()} + {args.reranker_type.upper()}")
    print("=======================================================")
    print(f"  Queries Evaluated : {metrics['n_queries']}")
    print(f"  MRR@5             : {metrics['mrr@5']:.4f}")
    print(f"  NDCG@5            : {metrics['ndcg@5']:.4f}")
    print(f"  Hit@1             : {metrics['hit@1'] * 100:.2f}%")
    print(f"  Hit@5             : {metrics['hit@5'] * 100:.2f}%")
    print(f"  Recall@5          : {metrics['recall@5'] * 100:.2f}%")
    print(f"  P50 Latency       : {metrics['latency_p50_ms']:.2f} ms")
    print(f"  P95 Latency       : {metrics['latency_p95_ms']:.2f} ms")
    print(f"  Retrieve / Rerank : {metrics['mean_retrieve_ms']:.2f} ms / {metrics['mean_rerank_ms']:.2f} ms")
    print("=======================================================\n")

    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
