"""
Evaluate the RAG pipeline against the prepared NewsQA test set.

Runs retrieval (and optionally generation + RAGAS) and writes a JSON report.

Usage:
    # Retrieval-only benchmark (fast, no API key needed for non-OpenAI embeddings)
    python scripts/run_benchmark.py \\
        --retriever dense \\
        --testset data/testset_1000.jsonl \\
        --n-eval 200 \\
        --report-dir reports/dense/

    # Hybrid retrieval
    python scripts/run_benchmark.py \\
        --retriever hybrid \\
        --chunks-path database/chroma/chunks/basic_collection.jsonl \\
        --bm25-path   database/chroma/bm25/basic_collection.pkl \\
        --testset data/testset_1000.jsonl \\
        --report-dir reports/hybrid/

    # Full pipeline including LLM generation + RAGAS (requires OPENAI_API_KEY)
    python scripts/run_benchmark.py \\
        --retriever dense --run-generator --run-ragas \\
        --testset data/testset_1000.jsonl \\
        --report-dir reports/dense_full/

Args:
    --retriever       dense | bm25 | hybrid  (default: dense)
    --reranker        noop  (default: noop)
    --testset         Path to JSONL test set from prepare_testset.py
    --n-eval          Max number of questions to evaluate (default: all)
    --top-k           Retriever top-k (default: from config)
    --rerank-top-n    Reranker top-n (default: from config)
    --db-path         ChromaDB path (default: database/chroma/)
    --collection      Collection name (default: basic_collection)
    --chunks-path     JSONL chunks path (needed for bm25/hybrid)
    --bm25-path       BM25 pickle path (optional; built from chunks if missing)
    --config          Config file (default: configs/config.yaml)
    --report-dir      Directory to write report.json + report_summary.txt
    --run-generator   Also run LLM generation and compute EM/F1
    --run-ragas       Also compute RAGAS metrics (requires --run-generator + OPENAI_API_KEY)
"""

import argparse
import json
import os
import sys
import random
import yaml
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexing.embeddings import get_embedding_function
from src.indexing.chroma_store import ChromaStore
from src.ingestion.chunker import load_chunks
from src.retrieval.retriever_factory import get_retriever
from src.retrieval.reranker import get_reranker
from src.agents.rag_agent import RAGAgent
from src.evaluation.testset import load_testset
from src.evaluation.metrics import (
    evaluate_retrieval,
    evaluate_qa,
    evaluate_ragas,
    build_report,
)


def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline benchmark.")
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="dense")
    parser.add_argument("--reranker", choices=["noop"], default="noop")
    parser.add_argument("--testset", required=True, help="JSONL test set path")
    parser.add_argument("--n-eval", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--top-k", type=int, default=None, help="Retriever top-k")
    parser.add_argument("--rerank-top-n", type=int, default=None, help="Reranker top-n")
    parser.add_argument("--db-path", default="database/chroma/")
    parser.add_argument("--collection", default="basic_collection")
    parser.add_argument("--chunks-path", default=None, help="JSONL chunks for bm25/hybrid")
    parser.add_argument("--bm25-path", default=None, help="BM25 pickle path")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--run-generator", action="store_true",
                        help="Generate answers with LLM and compute EM/F1")
    parser.add_argument("--run-ragas", action="store_true",
                        help="Compute RAGAS metrics (requires --run-generator + OPENAI_API_KEY)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", action="store_true",
                        help="Show a tqdm progress bar over questions")
    args = parser.parse_args()

    # Load .env so OPENAI_API_KEY / DEEPSEEK_API_KEY are available to generation + RAGAS.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    retrieval_cfg = config.get("retrieval", {})
    top_k = args.top_k or retrieval_cfg.get("top_k", 10)
    rerank_top_n = args.rerank_top_n or retrieval_cfg.get("reranker", {}).get("top_n", 5)
    bm25_path = args.bm25_path or os.path.join(args.db_path, "bm25", f"{args.collection}.pkl")

    # ------------------------------------------------------------------
    # Load test set
    # ------------------------------------------------------------------
    print(f"Loading test set from {args.testset} ...")
    test_entries = load_testset(args.testset)

    # Validate the data contract early with actionable messages (see docs/evaluation.md §6.1)
    if not test_entries:
        sys.exit(f"ERROR: test set '{args.testset}' is empty.")
    required = ["question", "ground_truth", "relevant_chunk_ids"]
    missing = [f for f in required if f not in test_entries[0]]
    if missing:
        sys.exit(
            f"ERROR: test set rows are missing required field(s): {missing}.\n"
            f"       Expected schema per line: {required} (+ optional fields).\n"
            f"       See docs/evaluation.md section 6.1."
        )

    # Filter to entries that have relevant chunk IDs (for retrieval metrics)
    scorable = [e for e in test_entries if e.get("relevant_chunk_ids")]
    print(f"  Total entries    : {len(test_entries)}")
    print(f"  With ground truth: {len(scorable)}")

    if not scorable:
        sys.exit(
            "ERROR: no rows have a non-empty 'relevant_chunk_ids', so retrieval cannot be scored.\n"
            "       This field must be engineered by mapping answer spans to chunk IDs\n"
            "       (see docs/evaluation.md section 6.1 and src/evaluation/testset.py)."
        )

    if args.n_eval:
        rng = random.Random(args.seed)
        scorable = rng.sample(scorable, min(args.n_eval, len(scorable)))
        print(f"  Evaluating       : {len(scorable)}")

    # ------------------------------------------------------------------
    # Set up pipeline
    # ------------------------------------------------------------------
    print(f"\nSetting up retriever: {args.retriever} ...")
    ef = get_embedding_function(config)
    store = ChromaStore(args.db_path, ef)

    chunks = None
    if args.retriever in ("bm25", "hybrid"):
        chunks_path = args.chunks_path or os.path.join(
            args.db_path, "chunks", f"{args.collection}.jsonl"
        )
        print(f"Loading chunks from {chunks_path} ...")
        chunks = load_chunks(chunks_path)

    retriever = get_retriever(
        retriever_type=args.retriever,
        config=config,
        store=store,
        collection_name=args.collection,
        chunks=chunks,
        bm25_path=bm25_path if args.retriever in ("bm25", "hybrid") else None,
    )

    reranker = get_reranker(config)

    llm = None
    if args.run_generator:
        from src.llm import get_llm
        llm = get_llm(config)

    agent = RAGAgent(
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
    )

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    print(f"\nRunning evaluation on {len(scorable)} questions ...")

    retrieval_samples = []
    qa_samples = []
    ragas_samples = []
    failures = []

    iterable = scorable
    if args.progress:
        from tqdm import tqdm
        iterable = tqdm(scorable, desc="Retrieval+gen", unit="q")

    for i, entry in enumerate(iterable, 1):
        question = entry["question"]

        if args.run_generator and llm is not None:
            result = agent.run(question)
            answer = result["answer"]
            retrieved_ids = [r["id"] for r in result["reranked_chunks"]]
            contexts = result["contexts"]
        else:
            result = agent.run_retrieval_only(question)
            retrieved_ids = result["retrieved_ids"]
            answer = ""
            contexts = [r["text"] for r in result["reranked_chunks"]]

        retrieval_samples.append({
            "relevant_chunk_ids": entry["relevant_chunk_ids"],
            "retrieved_ids": retrieved_ids,
        })

        # Record retrieval misses for the dashboard's Failure Analysis table (cap at 20)
        relevant = set(entry["relevant_chunk_ids"])
        if len(failures) < 20 and not any(rid in relevant for rid in retrieved_ids[:top_k]):
            failures.append({
                "question": question,
                "expected": entry["ground_truth"],
                "retrieved": (contexts[0][:200] if contexts else "No matching chunk found"),
                "reason": "No ground-truth chunk in top-k (retrieval miss)",
            })

        if args.run_generator:
            qa_samples.append({
                "prediction": answer,
                "ground_truth": entry["ground_truth"],
            })
            if args.run_ragas:
                ragas_samples.append({
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": entry["ground_truth"],
                })

        if not args.progress and i % 10 == 0:
            print(f"  {i}/{len(scorable)} done ...")

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    print("\nComputing metrics ...")
    retrieval_metrics = evaluate_retrieval(retrieval_samples)

    # Diagnose the classic silent failure: metrics all 0 because the test set's
    # relevant_chunk_ids don't belong to this collection (different articles/chunker/ID scheme).
    if retrieval_metrics.get("n_samples", 0) > 0 and all(
        v == 0 for k, v in retrieval_metrics.items() if k.startswith("hit_rate@")
    ):
        print(
            f"\n  WARNING: every retrieval metric is 0 across {retrieval_metrics['n_samples']} samples.\n"
            f"           The test set's 'relevant_chunk_ids' almost certainly do NOT match the chunk IDs\n"
            f"           in collection '{args.collection}'. Build the test set and the collection from the\n"
            f"           SAME articles + chunker (see scripts/build_mini_testset.py --build-collection)."
        )

    qa_metrics = evaluate_qa(qa_samples) if qa_samples else {}

    ragas_metrics = {}
    if ragas_samples:
        print(f"Running RAGAS judge on {len(ragas_samples)} samples (LLM calls, this can take a while) ...")
        ragas_cfg = config.get("evaluation", {})
        ragas_metrics = evaluate_ragas(
            ragas_samples,
            metrics=ragas_cfg.get("metrics"),
            llm_model=ragas_cfg.get("llm", "gpt-4o-mini"),
        )

    config_snapshot = {
        "retriever": args.retriever,
        "reranker": args.reranker,
        "top_k": top_k,
        "rerank_top_n": rerank_top_n,
        "collection": args.collection,
        "embedding": config.get("embedding", {}),
        "chunking": config.get("chunking", {}),
        "run_generator": args.run_generator,
        "run_ragas": args.run_ragas,
        "n_eval": len(scorable),
        "timestamp": datetime.now().isoformat(),
    }

    report = build_report(
        config_snapshot=config_snapshot,
        retrieval_metrics=retrieval_metrics,
        qa_metrics=qa_metrics or None,
        ragas_metrics=ragas_metrics or None,
    )
    report["failures"] = failures

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    os.makedirs(args.report_dir, exist_ok=True)
    report_path = os.path.join(args.report_dir, "report.json")
    summary_path = os.path.join(args.report_dir, "report_summary.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _write_summary(summary_path, report)

    print(f"\nReport saved to {report_path}")
    print(f"Summary  saved to {summary_path}")
    _print_summary(report)


def _write_summary(path: str, report: dict) -> None:
    lines = []
    cfg = report.get("config", {})
    lines.append(f"=== Benchmark Report ===")
    lines.append(f"Timestamp  : {cfg.get('timestamp', '')}")
    lines.append(f"Retriever  : {cfg.get('retriever')} | Reranker: {cfg.get('reranker')}")
    lines.append(f"top_k={cfg.get('top_k')} | rerank_top_n={cfg.get('rerank_top_n')} | n_eval={cfg.get('n_eval')}")
    lines.append(f"Collection : {cfg.get('collection')}")
    lines.append(f"Embedding  : {cfg.get('embedding', {}).get('provider')} / {cfg.get('embedding', {}).get('model_name')}")
    lines.append("")

    if "retrieval" in report:
        lines.append("--- Retrieval ---")
        for k, v in report["retrieval"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    if "qa" in report:
        lines.append("--- QA ---")
        for k, v in report["qa"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    if "ragas" in report:
        lines.append("--- RAGAS ---")
        for k, v in report["ragas"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _print_summary(report: dict) -> None:
    print("\n=== Summary ===")
    if "retrieval" in report:
        r = report["retrieval"]
        print(f"  hit_rate@5={r.get('hit_rate@5', '-')}  mrr@5={r.get('mrr@5', '-')}  "
              f"recall@5={r.get('recall@5', '-')}  ndcg@5={r.get('ndcg@5', '-')}")
    if "qa" in report:
        q = report["qa"]
        print(f"  EM={q.get('exact_match', '-')}  F1={q.get('f1', '-')}")
    if "ragas" in report:
        for k, v in report["ragas"].items():
            print(f"  {k}={v}")


if __name__ == "__main__":
    main()
