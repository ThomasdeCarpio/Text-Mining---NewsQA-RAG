"""
Evaluate the RAG pipeline against ground-truth QA pairs using Ragas metrics.

Usage:
    python scripts/run_benchmark.py \
        --db-path database/ \
        --collection basic_collection \
        --test-set data/test_qa.json \
        --output results/benchmark_results.json \
        --config configs/config.yaml

Args:
    --db-path       Path to ChromaDB persistent storage
    --collection    Collection name to query against
    --test-set      Path to test QA pairs JSON file
    --output        Path to save evaluation results (default: prints to stdout)
    --config        Path to config.yaml (default: configs/config.yaml)

Test set format (JSON):
    [
        {
            "question": "What caused the market crash?",
            "ground_truth": "The market crashed due to...",
            "contexts": ["optional list of ground-truth context passages"]
        },
        ...
    ]

Output format (JSON):
    {
        "metrics": {
            "faithfulness": 0.85,
            "answer_relevancy": 0.92,
            "context_precision": 0.78,
            "context_recall": 0.81
        },
        "per_question": [ ... ],
        "config": { ... },
        "timestamp": "2024-..."
    }

Pipeline steps:
    1. Load test QA pairs from JSON
    2. For each question: retrieve context via RAG pipeline (src.retrieval)
    3. For each question: generate answer via LLM (src.agents or src.llm)
    4. Compute Ragas metrics over all Q/A/context triples (src.evaluation.metrics)
    5. Output results
"""
