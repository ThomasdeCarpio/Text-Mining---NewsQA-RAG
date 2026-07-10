"""
Download and prepare the NewsQA evaluation test set.

Downloads the NewsQA dataset from HuggingFace, samples N articles, chunks them
using the configured chunker, maps each answer span to chunk IDs, and saves a
JSONL file for use by run_benchmark.py.

Usage:
    python scripts/prepare_testset.py \\
        --n-articles 1000 \\
        --output data/testset_1000.jsonl \\
        --config configs/config.yaml

Args:
    --n-articles    Number of unique articles (contexts) to sample (default: 1000)
    --output        Output JSONL path (default: data/testset.jsonl)
    --config        Path to config.yaml (default: configs/config.yaml)
    --split         HuggingFace dataset split: train/validation/test (default: train)
    --seed          Random seed for reproducible sampling (default: 42)
    --overlap-thr   Fuzzy word overlap threshold for chunk mapping (default: 0.6)
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.chunker import get_chunker
from src.evaluation.testset import NewsQATestSetBuilder, load_testset


def main():
    parser = argparse.ArgumentParser(description="Prepare NewsQA evaluation test set.")
    parser.add_argument("--n-articles", type=int, default=1000,
                        help="Number of unique articles to sample")
    parser.add_argument("--output", default="data/testset.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Config file path")
    parser.add_argument("--split", default="train",
                        choices=["train", "validation", "test"],
                        help="HuggingFace dataset split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for article sampling")
    parser.add_argument("--overlap-thr", type=float, default=0.6,
                        help="Fuzzy word overlap threshold for chunk mapping")
    parser.add_argument("--dataset", default="lucadiliello/newsqa",
                        help="HuggingFace dataset name")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    chunker = get_chunker(config)

    builder = NewsQATestSetBuilder(
        chunker=chunker,
        overlap_threshold=args.overlap_thr,
        seed=args.seed,
    )

    entries = builder.build(
        n_articles=args.n_articles,
        output_path=args.output,
        split=args.split,
        dataset_name=args.dataset,
    )

    # Quick summary
    print(f"\nSummary:")
    print(f"  Articles sampled : {args.n_articles}")
    print(f"  Total questions  : {len(entries)}")
    n_with = sum(1 for e in entries if e["relevant_chunk_ids"])
    print(f"  With chunk match : {n_with} ({100*n_with//max(len(entries),1)}%)")
    print(f"  Output           : {args.output}")


if __name__ == "__main__":
    main()
