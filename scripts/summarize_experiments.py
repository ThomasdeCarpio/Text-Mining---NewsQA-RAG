#!/usr/bin/env python3
"""Summarize completed experiment runs."""

import argparse

from newsqa_rag.experiments import summarize_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = summarize_experiment(
        args.experiment_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(f"Summarized {len(result['runs'])} runs")


if __name__ == "__main__":
    main()
