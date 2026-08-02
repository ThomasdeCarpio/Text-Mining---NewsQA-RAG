"""Focused checks for experiment locking and comparison logic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from newsqa_rag.experiments import (
    bootstrap_ci,
    build_article_partitions,
    expand_run_matrix,
    paired_comparison,
    pareto_frontier,
)


class ExperimentPipelineTests(unittest.TestCase):
    def test_matrix_and_article_partitions_are_stable_and_paired(self):
        spec = {
            "schema_version": 1,
            "experiment": {"id": "test-matrix"},
            "dataset": {
                "indexes": {
                    "base": {
                        "config": "config.yaml",
                        "variant_manifest": "manifest.json",
                        "testsets": {"original": "original.jsonl", "resolved": "resolved.jsonl"},
                    }
                }
            },
            "fixed": {"index": "base", "partition": "development"},
            "matrix": {
                "variant": ["original", "resolved"],
                "retriever": ["dense", "hybrid"],
            },
        }
        runs = expand_run_matrix(spec)
        self.assertEqual(len(runs), 4)
        self.assertEqual(runs, expand_run_matrix(spec))

        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for variant in ("original", "resolved"):
                path = Path(directory) / f"{variant}.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for index in range(4):
                        handle.write(json.dumps({
                            "question_id": f"{variant}-{index}",
                            "article_key": f"article-{index}",
                            "relevant_chunk_ids": [f"chunk-{index}"],
                        }) + "\n")
                paths[variant] = path

            partitions = build_article_partitions(paths, development_articles=2, seed=7)

        development = set(partitions["partitions"]["development"]["article_ids"])
        final_test = set(partitions["partitions"]["final_test"]["article_ids"])
        self.assertFalse(development & final_test)
        self.assertEqual(len(development), 2)
        self.assertEqual(
            len(partitions["partitions"]["development"]["question_ids"]["original"]),
            len(partitions["partitions"]["development"]["question_ids"]["resolved"]),
        )

    def test_bootstrap_paired_delta_and_pareto_are_deterministic(self):
        self.assertEqual(bootstrap_ci([1.0], samples=10), (1.0, 1.0))
        left = [
            {"source_question_id": "q1", "retrieval": {"mrr@5": 0.0}},
            {"source_question_id": "q2", "retrieval": {"mrr@5": 0.5}},
        ]
        right = [
            {"source_question_id": "q1", "retrieval": {"mrr@5": 1.0}},
            {"source_question_id": "q2", "retrieval": {"mrr@5": 0.5}},
        ]
        comparison = paired_comparison(left, right, "retrieval.mrr@5", samples=100, seed=1)
        self.assertEqual(comparison["mean_delta_right_minus_left"], 0.5)
        rows = [
            {"run_id": "fast", "quality": 0.8, "latency": 10},
            {"run_id": "slow", "quality": 0.8, "latency": 20},
            {"run_id": "best", "quality": 0.9, "latency": 15},
        ]
        self.assertEqual(pareto_frontier(rows, "quality", "latency"), ["fast", "best"])


if __name__ == "__main__":
    unittest.main()
