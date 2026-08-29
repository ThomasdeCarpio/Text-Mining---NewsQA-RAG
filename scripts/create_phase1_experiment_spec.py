#!/usr/bin/env python3
"""Create validated experiment YAMLs for each stage of the Phase 1 tournament."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def profile(value: str) -> tuple[str, dict]:
    parts = value.split(",", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("profile must be NAME,RETRIEVER,CONFIG,MANIFEST")
    name, retriever, config, manifest = parts
    return name, {"retriever": retriever, "config": config, "variant_manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["round1", "round2", "round3", "final"], required=True)
    parser.add_argument("--profile", action="append", type=profile, required=True)
    parser.add_argument("--testset", required=True)
    parser.add_argument("--resolved-testset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs-output-dir", default="outputs/experiments")
    parser.add_argument("--variant", default="original")
    parser.add_argument("--experiment-id")
    parser.add_argument("--shared-retrieval-cache")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--final-reranker", choices=["noop", "cross-encoder"], default="cross-encoder")
    args = parser.parse_args()

    profiles = dict(args.profile)
    testsets = {args.variant: args.testset}
    if args.resolved_testset:
        testsets["resolved"] = args.resolved_testset
    variants = list(testsets)
    indexes = {
        name: {
            "config": value["config"],
            "variant_manifest": value["variant_manifest"],
            "testsets": testsets,
        }
        for name, value in profiles.items()
    }
    common = {
        "retrieval_only": True,
        "top_k": 10 if args.stage == "round1" else 20,
        "rerank_top_n": 5,
    }
    runs = []
    if args.stage == "round1":
        for name, value in profiles.items():
            for variant in variants:
                runs.append({"index": name, "variant": variant, "retriever": value["retriever"], "reranker": "noop"})
    elif args.stage == "round2":
        rerankers = [
            ("noop", None),
            ("cross-encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            ("cross-encoder", "BAAI/bge-reranker-large"),
        ]
        for name, value in profiles.items():
            for reranker, model in rerankers:
                for variant in variants:
                    row = {"index": name, "variant": variant, "retriever": value["retriever"], "reranker": reranker}
                    if model:
                        row["reranker_model"] = model
                    runs.append(row)
    elif args.stage == "round3":
        for name, value in profiles.items():
            for variant in variants:
                runs.extend([
                    {"index": name, "variant": variant, "retriever": value["retriever"], "reranker": "noop"},
                    {"index": name, "variant": variant, "retriever": value["retriever"], "reranker": "cross-encoder", "reranker_model": args.reranker_model},
                ])
    else:
        if len(profiles) != 1:
            raise SystemExit("final stage requires exactly one locked profile")
        name, value = next(iter(profiles.items()))
        for variant in variants:
            row = {"index": name, "variant": variant, "retriever": value["retriever"], "reranker": args.final_reranker}
            if args.final_reranker == "cross-encoder":
                row["reranker_model"] = args.reranker_model
            runs.append(row)

    spec = {
        "schema_version": 1,
        "experiment": {"id": args.experiment_id or f"phase1-{args.stage}", "name": f"Phase 1 {args.stage}"},
        "output_dir": args.runs_output_dir,
        "seed": 42,
        "dataset": {"article_field": "article_key", "development_articles": 50, "indexes": indexes},
        "fixed": {**common, "partition": "final_test" if args.stage == "final" else "development"},
        "runs": runs,
        "runtime": {"max_attempts": 2, "progress": True, "shared_retrieval_cache": args.shared_retrieval_cache},
        "judge": {"enabled": False},
        "summary": {
            "metrics": ["retrieval.hit_rate@1", "retrieval.hit_rate@5", "retrieval.mrr@5", "retrieval.ndcg@5", "retrieval.recall@5"],
            "paired_metric": "retrieval.mrr@5",
            "quality_metric": "retrieval.mrr@5.mean",
            "latency_metric": "latency.total.p50_ms",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
