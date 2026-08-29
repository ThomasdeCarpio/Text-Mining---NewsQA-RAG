#!/usr/bin/env python3
"""Compose one manifest-validated Phase 1 dense/sparse/hybrid retrieval profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from newsqa_rag.evaluation.benchmark_io import stable_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-manifest", required=True)
    parser.add_argument("--sparse-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--dense-weight", type=float, default=0.7)
    parser.add_argument("--sparse-weight", type=float, default=0.3)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    dense_manifest = json.loads(Path(args.dense_manifest).read_text(encoding="utf-8"))
    sparse_manifest = json.loads(Path(args.sparse_manifest).read_text(encoding="utf-8"))
    dense_config = yaml.safe_load(Path(dense_manifest["pipeline"]["config_path"]).read_text(encoding="utf-8"))
    sparse_config = yaml.safe_load(Path(sparse_manifest["pipeline"]["config_path"]).read_text(encoding="utf-8"))
    if dense_manifest["artifacts"]["chunks"]["sha256"] != sparse_manifest["artifacts"]["chunks"]["sha256"]:
        raise SystemExit("Dense and sparse profiles were built from different chunk corpora")

    dense_config.setdefault("retrieval", {})["sparse"] = sparse_config["retrieval"]["sparse"]
    dense_config["retrieval"]["hybrid"] = {
        "enabled": True,
        "dense_weight": args.dense_weight,
        "sparse_weight": args.sparse_weight,
        "rrf_k": args.rrf_k,
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"config_{args.profile_id}.yaml"
    config_path.write_text(yaml.safe_dump(dense_config, sort_keys=False), encoding="utf-8")

    dense_manifest["pipeline"].update({
        "config_path": str(config_path),
        "config_sha256": stable_hash(dense_config),
        "embedding": dense_config["embedding"],
    })
    dense_manifest["artifacts"]["bm25"] = sparse_manifest["artifacts"]["bm25"]
    manifest_path = output_dir / f"variant_{args.profile_id}.json"
    manifest_path.write_text(json.dumps(dense_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"config": str(config_path), "variant_manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
