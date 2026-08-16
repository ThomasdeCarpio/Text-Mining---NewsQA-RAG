#!/usr/bin/env python3
"""Batch build evaluation datasets across multiple chunking sizes, overlaps, and strategies for ablation studies."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from newsqa_rag.evaluation.testset import DatasetBuildError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.getenv("NEWSQA_EVAL_REPO_ID"))
    parser.add_argument("--revision", default="v1.0.0")
    parser.add_argument("--local-bundle", help="Use a local exported bundle instead of downloading from HF")
    parser.add_argument(
        "--base-config",
        default=str(PROJECT_ROOT / "configs/config.yaml"),
        help="Base configuration YAML to clone parameters from",
    )
    parser.add_argument(
        "--output-base",
        default=str(PROJECT_ROOT / "data/evaluation/ablation"),
        help="Base output directory for materialized ablation datasets",
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[256, 512, 1024],
        help="List of chunk sizes to test in ablation study",
    )
    parser.add_argument(
        "--chunk-overlaps",
        nargs="+",
        type=int,
        default=None,
        help="List of chunk overlaps (defaults to 12.5%% of each chunk size)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["recursive"],
        help="List of chunking strategies to test (e.g. recursive, sentence, fixed)",
    )
    parser.add_argument("--db-path-base", default=str(PROJECT_ROOT / "data/chroma_db"))
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN_READ_ONLY"),
        help="Hugging Face access token (or set HF_TOKEN / HF_TOKEN_READ_ONLY env var)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-index", action="store_true", help="Skip building Chroma/BM25 indexes")
    return parser


def build_ablation_datasets(args: argparse.Namespace) -> dict:
    output_base = Path(args.output_base).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    base_config_path = Path(args.base_config).resolve()
    with base_config_path.open(encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    summary_records = []
    
    for strategy in args.strategies:
        for size in args.chunk_sizes:
            # Determine overlap: default to size // 8 if not specified
            if args.chunk_overlaps:
                overlaps = args.chunk_overlaps
            else:
                overlaps = [max(16, size // 8)]
            
            for overlap in overlaps:
                variant_id = f"chunk_{size}_{overlap}_{strategy}"
                variant_dir = output_base / variant_id
                config_dir = output_base / "configs"
                config_dir.mkdir(parents=True, exist_ok=True)
                variant_config_path = config_dir / f"{variant_id}.yaml"
                db_path = Path(f"{args.db_path_base}_{size}_{overlap}_{strategy}").resolve()

                # Build mutated config
                variant_config = yaml.safe_load(yaml.dump(base_config))
                variant_config.setdefault("chunking", {})
                variant_config["chunking"]["chunk_size"] = size
                variant_config["chunking"]["chunk_overlap"] = overlap
                variant_config["chunking"]["strategy"] = strategy

                with variant_config_path.open("w", encoding="utf-8") as handle:
                    yaml.dump(variant_config, handle, sort_keys=False)

                print(f"\n==================================================")
                print(f"Materializing Ablation Variant: {variant_id}")
                print(f"Config: {variant_config_path}")
                print(f"Output Root: {variant_dir}")
                print(f"==================================================\n")

                cmd = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/materialize_evaluation_dataset.py"),
                    "--revision", args.revision,
                    "--config", str(variant_config_path),
                    "--output-root", str(variant_dir),
                    "--db-path", str(db_path),
                ]

                if args.local_bundle:
                    cmd.extend(["--local-bundle", args.local_bundle])
                elif args.repo_id:
                    cmd.extend(["--repo-id", args.repo_id])
                else:
                    raise DatasetBuildError("Either --repo-id or --local-bundle is required")

                if args.token:
                    cmd.extend(["--token", args.token])
                if args.overwrite:
                    cmd.append("--overwrite")
                if args.skip_index:
                    cmd.extend(["--skip-index", "--no-deduplicate"])

                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join(
                    v for v in [str(PROJECT_ROOT / "backend"), env.get("PYTHONPATH", "")] if v
                )

                proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
                if proc.returncode != 0:
                    print(f"ERROR materializing {variant_id}:\n{proc.stderr}", file=sys.stderr)
                    continue

                # Parse materialization stdout JSON
                try:
                    result = json.loads(proc.stdout)
                except Exception:
                    result = {"raw_output": proc.stdout}

                # Read integrity report if available
                integrity_path = variant_dir / "final_deduplicated/integrity_report.json"
                if not integrity_path.exists():
                    integrity_path = variant_dir / "final/integrity_report.json"
                integrity_data = {}
                if integrity_path.exists():
                    integrity_data = json.loads(integrity_path.read_text(encoding="utf-8"))

                summary_records.append({
                    "variant_id": variant_id,
                    "chunk_size": size,
                    "chunk_overlap": overlap,
                    "strategy": strategy,
                    "output_root": str(variant_dir),
                    "db_path": str(db_path),
                    "chunks": integrity_data.get("chunks"),
                    "deduplicated_questions": integrity_data.get("deduplicated_questions"),
                    "dataset_sha256": result.get("dataset_sha256"),
                    "index_fingerprint": result.get("index_fingerprint"),
                })

    summary_file = output_base / "ablation_summary.json"
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary_records, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"\nCompleted ablation dataset materialization for {len(summary_records)} variants.")
    print(f"Summary saved to: {summary_file}")
    return {"summary_file": str(summary_file), "variants": summary_records}


def main() -> int:
    try:
        build_ablation_datasets(build_parser().parse_args())
        return 0
    except DatasetBuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
