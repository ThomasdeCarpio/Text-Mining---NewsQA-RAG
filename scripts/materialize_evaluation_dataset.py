#!/usr/bin/env python3
"""Download a private canonical evaluation release and build local RAG artifacts."""

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

from newsqa_rag.evaluation.cloud_dataset import (
    index_fingerprint,
    local_materialized_paths,
    materialize_canonical_source,
    rebind_dedup_approval,
    verify_canonical_bundle,
)
from newsqa_rag.evaluation.testset import DatasetBuildError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.getenv("NEWSQA_EVAL_REPO_ID"))
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output-root", default=str(PROJECT_ROOT / "data/evaluation/newsqa_200_11064")
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/config.yaml"))
    parser.add_argument("--db-path", default=str(PROJECT_ROOT / "data/chroma_db"))
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN_READ_ONLY"),
        help="Hugging Face access token (or set HF_TOKEN / HF_TOKEN_READ_ONLY env var)",
    )
    parser.add_argument("--local-bundle", help="Use an exported bundle without network access")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument(
        "--skip-vector-index",
        action="store_true",
        help="Skip the baseline Chroma build but retain the BM25 artifact required by deduplication",
    )
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _run(command: list[str]) -> None:
    environment = os.environ.copy()
    backend = str(PROJECT_ROOT / "backend")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (backend, environment.get("PYTHONPATH", "")) if value
    )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=environment)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _augment_variant_manifest(
    path: Path,
    *,
    repo_id: str,
    revision: str,
    commit_sha: str,
    dataset_sha256: str,
    fingerprint: str,
) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value["cloud_source"] = {
        "provider": "huggingface",
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_commit": commit_sha,
        "dataset_sha256": dataset_sha256,
        "index_fingerprint": fingerprint,
    }
    _write_json(path, value)


def _download(args: argparse.Namespace) -> tuple[Path, str]:
    if args.local_bundle:
        bundle = Path(args.local_bundle).resolve()
        return bundle, f"local:{verify_canonical_bundle(bundle)['dataset_sha256']}"
    if not args.repo_id:
        raise DatasetBuildError("--repo-id or NEWSQA_EVAL_REPO_ID is required")
    if not args.token:
        raise DatasetBuildError("HF_TOKEN with dataset read permission is required")
    if args.revision in {"main", "master"}:
        raise DatasetBuildError("Pin --revision to an immutable version tag or commit SHA")

    from huggingface_hub import HfApi, snapshot_download

    bundle = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            cache_dir=args.cache_dir,
            token=args.token,
        )
    )
    info = HfApi(token=args.token).repo_info(
        args.repo_id, repo_type="dataset", revision=args.revision, token=args.token
    )
    return bundle, str(info.sha)


def materialize(args: argparse.Namespace) -> dict:
    if args.revision in {"main", "master"}:
        raise DatasetBuildError("Pin --revision to an immutable version tag or commit SHA")
    bundle, commit_sha = _download(args)
    cloud_manifest = verify_canonical_bundle(bundle)
    output_root = Path(args.output_root).resolve()
    paths = materialize_canonical_source(bundle, output_root, overwrite=args.overwrite)
    config_path = Path(args.config).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    fingerprint = index_fingerprint(
        dataset_sha256=cloud_manifest["dataset_sha256"],
        dataset_commit=commit_sha,
        config=config,
    )

    source_state = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    expected_outputs = [
        paths["variant_manifest"],
        output_root / "final/testset_resolved.jsonl",
        output_root / "final/chunks.jsonl",
    ]
    if args.deduplicate:
        expected_outputs.extend(
            [
                paths["dedup_variant_manifest"],
                output_root / "final_deduplicated/testset_resolved.jsonl",
            ]
        )
    if not args.overwrite and source_state.get("index_fingerprint") == fingerprint:
        if all(path.exists() for path in expected_outputs):
            result = {
                "dataset_version": cloud_manifest["dataset_version"],
                "resolved_commit": commit_sha,
                "dataset_sha256": cloud_manifest["dataset_sha256"],
                "index_fingerprint": fingerprint,
                "output_root": str(output_root),
                "variant_manifest": str(paths["variant_manifest"]),
                "deduplicated_variant_manifest": (
                    str(paths["dedup_variant_manifest"]) if args.deduplicate else None
                ),
                "resumed": True,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return result
        raise DatasetBuildError(
            "Matching materialization is incomplete; pass --overwrite to rebuild it"
        )
    if (
        not args.overwrite
        and source_state.get("index_fingerprint")
        and source_state.get("index_fingerprint") != fingerprint
        and any(path.exists() for path in expected_outputs)
    ):
        raise DatasetBuildError(
            "Output root belongs to a different index fingerprint; use a new "
            "--output-root or pass --overwrite"
        )

    baseline_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/prepare_evaluation_dataset.py"),
        "build-baseline",
        "--output-root",
        str(output_root),
        "--selection-manifest",
        str(paths["selection_manifest"]),
        "--variant-manifest",
        str(paths["variant_manifest"]),
        "--config",
        str(config_path),
        "--db-path",
        str(Path(args.db_path).resolve()),
    ]
    if args.skip_index:
        baseline_command.append("--skip-index")
    elif args.skip_vector_index:
        baseline_command.append("--skip-vector-index")
    if args.overwrite:
        baseline_command.append("--overwrite")
    _run(baseline_command)
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/prepare_evaluation_dataset.py"),
            "finalize",
            "--output-root",
            str(output_root),
            "--selection-manifest",
            str(paths["selection_manifest"]),
            "--variant-manifest",
            str(paths["variant_manifest"]),
            "--config",
            str(config_path),
        ]
    )
    _augment_variant_manifest(
        paths["variant_manifest"],
        repo_id=args.repo_id or "local-bundle",
        revision=args.revision,
        commit_sha=commit_sha,
        dataset_sha256=cloud_manifest["dataset_sha256"],
        fingerprint=fingerprint,
    )

    dedup_manifest = None
    if args.deduplicate:
        if args.skip_index:
            raise DatasetBuildError("Deduplication currently requires BM25; omit --skip-index")
        rebound_dir = output_root / "staging/dedup/rebound"
        rebound_decisions, rebound_approval = rebind_dedup_approval(
            decisions_path=paths["dedup_decisions"],
            approval_path=paths["dedup_approval"],
            resolved_testset=output_root / "final/testset_resolved.jsonl",
            output_dir=rebound_dir,
        )
        dedup_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/deduplicate_evaluation_dataset.py"),
            "--base-root",
            str(output_root / "final"),
            "--output-root",
            str(output_root / "final_deduplicated"),
            "--decisions",
            str(rebound_decisions),
            "--approval",
            str(rebound_approval),
            "--base-manifest",
            str(paths["variant_manifest"]),
            "--output-manifest",
            str(paths["dedup_variant_manifest"]),
        ]
        if args.overwrite:
            dedup_command.append("--overwrite")
        _run(dedup_command)
        _augment_variant_manifest(
            paths["dedup_variant_manifest"],
            repo_id=args.repo_id or "local-bundle",
            revision=args.revision,
            commit_sha=commit_sha,
            dataset_sha256=cloud_manifest["dataset_sha256"],
            fingerprint=fingerprint,
        )
        dedup_manifest = str(paths["dedup_variant_manifest"])

    local_source = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    local_source.update(
        {
            "repository": args.repo_id or "local-bundle",
            "requested_revision": args.revision,
            "resolved_commit": commit_sha,
            "index_fingerprint": fingerprint,
            "config_path": str(config_path),
        }
    )
    _write_json(paths["source_manifest"], local_source)
    result = {
        "dataset_version": cloud_manifest["dataset_version"],
        "resolved_commit": commit_sha,
        "dataset_sha256": cloud_manifest["dataset_sha256"],
        "index_fingerprint": fingerprint,
        "output_root": str(output_root),
        "variant_manifest": str(paths["variant_manifest"]),
        "deduplicated_variant_manifest": dedup_manifest,
        "resumed": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    try:
        materialize(build_parser().parse_args())
        return 0
    except (DatasetBuildError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
