#!/usr/bin/env python3
"""Validate and publish a private NewsQA evaluation dataset release."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import shutil

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from newsqa_rag.evaluation.cloud_dataset import export_canonical_bundle
from newsqa_rag.evaluation.testset import DatasetBuildError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.getenv("NEWSQA_EVAL_REPO_ID"))
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--source-root", default=str(PROJECT_ROOT / "data/evaluation/newsqa_200_11064")
    )
    parser.add_argument(
        "--export-dir", default=str(PROJECT_ROOT / "data/evaluation_exports")
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--overwrite-export", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--zip", action="store_true", help="Create a .zip archive of the export bundle")
    return parser


def publish(args: argparse.Namespace) -> dict:
    if not args.repo_id:
        raise DatasetBuildError("--repo-id or NEWSQA_EVAL_REPO_ID is required")
    if not args.version.startswith("v"):
        raise DatasetBuildError("--version must be an immutable tag such as v1.0.0")
    if not args.private:
        raise DatasetBuildError("Evaluation dataset publishing is private-only")

    export_dir = Path(args.export_dir) / "newsqa_200_11064" / args.version
    manifest = export_canonical_bundle(
        project_root=PROJECT_ROOT,
        source_root=args.source_root,
        output_dir=export_dir,
        version=args.version,
        overwrite=args.overwrite_export,
    )
    zip_path = None
    if args.zip:
        zip_base = Path(args.export_dir) / f"newsqa_200_11064_{args.version}"
        archive = shutil.make_archive(str(zip_base), "zip", root_dir=str(export_dir))
        zip_path = str(archive)
        print(f"Created dataset zip archive: {zip_path}")

    if args.dry_run:
        print(f"Validated private release: {export_dir}")
        print(f"Dataset SHA-256: {manifest['dataset_sha256']}")
        return {"manifest": manifest, "export_dir": str(export_dir), "zip_path": zip_path, "uploaded": False}
    if not args.token:
        raise DatasetBuildError("HF_TOKEN with dataset write permission is required")

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=args.token)
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        info = api.repo_info(args.repo_id, repo_type="dataset", token=args.token)
    except HfHubHTTPError as error:
        raise DatasetBuildError(f"Cannot create or access private dataset repo: {error}") from error
    if not getattr(info, "private", False):
        raise DatasetBuildError("Refusing to upload because the Hugging Face dataset is public")
    try:
        refs = api.list_repo_refs(args.repo_id, repo_type="dataset", token=args.token)
        existing_tags = {tag.name for tag in refs.tags}
    except HfHubHTTPError:
        # A newly created empty repository may not have a refs namespace yet.
        existing_tags = set()
    if args.version in existing_tags:
        raise DatasetBuildError(f"Release tag already exists: {args.version}")

    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=export_dir,
        commit_message=f"Publish private evaluation dataset {args.version}",
        token=args.token,
    )
    api.create_tag(
        args.repo_id,
        repo_type="dataset",
        tag=args.version,
        revision=commit.oid,
        tag_message=f"Immutable evaluation dataset release {args.version}",
        token=args.token,
    )
    print(f"Private dataset: https://huggingface.co/datasets/{args.repo_id}")
    print(f"Revision: {args.version} ({commit.oid})")
    print(f"Dataset SHA-256: {manifest['dataset_sha256']}")
    return {
        "manifest": manifest,
        "export_dir": str(export_dir),
        "uploaded": True,
        "commit": commit.oid,
    }


def main() -> int:
    try:
        publish(build_parser().parse_args())
        return 0
    except DatasetBuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
