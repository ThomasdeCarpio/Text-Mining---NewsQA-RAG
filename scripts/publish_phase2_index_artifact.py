#!/usr/bin/env python3
"""Validate and publish the locked Phase 2 BGE-M3 bundle to a private Hub repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Mapping

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from newsqa_rag.evaluation.testset import DatasetBuildError

load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_REPO_ID = "ThomasAnderson2009/newsqa-rag-evaluation-artifacts"
DEFAULT_BUNDLE = (
    PROJECT_ROOT / "results/datasets/phase2-bge-m3-512-64-v1.zip"
)
MANIFEST_NAME = "phase2_index_bundle_manifest.json"
CONFIG_NAME = "configs/phase2_locked_512_64.yaml"
REQUIRED_ARTIFACTS = {
    "chunks",
    "testset_resolved",
    "deduplicated_variant_manifest",
    "sparse_index",
    "sparse_config",
    "sparse_variant_manifest",
    "index_manifest",
}
LOCKED_PIPELINE = {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "chunk_strategy": "recursive",
    "sparse_id": "bge_m3_sparse",
    "sparse_model": "BAAI/bge-m3",
}
LOCKED_STATISTICS = {"chunks": 19263, "resolved_questions": 1152}


def resolve_master_token(environ: Mapping[str, str] | None = None) -> str | None:
    """Return only the write-capable publisher token, never the shared read token."""

    source = os.environ if environ is None else environ
    return source.get("HF_TOKEN_MASTER", "").strip() or None


def sha256_stream(handle: BinaryIO, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(block_size), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def _safe_archive_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _jsonl_records(archive: zipfile.ZipFile, name: str) -> int:
    with archive.open(name) as handle:
        return sum(1 for line in handle if line.strip())


def validate_bundle(bundle_path: str | Path) -> dict:
    """Validate archive safety, provenance, counts, and every recorded checksum."""

    bundle = Path(bundle_path).resolve()
    if not bundle.is_file() or not zipfile.is_zipfile(bundle):
        raise DatasetBuildError(f"Phase 2 bundle is not a valid ZIP file: {bundle}")

    try:
        with zipfile.ZipFile(bundle) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise DatasetBuildError("Phase 2 bundle contains duplicate member names")
            for info in infos:
                mode = info.external_attr >> 16
                if not _safe_archive_path(info.filename):
                    raise DatasetBuildError(
                        f"Phase 2 bundle contains an unsafe path: {info.filename!r}"
                    )
                if info.flag_bits & 0x1:
                    raise DatasetBuildError("Encrypted bundle members are not supported")
                if stat.S_ISLNK(mode):
                    raise DatasetBuildError("Symbolic links are not allowed in the bundle")

            if MANIFEST_NAME not in names:
                raise DatasetBuildError(f"Bundle is missing {MANIFEST_NAME}")
            if CONFIG_NAME not in names:
                raise DatasetBuildError(f"Bundle is missing {CONFIG_NAME}")
            manifest = json.loads(archive.read(MANIFEST_NAME))
            if manifest.get("schema_version") != 1:
                raise DatasetBuildError("Unsupported Phase 2 bundle schema")
            version = manifest.get("artifact_version")
            if not isinstance(version, str) or not version.startswith("phase2-bge-m3-"):
                raise DatasetBuildError("Invalid Phase 2 artifact version")
            if any(
                manifest.get("pipeline", {}).get(key) != expected
                for key, expected in LOCKED_PIPELINE.items()
            ):
                raise DatasetBuildError("Bundle pipeline does not match the locked Phase 2 configuration")
            if any(
                manifest.get("statistics", {}).get(key) != expected
                for key, expected in LOCKED_STATISTICS.items()
            ):
                raise DatasetBuildError("Bundle statistics do not match the locked Phase 2 dataset")

            artifacts = manifest.get("artifacts", {})
            if set(artifacts) != REQUIRED_ARTIFACTS:
                raise DatasetBuildError(
                    "Bundle artifact set does not match the locked Phase 2 contract"
                )
            info_by_name = {info.filename: info for info in infos}
            for artifact_name, record in artifacts.items():
                member = record.get("path")
                if not isinstance(member, str) or not _safe_archive_path(member):
                    raise DatasetBuildError(f"Invalid path for artifact {artifact_name}")
                info = info_by_name.get(member)
                if info is None:
                    raise DatasetBuildError(f"Bundle is missing artifact {artifact_name}: {member}")
                if info.file_size != record.get("bytes"):
                    raise DatasetBuildError(f"Size mismatch for artifact {artifact_name}")
                with archive.open(member) as handle:
                    actual = sha256_stream(handle)
                if actual != record.get("sha256"):
                    raise DatasetBuildError(f"Checksum mismatch for artifact {artifact_name}")

            chunk_records = _jsonl_records(archive, artifacts["chunks"]["path"])
            question_records = _jsonl_records(
                archive, artifacts["testset_resolved"]["path"]
            )
            if chunk_records != LOCKED_STATISTICS["chunks"]:
                raise DatasetBuildError("Chunk record count does not match the locked manifest")
            if question_records != LOCKED_STATISTICS["resolved_questions"]:
                raise DatasetBuildError(
                    "Resolved-question count does not match the locked manifest"
                )
            manifest_bytes = archive.read(MANIFEST_NAME)
    except (json.JSONDecodeError, KeyError, TypeError, zipfile.BadZipFile) as error:
        raise DatasetBuildError(f"Malformed Phase 2 bundle: {error}") from error

    return {
        "bundle": str(bundle),
        "bundle_name": bundle.name,
        "bundle_bytes": bundle.stat().st_size,
        "bundle_sha256": sha256_file(bundle),
        "manifest": manifest,
        "manifest_bytes": manifest_bytes,
    }


def _readme(validation: dict, repo_id: str) -> str:
    manifest = validation["manifest"]
    source = manifest["source"]
    pipeline = manifest["pipeline"]
    return f"""---
private: true
---

# NewsQA RAG evaluation artifacts

Private, derived retrieval artifacts for the NewsQA RAG evaluation pipeline.
This repository is separate from the canonical raw evaluation dataset.

## {manifest['artifact_version']}

- Source dataset: `{source['hf_repo_id']}` at `{source['hf_revision']}`
- Builder commit: `{source['repo_commit']}`
- Chunking: `{pipeline['chunk_strategy']}` `{pipeline['chunk_size']}/{pipeline['chunk_overlap']}`
- Sparse retrieval: `{pipeline['sparse_model']}`
- Chunks: `{manifest['statistics']['chunks']}`
- Resolved questions: `{manifest['statistics']['resolved_questions']}`
- Bundle SHA-256: `{validation['bundle_sha256']}`

Download from the private dataset repository `{repo_id}`, pin the immutable tag,
and verify `SHA256SUMS` before extraction. The bundle contains a Python pickle;
load it only from this trusted repository after checksum verification.
"""


def _existing_tags(api, repo_id: str, token: str) -> set[str]:
    try:
        refs = api.list_repo_refs(repo_id, repo_type="dataset", token=token)
    except Exception as error:
        # Newly created empty repositories can lack a refs namespace. Do not hide
        # authorization, connectivity, or server errors for established repos.
        if "empty" not in str(error).lower() and "404" not in str(error):
            raise
        return set()
    return {tag.name for tag in refs.tags}


def publish(args: argparse.Namespace) -> dict:
    validation = validate_bundle(args.bundle)
    manifest = validation["manifest"]
    tag = args.tag or manifest["artifact_version"]
    if tag in {"main", "master"} or not tag.startswith("phase2-"):
        raise DatasetBuildError("Use an immutable Phase 2 tag such as phase2-bge-m3-512-64-v1")

    result = {
        "repo_id": args.repo_id,
        "tag": tag,
        "artifact_version": manifest["artifact_version"],
        "bundle": validation["bundle"],
        "bundle_sha256": validation["bundle_sha256"],
        "uploaded": False,
    }
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    token = args.token or resolve_master_token()
    if not token:
        raise DatasetBuildError("HF_TOKEN_MASTER with dataset write permission is required")

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=token)
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        info = api.repo_info(
            args.repo_id, repo_type="dataset", token=token
        )
    except HfHubHTTPError as error:
        raise DatasetBuildError(f"Cannot create or access private artifact repo: {error}") from error
    if not getattr(info, "private", False):
        raise DatasetBuildError("Refusing to upload because the artifact repository is public")
    if tag in _existing_tags(api, args.repo_id, token):
        raise DatasetBuildError(f"Immutable artifact tag already exists: {tag}")

    with tempfile.TemporaryDirectory(prefix="newsqa-phase2-publish-") as directory:
        staging = Path(directory)
        artifact_dir = staging / "artifacts" / manifest["artifact_version"]
        artifact_dir.mkdir(parents=True)
        shutil.copy2(validation["bundle"], artifact_dir / validation["bundle_name"])
        (artifact_dir / MANIFEST_NAME).write_bytes(validation["manifest_bytes"])
        (artifact_dir / "SHA256SUMS").write_text(
            f"{validation['bundle_sha256']}  {validation['bundle_name']}\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            _readme(validation, args.repo_id), encoding="utf-8"
        )
        try:
            commit = api.upload_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                folder_path=staging,
                commit_message=f"Publish {manifest['artifact_version']}",
                token=token,
            )
            api.create_tag(
                repo_id=args.repo_id,
                repo_type="dataset",
                tag=tag,
                revision=commit.oid,
                tag_message=f"Immutable {manifest['artifact_version']} release",
                token=token,
            )
        except HfHubHTTPError as error:
            raise DatasetBuildError(f"Artifact upload failed: {error}") from error

    result.update({"uploaded": True, "commit": commit.oid})
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.getenv("NEWSQA_ARTIFACT_REPO_ID", DEFAULT_REPO_ID),
    )
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--tag", help="Immutable Hub tag; defaults to artifact_version")
    parser.add_argument(
        "--token",
        default=resolve_master_token(),
        help="Publisher token; defaults exclusively to HF_TOKEN_MASTER",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    try:
        publish(build_parser().parse_args())
        return 0
    except DatasetBuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
