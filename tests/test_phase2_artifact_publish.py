"""Tests for the private Phase 2 artifact publisher."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from newsqa_rag.evaluation.testset import DatasetBuildError
from scripts.publish_phase2_index_artifact import (
    CONFIG_NAME,
    LOCKED_PIPELINE,
    LOCKED_STATISTICS,
    MANIFEST_NAME,
    resolve_master_token,
    validate_bundle,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bundle(path: Path, *, bad_checksum: bool = False) -> Path:
    files = {
        "data/chunks.jsonl": b"{}\n" * LOCKED_STATISTICS["chunks"],
        "data/testset_resolved.jsonl": b"{}\n"
        * LOCKED_STATISTICS["resolved_questions"],
        "data/deduplicated.variant.json": b"{}\n",
        "index/bge_m3_sparse.pkl": b"trusted fixture only",
        "index/config_sparse_bge_m3_sparse.yaml": b"retrieval: {}\n",
        "index/variant_sparse_bge_m3_sparse.json": b"{}\n",
        "index/index_manifest.json": b"{}\n",
    }
    keys = [
        "chunks",
        "testset_resolved",
        "deduplicated_variant_manifest",
        "sparse_index",
        "sparse_config",
        "sparse_variant_manifest",
        "index_manifest",
    ]
    artifacts = {
        key: {"path": name, "bytes": len(value), "sha256": _sha(value)}
        for key, (name, value) in zip(keys, files.items())
    }
    if bad_checksum:
        artifacts["sparse_index"]["sha256"] = "0" * 64
    manifest = {
        "schema_version": 1,
        "artifact_version": "phase2-bge-m3-512-64-v1",
        "source": {
            "hf_repo_id": "private/source",
            "hf_revision": "v1.0.0",
            "repo_commit": "a" * 40,
        },
        "pipeline": {**LOCKED_PIPELINE, "device_at_build": "cuda"},
        "statistics": LOCKED_STATISTICS,
        "artifacts": artifacts,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest))
        archive.writestr(CONFIG_NAME, "chunking: {}\n")
        for name, value in files.items():
            archive.writestr(name, value)
    return path


def test_validate_bundle_checks_contract_counts_and_hashes(tmp_path):
    result = validate_bundle(_bundle(tmp_path / "bundle.zip"))

    assert result["manifest"]["statistics"] == LOCKED_STATISTICS
    assert len(result["bundle_sha256"]) == 64


def test_validate_bundle_rejects_checksum_mismatch(tmp_path):
    with pytest.raises(DatasetBuildError, match="Checksum mismatch"):
        validate_bundle(_bundle(tmp_path / "bundle.zip", bad_checksum=True))


def test_master_token_never_falls_back_to_shared_read_token():
    assert resolve_master_token({"HF_TOKEN": "read-only"}) is None
    assert resolve_master_token(
        {"HF_TOKEN": "read-only", "HF_TOKEN_MASTER": "publisher"}
    ) == "publisher"
