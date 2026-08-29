#!/usr/bin/env python3
"""Fail-fast integrity validation for Phase 1 retrieval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chromadb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-manifest", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.index_manifest).read_text(encoding="utf-8"))
    chunks_path = Path(manifest["chunks_path"])
    actual_hash = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    if actual_hash != manifest["chunks_sha256"]:
        raise SystemExit("Chunk corpus hash does not match index manifest")
    if len(manifest.get("dense_indexes", {})) != 4 or len(manifest.get("sparse_indexes", {})) != 4:
        raise SystemExit("Phase 1 requires exactly four dense and four sparse indexes")
    for name, record in manifest["dense_indexes"].items():
        count = chromadb.PersistentClient(record["db_path"]).get_collection(record["collection_name"]).count()
        if count != manifest["total_chunks"] or record.get("chunk_count") != count:
            raise SystemExit(f"Dense index {name} has {count} chunks; expected {manifest['total_chunks']}")
        if not Path(record["variant_manifest"]).exists():
            raise SystemExit(f"Dense profile manifest missing for {name}")
    for name, record in manifest["sparse_indexes"].items():
        if not Path(record["index_path"]).exists() or not Path(record["variant_manifest"]).exists():
            raise SystemExit(f"Sparse artifacts missing for {name}")
    print(json.dumps({"status": "ready", "dense": 4, "sparse": 4, "chunks": manifest["total_chunks"]}))


if __name__ == "__main__":
    main()
