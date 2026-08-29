#!/usr/bin/env python3
"""Merge isolated Phase 1 index manifests produced by parallel workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    values = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.manifests]
    reference = values[0]
    merged = {**reference, "dense_indexes": {}, "sparse_indexes": {}}
    for value in values:
        if value["chunks_sha256"] != reference["chunks_sha256"]:
            raise SystemExit("Cannot merge indexes built from different chunk corpora")
        merged["dense_indexes"].update(value.get("dense_indexes", {}))
        merged["sparse_indexes"].update(value.get("sparse_indexes", {}))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
