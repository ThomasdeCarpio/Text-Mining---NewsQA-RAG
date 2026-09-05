#!/usr/bin/env python3
"""Validate abstention cases against their immutable source corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from newsqa_rag.evaluation.abstention import load_jsonl, validate_cases
from newsqa_rag.evaluation.benchmark_io import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output")
    parser.add_argument("--require-approved", action="store_true")
    args = parser.parse_args()
    report = validate_cases(
        load_jsonl(args.cases),
        load_jsonl(args.chunks),
        require_approved=args.require_approved,
    )
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
