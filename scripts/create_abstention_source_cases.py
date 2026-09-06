#!/usr/bin/env python3
"""Create full-corpus retrieval inputs for Phase 3 source-question screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from newsqa_rag.evaluation.abstention import load_jsonl, save_jsonl
from newsqa_rag.evaluation.testset import DatasetBuildError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", required=True)
    parser.add_argument("--question-ids-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.testset)
    requested = json.loads(Path(args.question_ids_file).read_text(encoding="utf-8"))
    if not isinstance(requested, list) or len(requested) != len(set(requested)):
        raise DatasetBuildError("--question-ids-file must contain a unique JSON list")
    by_id = {str(row["question_id"]): row for row in rows}
    missing = [str(value) for value in requested if str(value) not in by_id]
    if missing:
        raise DatasetBuildError(f"Unknown question IDs: {missing[:5]}")
    cases = [
        {
            "case_id": question_id,
            "base_question_id": question_id,
            "case_type": "source_screening",
            "question": by_id[question_id]["question"],
            "scope": "full_corpus",
            "excluded_chunk_ids": [],
            "excluded_article_ids": [],
        }
        for question_id in map(str, requested)
    ]
    save_jsonl(cases, args.output)
    print(f"Source retrieval cases: {args.output} ({len(cases)} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
