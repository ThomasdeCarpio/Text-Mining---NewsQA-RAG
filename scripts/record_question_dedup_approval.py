#!/usr/bin/env python3
"""Record explicit human approval for all proposed duplicate clusters."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.question_dedup import (
    DEDUP_APPROVAL_SCHEMA_VERSION,
    validate_cluster_decisions,
)
from src.evaluation.testset import DatasetBuildError, load_testset, sha256_file


DEFAULT_PROPOSAL = PROJECT_ROOT / "evaluation/question_dedup/newsqa_200_11064.semantic_clusters.json"
DEFAULT_APPROVAL = PROJECT_ROOT / "evaluation/question_dedup/newsqa_200_11064.human_approval.json"
DEFAULT_RESOLVED = PROJECT_ROOT / "data/evaluation/newsqa_200_11064/final/testset_resolved.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data/evaluation/newsqa_200_11064/staging/dedup/duplicate_questions_readable.md"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--resolved", type=Path, default=DEFAULT_RESOLVED)
    parser.add_argument("--reviewed-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--approve-all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.approve_all:
        raise DatasetBuildError("Pass --approve-all only after reviewing every proposed cluster")
    if args.output.exists() and not args.overwrite:
        raise DatasetBuildError(f"Approval already exists: {args.output}")
    if not args.reviewed_report.exists():
        raise DatasetBuildError(f"Reviewed report is missing: {args.reviewed_report}")

    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    resolved = load_testset(args.resolved)
    clusters = validate_cluster_decisions(proposal, resolved)
    multi_clusters = [
        cluster for cluster in clusters if len(cluster["member_question_ids"]) > 1
    ]
    approval = {
        "schema_version": DEDUP_APPROVAL_SCHEMA_VERSION,
        "proposal_path": os.path.relpath(args.proposal.resolve(), PROJECT_ROOT),
        "proposal_sha256": sha256_file(args.proposal),
        "base_testset_sha256": sha256_file(args.resolved),
        "reviewed_report": {
            "path": os.path.relpath(args.reviewed_report.resolve(), PROJECT_ROOT),
            "sha256": sha256_file(args.reviewed_report),
        },
        "human_review": {
            "status": "approved",
            "reviewer_id": args.reviewer_id,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "method": "Manual review of every proposed multi-question cluster.",
            "notes": args.notes,
        },
        "cluster_reviews": [
            {"cluster_id": cluster["cluster_id"], "decision": "approve"}
            for cluster in sorted(multi_clusters, key=lambda item: item["cluster_id"])
        ],
    }
    write_json(args.output, approval)
    print(f"Approved clusters: {len(multi_clusters)}")
    print(f"Approval: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetBuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
