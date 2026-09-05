"""Selection and reporting helpers for the staged Phase 1 tournament."""

from __future__ import annotations

import csv
import json
from pathlib import Path


QUALITY_KEYS = (
    "retrieval.mrr@5.mean",
    "retrieval.ndcg@5.mean",
    "retrieval.hit_rate@5.mean",
)


def load_comparison_rows(paths: list[str | Path]) -> list[dict]:
    rows = []
    for path in paths:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.extend(value.get("runs", []))
    return rows


def select_winner(rows: list[dict], *, variant: str = "resolved") -> dict:
    """Select a complete winner with deterministic tie-breaks.

    Resolved questions are the default because they are the deployment-realistic
    set: users ask standalone questions with no article already open. Pass
    variant="original" only to report the conservative paired figure.
    """
    candidates = [
        row for row in rows
        if row.get("variant") == variant and row.get("coverage.success_rate", 1.0) == 1.0
    ]
    if not candidates:
        raise ValueError(f"No complete {variant!r} candidates are available")

    def key(row: dict):
        quality = tuple(float(row.get(name, float("-inf"))) for name in QUALITY_KEYS)
        latency = -float(row.get("latency.total.p50_ms", float("inf")))
        return (*quality, latency, str(row.get("run_id", "")))

    return max(candidates, key=key)


def write_rows_csv(rows: list[dict], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return target

