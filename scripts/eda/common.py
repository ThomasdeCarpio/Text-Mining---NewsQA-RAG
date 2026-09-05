"""Shared loaders and helpers for the NewsQA EDA.

Every turn-by-turn EDA script imports from here so paths and record
schemas are declared once.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "data" / "evaluation" / "newsqa_200_11064"
STAGING = DATASET / "staging"
FINAL = DATASET / "final"
OUT = PROJECT / "outputs" / "eda"

# Question variants, in pipeline order.
VARIANTS = {
    "original": FINAL / "testset_original.jsonl",
    "reviewed_original": FINAL / "testset_reviewed_original.jsonl",
    "resolved": FINAL / "testset_resolved.jsonl",
    "clarified": FINAL / "testset_clarified.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def articles(role: str) -> list[dict]:
    """Load staged corpus articles. role is 'evaluation' or 'distractor'."""
    return read_jsonl(STAGING / "corpus" / f"{role}_articles.jsonl")


def source_questions() -> list[dict]:
    """Load the 1,340 questions as selected from raw NewsQA, pre-review."""
    return read_jsonl(STAGING / "questions" / "original_questions.jsonl")


def variant(name: str) -> list[dict]:
    return read_jsonl(VARIANTS[name])


def chunks() -> list[dict]:
    return read_jsonl(FINAL / "chunks.jsonl")


def annotations() -> list[dict]:
    return read_jsonl(FINAL / "review_annotations.jsonl")


def describe(values: list[float], name: str = "") -> dict:
    """Five-number summary plus mean, for any numeric series."""
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return {"name": name, "n": 0}

    def pct(p: float):
        return ordered[min(n - 1, int(p * n))]

    return {
        "name": name,
        "n": n,
        "min": ordered[0],
        "p25": pct(0.25),
        "median": st.median(ordered),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": ordered[-1],
        "mean": round(st.fmean(ordered), 2),
    }


def table(rows: list[dict], columns: list[str], widths: list[int]) -> str:
    """Render a fixed-width text table for terminal output."""
    head = "  ".join(c.ljust(w) for c, w in zip(columns, widths))
    line = "  ".join("-" * w for w in widths)
    body = [
        "  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(columns, widths))
        for row in rows
    ]
    return "\n".join([head, line, *body])


def save(name: str, payload: dict) -> Path:
    """Write a result payload so later turns can reuse it without recomputing."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path
