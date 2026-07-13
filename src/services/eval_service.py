import json
from pathlib import Path

import pandas as pd

from src.services.session_store import get_session_store
from src.services.types import AgentEvent

# Reports written by scripts/run_benchmark.py --report-dir reports/<name>/
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

# (dashboard card label, report group, key inside that group)
_METRIC_SPECS = [
    ("MRR", "retrieval", "mrr@5"),
    ("Faithfulness", "ragas", "faithfulness"),
    ("Context Precision", "ragas", "context_precision"),
    ("Context Recall", "ragas", "context_recall"),
]


def _load_reports() -> dict[str, dict]:
    """All report.json under reports/, keyed by retriever type (dense/bm25/hybrid)."""
    reports = {}
    for path in sorted(REPORTS_DIR.glob("*/report.json")):
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        reports[report.get("config", {}).get("retriever", path.parent.name)] = report
    return reports


def _pick(report: dict, group: str, key: str) -> float | None:
    return report.get(group, {}).get(key) if report else None


def get_dashboard_metrics() -> dict[str, float]:
    reports = _load_reports()
    primary = reports.get("hybrid") or next(iter(reports.values()), None)
    if not primary:
        return {}
    return {
        label: v
        for label, group, key in _METRIC_SPECS
        if (v := _pick(primary, group, key)) is not None
    }


def get_search_comparison() -> pd.DataFrame:
    reports = _load_reports()
    dense, hybrid = reports.get("dense"), reports.get("hybrid")
    return pd.DataFrame([
        {
            "metric": label,
            "vector_search": _pick(dense, group, key) or 0.0,
            "hybrid_search": _pick(hybrid, group, key) or 0.0,
        }
        for label, group, key in _METRIC_SPECS
    ])


def get_failure_cases() -> pd.DataFrame:
    for report in _load_reports().values():
        if report.get("failures"):
            return pd.DataFrame(report["failures"])
    return pd.DataFrame(columns=["question", "expected", "retrieved", "reason"])


def get_pipeline_logs(limit: int = 50) -> list[AgentEvent]:
    return get_session_store().get_recent_trace(limit=limit)


def trigger_crawler() -> bool:
    return True
