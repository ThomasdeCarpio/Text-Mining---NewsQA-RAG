import pandas as pd

from src.services.session_store import get_session_store
from src.services.types import AgentEvent


def get_dashboard_metrics() -> dict[str, float]:
    return {
        "MRR": 0.85,
        "Faithfulness": 0.92,
        "Context Precision": 0.78,
        "Context Recall": 0.81,
    }


def get_search_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["MRR", "Faithfulness", "Context Precision", "Context Recall"],
            "vector_search": [0.71, 0.85, 0.68, 0.73],
            "hybrid_search": [0.85, 0.92, 0.78, 0.81],
        }
    )


def get_failure_cases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question": "What was the reported unemployment rate?",
                "expected": "3.9%",
                "retrieved": "No matching chunk found",
                "reason": "Query terms did not match article vocabulary (retrieval miss)",
            },
            {
                "question": "Contrast WSJ and AP reactions to the jobs report",
                "expected": "Synthesis of both sources",
                "retrieved": "Only AP chunk retrieved",
                "reason": "Multi-source synthesis not yet implemented",
            },
        ]
    )


def get_pipeline_logs(limit: int = 50) -> list[AgentEvent]:
    return get_session_store().get_recent_trace(limit=limit)


def trigger_crawler() -> bool:
    return True
