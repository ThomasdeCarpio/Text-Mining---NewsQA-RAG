from dataclasses import asdict

from fastapi import APIRouter

from newsqa_rag.api.schemas import AgentEventSchema
from newsqa_rag.services import eval_service

router = APIRouter()


@router.get("/metrics", response_model=dict[str, float])
def get_metrics():
    return eval_service.get_dashboard_metrics()


@router.get("/search-comparison")
def get_search_comparison():
    return eval_service.get_search_comparison()


@router.get("/failure-cases")
def get_failure_cases():
    return eval_service.get_failure_cases()


@router.get("/pipeline-logs", response_model=list[AgentEventSchema])
def get_pipeline_logs(limit: int = 50):
    return [asdict(event) for event in eval_service.get_pipeline_logs(limit=limit)]
