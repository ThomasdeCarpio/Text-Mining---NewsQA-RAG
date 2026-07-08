from dataclasses import asdict

from fastapi import APIRouter

from api.schemas import AgentEventSchema, TriggerCrawlerResponse
from src.services import eval_service

router = APIRouter()


@router.get("/metrics", response_model=dict[str, float])
def get_metrics():
    return eval_service.get_dashboard_metrics()


@router.get("/search-comparison")
def get_search_comparison():
    return eval_service.get_search_comparison().to_dict(orient="records")


@router.get("/failure-cases")
def get_failure_cases():
    return eval_service.get_failure_cases().to_dict(orient="records")


@router.get("/pipeline-logs", response_model=list[AgentEventSchema])
def get_pipeline_logs(limit: int = 50):
    return [asdict(event) for event in eval_service.get_pipeline_logs(limit=limit)]


@router.post("/trigger-crawler", response_model=TriggerCrawlerResponse)
def trigger_crawler():
    return TriggerCrawlerResponse(triggered=eval_service.trigger_crawler())
