from fastapi import APIRouter, HTTPException, status

from newsqa_rag.services import experiment_service

router = APIRouter()


@router.get("")
def list_experiments():
    return experiment_service.list_experiments()


@router.get("/{filename}/preview")
def preview_experiment(filename: str):
    try:
        return experiment_service.describe_experiment(filename, include_runs=True)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{filename}/run", status_code=status.HTTP_202_ACCEPTED)
def run_experiment(filename: str):
    try:
        return experiment_service.start_experiment(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{filename}/results")
def get_results(filename: str):
    try:
        return experiment_service.get_results(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{filename}/runs/{run_id}")
def get_run_detail(filename: str, run_id: str):
    try:
        return experiment_service.get_run_detail(filename, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
