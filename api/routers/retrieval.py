from fastapi import APIRouter, HTTPException

from api.schemas import (
    AlgorithmOption,
    CollectionStats,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)
from src.services import retrieval_service

router = APIRouter()


@router.get("/algorithms", response_model=list[AlgorithmOption])
def get_algorithms():
    return retrieval_service.list_algorithms()


@router.get("/stats", response_model=CollectionStats)
def get_stats():
    return retrieval_service.get_collection_stats()


@router.post("/search", response_model=RetrievalSearchResponse)
def search(payload: RetrievalSearchRequest):
    try:
        results, timing = retrieval_service.search(payload.query, payload.algorithm, payload.top_k)
        return {"results": results, "timing": timing}
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
