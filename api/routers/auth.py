import uuid

from fastapi import APIRouter, HTTPException

from api.schemas import LoginRequest, LoginResponse
from src.services import auth_service

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest):
    user = auth_service.login(payload.username, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return LoginResponse(
        session_id=str(uuid.uuid4()), username=user.username, role=user.role
    )
