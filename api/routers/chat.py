import json
from dataclasses import asdict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas import AskRequest, ChatMessageSchema
from src.services import chat_service
from src.services.session_store import get_session_store
from src.services.types import ChatMessage, Citation

router = APIRouter()


def _event_to_sse(event) -> str:
    payload = asdict(event)
    payload["timestamp"] = event.timestamp.isoformat()
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/ask")
async def ask(payload: AskRequest):
    store = get_session_store()
    store.append(payload.session_id, ChatMessage(role="user", content=payload.question))

    async def event_stream():
        final_content = ""
        final_citations: list[Citation] = []
        async for event in chat_service.ask(payload.session_id, payload.question):
            if event.type == "final_answer":
                final_content = event.content
                final_citations = event.citations or []
            yield _event_to_sse(event)
        store.append(
            payload.session_id,
            ChatMessage(role="assistant", content=final_content, citations=final_citations),
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/history/{session_id}", response_model=list[ChatMessageSchema])
def get_history(session_id: str):
    return [asdict(message) for message in get_session_store().get_history(session_id)]


@router.post("/clear/{session_id}")
def clear_chat(session_id: str):
    get_session_store().clear(session_id)
    return {"cleared": True}
