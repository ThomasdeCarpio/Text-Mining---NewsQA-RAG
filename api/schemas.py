from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class CitationSchema(BaseModel):
    source: str
    title: str
    date: str
    url: str
    chunk_text: str


class AgentEventSchema(BaseModel):
    type: Literal["thought", "tool_call", "tool_result", "final_answer"]
    content: str
    tool_name: Optional[str] = None
    citations: Optional[list[CitationSchema]] = None
    timestamp: datetime


class ChatMessageSchema(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    citations: list[CitationSchema] = []


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    session_id: str
    username: str
    role: Literal["user", "admin"]


class AskRequest(BaseModel):
    session_id: str
    question: str


class TriggerCrawlerResponse(BaseModel):
    triggered: bool
