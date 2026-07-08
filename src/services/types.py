from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


@dataclass
class Citation:
    source: str
    title: str
    date: str
    url: str
    chunk_text: str


@dataclass
class AgentEvent:
    type: Literal["thought", "tool_call", "tool_result", "final_answer"]
    content: str
    tool_name: Optional[str] = None
    citations: Optional[list[Citation]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class User:
    username: str
    role: Literal["user", "admin"]
