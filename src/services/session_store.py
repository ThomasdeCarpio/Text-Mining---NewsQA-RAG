import threading
from collections import deque

from src.services.types import AgentEvent, ChatMessage


class SessionStore:
    """In-memory chat history + trace log, keyed by session_id.

    A module-level singleton for now. Swap for a Redis/DB-backed
    implementation later without touching callers (routers only ever
    call get_session_store()).
    """

    def __init__(self, max_trace: int = 200):
        self._lock = threading.Lock()
        self._history: dict[str, list[ChatMessage]] = {}
        self._trace: deque[AgentEvent] = deque(maxlen=max_trace)

    def get_history(self, session_id: str) -> list[ChatMessage]:
        with self._lock:
            return list(self._history.get(session_id, []))

    def append(self, session_id: str, message: ChatMessage) -> None:
        with self._lock:
            self._history.setdefault(session_id, []).append(message)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._history.pop(session_id, None)

    def record_trace(self, event: AgentEvent) -> None:
        with self._lock:
            self._trace.append(event)

    def get_recent_trace(self, limit: int = 50) -> list[AgentEvent]:
        with self._lock:
            return list(self._trace)[-limit:]


_store = SessionStore()


def get_session_store() -> SessionStore:
    return _store
