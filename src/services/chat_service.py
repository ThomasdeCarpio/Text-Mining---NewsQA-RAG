from collections.abc import AsyncIterator

from src.services.session_store import get_session_store
from src.services.types import AgentEvent, Citation

_MOCK_CITATIONS = [
    Citation(
        source="AP",
        title="Labor Department Reports Latest Unemployment Figures",
        date="2026-06-30",
        url="https://example.com/ap/unemployment-report",
        chunk_text="The Labor Department reported the unemployment rate fell to 3.9% last month...",
    ),
    Citation(
        source="Wall Street Journal",
        title="Jobs Report Beats Expectations",
        date="2026-07-01",
        url="https://example.com/wsj/jobs-report",
        chunk_text="Economists had forecast a smaller decline, but the report showed broad gains across sectors...",
    ),
]


async def ask(session_id: str, question: str) -> AsyncIterator[AgentEvent]:
    """Mock ReAct-style agent loop.

    Yields step events (thought/tool_call/tool_result) then a final_answer
    event carrying the answer + citations. Real implementation later plugs
    into src/agents/orchestrator.py behind this same async generator shape,
    so the FastAPI streaming endpoint doesn't need to change.
    """
    store = get_session_store()

    events = [
        AgentEvent(type="thought", content=f"Analyzing question: '{question}'"),
        AgentEvent(
            type="tool_call",
            tool_name="hybrid_search",
            content=f"Searching corpus for: {question}",
        ),
        AgentEvent(
            type="tool_result",
            tool_name="hybrid_search",
            content="Found 2 relevant chunks.",
        ),
    ]
    for event in events:
        store.record_trace(event)
        yield event

    final = AgentEvent(
        type="final_answer",
        content=f"(mock answer) Based on the retrieved articles, here is a response to: '{question}'",
        citations=_MOCK_CITATIONS,
    )
    store.record_trace(final)
    yield final
