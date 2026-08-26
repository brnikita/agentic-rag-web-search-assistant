"""Graph state, and the reducers that maintain it.

The subtle part is `append_or_reset`. `documents` and `errors` are evidence
gathered *within one turn*, but the checkpointer persists the whole state
across turns. A plain append-only reducer (`operator.add`) therefore leaks:
turn 3 would cite documents retrieved in turn 1, and the breaker would count
errors from turns that already finished.

`update_state({"documents": []})` does not fix this — writes go *through* the
reducer, so `add(existing, [])` returns the list unchanged. Instead the reducer
understands a sentinel: `analyze` writes RESET at the start of every turn to
clear the channel, and later nodes append to it normally.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict, TypeVar

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

RESET: Literal["__reset__"] = "__reset__"

T = TypeVar("T")


def append_or_reset(left: list[T] | None, right: list[T] | str) -> list[T]:
    """Append `right` to `left`, unless `right` is the RESET sentinel.

    A bare string is never a legitimate value for these channels — they hold
    lists — so the sentinel is unambiguous.
    """
    if isinstance(right, str):
        if right != RESET:
            raise ValueError(f"expected a list or {RESET!r}, got {right!r}")
        return []
    return (left or []) + (right or [])


class Document(TypedDict):
    """One piece of retrieved evidence, from either source."""

    id: str
    source: Literal["internal", "web"]
    title: str
    url: str | None
    content: str
    score: float


class AgentState(TypedDict, total=False):
    """State threaded through the graph and persisted per `thread_id`."""

    # Conversation. Persisted across turns; `add_messages` appends and
    # de-dupes by id.
    messages: Annotated[list[BaseMessage], add_messages]

    # Per-turn routing, set by `analyze`.
    rewritten_query: str
    needs_rag: bool
    needs_web: bool

    # Per-turn evidence. Cleared by `analyze` via RESET.
    documents: Annotated[list[Document], append_or_reset]
    errors: Annotated[list[str], append_or_reset]

    # Safety counters. `tool_calls` is per-turn (reset by `analyze`);
    # `total_tool_calls` is the lifetime of the thread and is never reset.
    tool_calls: int
    total_tool_calls: int
    breaker_tripped: bool
