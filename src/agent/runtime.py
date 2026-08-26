"""Multi-turn entrypoint.

One `thread_id` is one conversation. The checkpointer holds the history, so a
turn sends only the new user message — never the whole transcript.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from agent.config import Settings, get_settings
from agent.graph import build_graph

log = structlog.get_logger()

RECURSION_MESSAGE = (
    "I couldn't complete this request: the agent exceeded its step limit. "
    "This usually means a routing problem rather than anything you did."
)


class Agent:
    """A compiled graph plus its settings.

    Holds the graph so the checkpointer survives across turns — rebuilding per
    turn would discard an in-memory saver and lose the conversation.
    """

    def __init__(self, settings: Settings | None = None, *, checkpointer: Any = None) -> None:
        self.settings = settings or get_settings()
        self.graph = build_graph(self.settings, checkpointer=checkpointer)

    def chat(self, thread_id: str, user_text: str) -> str:
        """Run one turn and return the assistant's reply."""
        turn_id = str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(thread_id=thread_id, turn_id=turn_id)
        try:
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": self.settings.recursion_limit,
            }
            try:
                result = self.graph.invoke({"messages": [HumanMessage(content=user_text)]}, config)
            except GraphRecursionError:
                # The graph-level breaker. There is no API layer in v1, so it
                # is handled here rather than being allowed to escape.
                log.error("recursion_limit_exceeded", limit=self.settings.recursion_limit)
                return RECURSION_MESSAGE

            log.info(
                "turn_complete",
                tool_calls=result.get("tool_calls", 0),
                total_tool_calls=result.get("total_tool_calls", 0),
                documents=len(result.get("documents", [])),
                errors=result.get("errors", []),
                breaker_tripped=result.get("breaker_tripped", False),
            )
            return str(result["messages"][-1].content)
        finally:
            structlog.contextvars.unbind_contextvars("thread_id", "turn_id")


def chat(thread_id: str, user_text: str) -> str:
    """One-shot convenience wrapper.

    Builds a fresh Agent per call, so with the in-memory checkpointer each call
    starts an empty thread. Use `Agent` directly for real multi-turn work.
    """
    return Agent().chat(thread_id, user_text)
