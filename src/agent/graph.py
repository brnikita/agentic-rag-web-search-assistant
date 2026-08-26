"""Graph assembly.

    START -> analyze -> [rag_search] -> [web_search] -> check_breaker
                     -> synthesize | fallback -> END

The LLM never selects tools itself: `analyze` emits a structured decision and
the edges below act on it. That keeps the tool path deterministic and directly
assertable in tests.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent import nodes, routing
from agent.config import Settings, get_settings
from agent.state import AgentState

NODES = ("analyze", "rag_search", "web_search", "check_breaker", "synthesize", "fallback")


def build_checkpointer(settings: Settings | None = None) -> Any:
    """Persistence backend. Memory for dev, Postgres for multi-instance runs."""
    settings = settings or get_settings()
    if settings.checkpointer == "postgres":
        if not settings.postgres_dsn:
            raise ValueError("AGENT_CHECKPOINTER=postgres requires AGENT_POSTGRES_DSN")
        from langgraph.checkpoint.postgres import PostgresSaver

        saver = PostgresSaver.from_conn_string(settings.postgres_dsn)
        saver.setup()
        return saver

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


def build_graph(settings: Settings | None = None, *, checkpointer: Any = None):
    """Compile the agent graph.

    Args:
        settings: Overrides the process-wide settings; nodes are bound to it so
            tests get full isolation.
        checkpointer: Persistence backend. Defaults to one built from settings.
            Pass `False` to compile without one — `langgraph dev` and LangGraph
            Platform inject their own, and compiling with one already attached
            is an error there.
    """
    settings = settings or get_settings()

    builder = StateGraph(AgentState)
    for name in NODES:
        builder.add_node(name, partial(getattr(nodes, name), settings=settings))

    builder.add_edge(START, "analyze")
    builder.add_conditional_edges(
        "analyze",
        routing.route_after_analyze,
        ["rag_search", "web_search", "check_breaker"],
    )
    builder.add_conditional_edges(
        "rag_search", routing.route_after_rag, ["web_search", "check_breaker"]
    )
    builder.add_edge("web_search", "check_breaker")
    builder.add_conditional_edges(
        "check_breaker", routing.route_after_breaker, ["synthesize", "fallback"]
    )
    builder.add_conditional_edges(
        "synthesize", routing.route_after_synthesize, {"fallback": "fallback", END: END}
    )
    builder.add_edge("fallback", END)

    if checkpointer is False:
        return builder.compile()
    return builder.compile(checkpointer=checkpointer or build_checkpointer(settings))


def make_graph():
    """Entry point for `langgraph dev` / LangGraph Platform (see langgraph.json)."""
    return build_graph(checkpointer=False)
