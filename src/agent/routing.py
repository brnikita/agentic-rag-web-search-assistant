"""Conditional-edge functions.

Pure and side-effect free — they read state and name the next node, nothing
else, which makes every branch directly unit-testable.
"""

from __future__ import annotations

from langgraph.graph import END

from agent.state import AgentState


def route_after_analyze(state: AgentState) -> str:
    """RAG runs first when wanted, so web results append after internal ones."""
    if state.get("needs_rag"):
        return "rag_search"
    if state.get("needs_web"):
        return "web_search"
    return "check_breaker"  # no tools needed (small talk)


def route_after_rag(state: AgentState) -> str:
    return "web_search" if state.get("needs_web") else "check_breaker"


def route_after_breaker(state: AgentState) -> str:
    return "fallback" if state.get("breaker_tripped") else "synthesize"


def route_after_synthesize(state: AgentState) -> str:
    """`synthesize` trips the breaker on LLM failure, which lands here."""
    return "fallback" if state.get("breaker_tripped") else END
