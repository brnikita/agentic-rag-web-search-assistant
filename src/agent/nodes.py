"""Graph nodes.

Every node catches its own exceptions and returns a state delta rather than
raising. That is what keeps the graph from dying mid-turn: state is always
written, so the checkpoint stays consistent and the breaker gets a chance to
route to `fallback`.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import AIMessage, SystemMessage

from agent.citations import validate_citations
from agent.config import Settings, get_settings
from agent.llm import build_llm, build_router
from agent.prompts import ANALYZE_PROMPT, NO_SOURCES, SYNTH_PROMPT
from agent.state import RESET, AgentState, Document
from agent.tools import SearchFn, get_rag_search, get_web_search, guarded

log = structlog.get_logger()

# How much history each LLM call sees. Bounded so a long thread cannot grow
# the prompt without limit.
ROUTER_HISTORY = 6
SYNTH_HISTORY = 10


def analyze(state: AgentState, settings: Settings | None = None) -> dict:
    """Decide which tools this turn needs, and clear last turn's evidence.

    RESET is what stops `documents` and `errors` accumulating across turns —
    see `agent.state` for why an empty-list write cannot do it.
    """
    settings = settings or get_settings()
    base: dict = {
        "documents": RESET,
        "errors": RESET,
        "tool_calls": 0,
        "breaker_tripped": False,
    }
    try:
        router = build_router(build_llm(settings))
        messages = [SystemMessage(ANALYZE_PROMPT), *state["messages"][-ROUTER_HISTORY:]]
        decision = router.invoke(messages)
        log.info("route", **decision.model_dump())
        return base | {
            "rewritten_query": decision.rewritten_query,
            "needs_rag": decision.needs_rag,
            "needs_web": decision.needs_web,
        }
    except Exception as exc:
        # Fail safe: internal docs only. The error is prefixed so the breaker
        # can tell a routing failure from a tool failure.
        log.warning("analyze_failed", error=str(exc))
        return base | {
            "rewritten_query": str(state["messages"][-1].content),
            "needs_rag": True,
            "needs_web": False,
            "errors": [f"analyze: {exc}"],
        }


def _run_tool(name: str, fn: SearchFn, top_k: int, state: AgentState, settings: Settings) -> dict:
    """Run one search, counting the call whether or not it succeeds."""
    counters = {
        "tool_calls": state.get("tool_calls", 0) + 1,
        "total_tool_calls": state.get("total_tool_calls", 0) + 1,
    }
    try:
        documents = guarded(fn, state["rewritten_query"], top_k, settings)
        log.info("tool_ok", tool=name, n=len(documents))
        return counters | {"documents": documents}
    except Exception as exc:
        log.error("tool_failed", tool=name, error=str(exc))
        return counters | {"errors": [f"{name}: {exc}"]}


def rag_search(state: AgentState, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    return _run_tool("rag", get_rag_search(settings), settings.rag_top_k, state, settings)


def web_search(state: AgentState, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    return _run_tool("web", get_web_search(settings), settings.web_top_k, state, settings)


def check_breaker(state: AgentState, settings: Settings | None = None) -> dict:
    """Decide whether to give up on this turn.

    Routing failures are excluded from the error count: `analyze` already
    degraded gracefully, so they should not on their own trip the breaker.
    """
    settings = settings or get_settings()
    tool_errors = [e for e in state.get("errors", []) if not e.startswith("analyze:")]
    tripped = (
        state.get("tool_calls", 0) >= settings.max_tool_calls_per_turn
        or state.get("total_tool_calls", 0) >= settings.max_tool_calls_per_thread
        or len(tool_errors) >= settings.max_error_count
    )
    if tripped:
        log.warning(
            "breaker_tripped",
            tool_calls=state.get("tool_calls", 0),
            total_tool_calls=state.get("total_tool_calls", 0),
            errors=state.get("errors", []),
        )
    return {"breaker_tripped": tripped}


def format_sources(documents: list[Document]) -> str:
    """Render evidence as the numbered block the citation contract refers to."""
    if not documents:
        return NO_SOURCES
    lines = []
    for index, doc in enumerate(documents, start=1):
        origin = doc["source"] + (f", {doc['url']}" if doc["url"] else "")
        lines.append(f"[{index}] {doc['title']} ({origin})\n{doc['content']}")
    return "\n\n".join(lines)


def synthesize(state: AgentState, settings: Settings | None = None) -> dict:
    """Answer from the gathered evidence, with numbered citations."""
    settings = settings or get_settings()
    documents = state.get("documents", [])
    sources = format_sources(documents)
    try:
        llm = build_llm(settings)
        messages = [
            SystemMessage(SYNTH_PROMPT.format(sources=sources)),
            *state["messages"][-SYNTH_HISTORY:],
        ]
        answer = llm.invoke(messages)
        # A prompt cannot guarantee the contract; enforce it on the way out.
        answer.content = validate_citations(str(answer.content), documents)
        return {"messages": [answer]}
    except Exception as exc:
        log.error("synthesize_failed", error=str(exc))
        # Trip the breaker so the graph routes on to `fallback`.
        return {"errors": [f"synthesize: {exc}"], "breaker_tripped": True}


def fallback(state: AgentState, settings: Settings | None = None) -> dict:
    """Say honestly that the turn could not be completed."""
    documents = state.get("documents", [])
    partial = f" I did find {len(documents)} partial source(s)." if documents else ""
    return {
        "messages": [
            AIMessage(
                content=(
                    "I couldn't fully complete this request due to a tool or service "
                    f"issue.{partial} Please try again or rephrase your question."
                )
            )
        ]
    }
