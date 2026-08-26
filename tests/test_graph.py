"""Graph wiring: which nodes actually run, for each routing decision."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from agent.tools.base import FAIL_NEXT
from helpers import decision


def run(graph, text, thread="t1"):
    """Invoke one turn and return (final_state, ordered node names that ran)."""
    config = {"configurable": {"thread_id": thread}}
    path = []
    for update in graph.stream(
        {"messages": [HumanMessage(content=text)]}, config, stream_mode="updates"
    ):
        path.extend(update)  # one key per node that ran in this superstep
    return graph.get_state(config).values, path


def test_rag_only_path(settings, patch_llm):
    patch_llm(decisions=[decision(rag=True, web=False)], answers=["Refunds take 14 days [1]"])
    graph = build_graph(settings)

    state, path = run(graph, "what is the refund policy?")

    assert path == ["analyze", "rag_search", "check_breaker", "synthesize"]
    assert state["tool_calls"] == 1
    assert all(d["source"] == "internal" for d in state["documents"])


def test_rag_and_web_path(settings, patch_llm):
    patch_llm(decisions=[decision(rag=True, web=True)], answers=["Both [1][2]"])
    graph = build_graph(settings)

    state, path = run(graph, "latest news about refund law vs our policy")

    assert path == ["analyze", "rag_search", "web_search", "check_breaker", "synthesize"]
    assert state["tool_calls"] == 2
    assert {d["source"] for d in state["documents"]} == {"internal", "web"}


def test_web_only_path(settings, patch_llm):
    patch_llm(decisions=[decision(rag=False, web=True)], answers=["News [1]"])
    graph = build_graph(settings)

    _, path = run(graph, "what happened in the news today?")

    assert path == ["analyze", "web_search", "check_breaker", "synthesize"]


def test_no_tools_path(settings, patch_llm):
    patch_llm(decisions=[decision(rag=False, web=False)], answers=["Hello!"])
    graph = build_graph(settings)

    state, path = run(graph, "hello")

    assert path == ["analyze", "check_breaker", "synthesize"]
    assert state["tool_calls"] == 0
    assert state["documents"] == []


def test_breaker_routes_to_fallback_when_all_tools_fail(settings, patch_llm):
    patch_llm(decisions=[decision(rag=True, web=True)], answers=[])
    FAIL_NEXT["rag"] = 99
    FAIL_NEXT["web"] = 99
    graph = build_graph(settings)

    state, path = run(graph, "anything")

    assert path == ["analyze", "rag_search", "web_search", "check_breaker", "fallback"]
    assert state["breaker_tripped"] is True
    assert "couldn't fully complete" in state["messages"][-1].content
    assert len(state["errors"]) == 2


def test_partial_failure_still_synthesizes(settings, patch_llm):
    """One tool down is below the error threshold; answer from what we have."""
    patch_llm(decisions=[decision(rag=True, web=True)], answers=["Partial [1]"])
    FAIL_NEXT["web"] = 99
    graph = build_graph(settings)

    state, path = run(graph, "anything")

    assert path[-1] == "synthesize"
    assert state["breaker_tripped"] is False
    assert len(state["errors"]) == 1
    assert all(d["source"] == "internal" for d in state["documents"])


def test_synthesis_failure_routes_to_fallback(settings, patch_llm):
    patch_llm(decisions=[decision(rag=True)], answers=[RuntimeError("llm down")])
    graph = build_graph(settings)

    state, path = run(graph, "refund policy")

    assert path == ["analyze", "rag_search", "check_breaker", "synthesize", "fallback"]
    assert "partial source" in state["messages"][-1].content


def test_router_failure_falls_back_to_rag_only(settings, patch_llm):
    patch_llm(decisions=[RuntimeError("router down")], answers=["Answer [1]"])
    graph = build_graph(settings)

    state, path = run(graph, "what is the refund policy?")

    assert path == ["analyze", "rag_search", "check_breaker", "synthesize"]
    assert state["rewritten_query"] == "what is the refund policy?"
    assert state["errors"] == ["analyze: router down"]
    assert state["breaker_tripped"] is False  # routing errors alone must not trip


def test_synthesis_prompt_carries_numbered_sources(settings, patch_llm):
    chat = patch_llm(decisions=[decision(rag=True)], answers=["ok"])
    graph = build_graph(settings)

    run(graph, "refund policy")

    system_prompt = chat.calls[0][0].content
    assert "[1]" in system_prompt and "Refund Policy v3" in system_prompt
