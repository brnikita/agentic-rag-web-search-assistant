"""The reducer that keeps per-turn evidence from leaking across turns."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.state import RESET, AgentState, append_or_reset


def test_appends_to_existing():
    assert append_or_reset(["a"], ["b"]) == ["a", "b"]


def test_appends_to_none():
    assert append_or_reset(None, ["a"]) == ["a"]


def test_reset_clears():
    assert append_or_reset(["a", "b"], RESET) == []


def test_reset_on_empty_stays_empty():
    assert append_or_reset(None, RESET) == []


def test_rejects_a_non_sentinel_string():
    # Guards against a node returning a bare string by mistake, which would
    # otherwise be silently swallowed.
    with pytest.raises(ValueError, match="expected a list"):
        append_or_reset([], "oops")


def test_documents_do_not_leak_across_turns():
    """The regression the sentinel exists to prevent.

    With `operator.add` this accumulates one document per turn; the channel is
    persisted by the checkpointer and a plain `[]` write cannot clear it.
    """

    def analyze(state: AgentState) -> dict:
        return {"documents": RESET, "errors": RESET}

    def rag(state: AgentState) -> dict:
        return {
            "documents": [
                {
                    "id": "d1",
                    "source": "internal",
                    "title": "t",
                    "url": None,
                    "content": "c",
                    "score": 1.0,
                }
            ]
        }

    builder = StateGraph(AgentState)
    builder.add_node("analyze", analyze)
    builder.add_node("rag", rag)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "rag")
    builder.add_edge("rag", END)
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "t1"}}
    for _ in range(3):
        result = graph.invoke({}, config)
        assert len(result["documents"]) == 1
