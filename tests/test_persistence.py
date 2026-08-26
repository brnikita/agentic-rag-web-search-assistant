"""Multi-turn behaviour: history, per-turn resets, and lifetime limits."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent.runtime import Agent
from agent.tools.base import FAIL_NEXT
from helpers import decision

THREAD = "conversation-1"


def test_history_accumulates_across_turns(settings, patch_llm):
    patch_llm(
        decisions=[decision(query="refund policy"), decision(query="refund policy days")],
        answers=["Refunds take 14 days [1]", "Fourteen days [1]"],
    )
    agent = Agent(settings)

    agent.chat(THREAD, "what is the refund policy?")
    agent.chat(THREAD, "how many days?")

    state = agent.graph.get_state({"configurable": {"thread_id": THREAD}}).values
    assert [m.type for m in state["messages"]] == ["human", "ai", "human", "ai"]


def test_second_turn_sees_the_first(settings, patch_llm):
    """The router is handed history, which is how it resolves 'how many days?'."""
    chat = patch_llm(
        decisions=[decision(query="refund policy"), decision(query="refund policy days")],
        answers=["Refunds take 14 days [1]", "Fourteen days [1]"],
    )
    agent = Agent(settings)

    agent.chat(THREAD, "what is the refund policy?")
    agent.chat(THREAD, "how many days?")

    # Second synthesis call carries both user turns plus the first answer.
    second_call = chat.calls[1]
    contents = [str(m.content) for m in second_call]
    assert any("what is the refund policy?" in c for c in contents)
    assert any("how many days?" in c for c in contents)


def test_documents_do_not_leak_between_turns(settings, patch_llm):
    """The §7 regression: without the RESET sentinel these accumulate."""
    patch_llm(
        decisions=[decision(), decision(), decision()],
        answers=["one [1]", "two [1]", "three [1]"],
    )
    agent = Agent(settings)
    config = {"configurable": {"thread_id": THREAD}}

    counts = []
    for text in ("first", "second", "third"):
        agent.chat(THREAD, text)
        counts.append(len(agent.graph.get_state(config).values["documents"]))

    assert counts == [3, 3, 3]  # top_k per turn, never 3 -> 6 -> 9


def test_errors_do_not_leak_between_turns(settings, patch_llm):
    """A failure in turn 1 must not count toward the breaker in turn 2."""
    patch_llm(decisions=[decision(), decision()], answers=["failed [1]", "ok [1]"])
    agent = Agent(settings)
    config = {"configurable": {"thread_id": THREAD}}

    FAIL_NEXT["rag"] = 99
    agent.chat(THREAD, "first")
    assert len(agent.graph.get_state(config).values["errors"]) == 1

    FAIL_NEXT["rag"] = 0
    agent.chat(THREAD, "second")
    state = agent.graph.get_state(config).values
    assert state["errors"] == []
    assert state["breaker_tripped"] is False


def test_tool_calls_reset_per_turn_but_total_accumulates(settings, patch_llm):
    patch_llm(decisions=[decision(), decision()], answers=["one [1]", "two [1]"])
    agent = Agent(settings)
    config = {"configurable": {"thread_id": THREAD}}

    agent.chat(THREAD, "first")
    agent.chat(THREAD, "second")

    state = agent.graph.get_state(config).values
    assert state["tool_calls"] == 1
    assert state["total_tool_calls"] == 2


def test_thread_lifetime_breaker_trips_on_a_later_turn(settings, patch_llm):
    """Lifetime budget is what stops a long thread burning tools forever."""
    # check_breaker uses >=, so with a budget of 3 the third tool call trips it.
    tight = settings.model_copy(update={"max_tool_calls_per_thread": 3})
    patch_llm(
        decisions=[decision(), decision(), decision()],
        answers=["one [1]", "two [1]"],  # third turn must not reach synthesis
    )
    agent = Agent(tight)

    assert "couldn't fully complete" not in agent.chat(THREAD, "first")
    assert "couldn't fully complete" not in agent.chat(THREAD, "second")
    assert "couldn't fully complete" in agent.chat(THREAD, "third")


def test_threads_are_isolated(settings, patch_llm):
    patch_llm(decisions=[decision(), decision()], answers=["a [1]", "b [1]"])
    agent = Agent(settings)

    agent.chat("thread-a", "question a")
    agent.chat("thread-b", "question b")

    for thread, expected in (("thread-a", "question a"), ("thread-b", "question b")):
        state = agent.graph.get_state({"configurable": {"thread_id": thread}}).values
        assert len(state["messages"]) == 2
        assert state["messages"][0].content == expected


def test_recursion_limit_is_reported_not_raised(settings, patch_llm):
    patch_llm(decisions=[decision()], answers=["ok [1]"])
    agent = Agent(settings.model_copy(update={"recursion_limit": 1}))

    reply = agent.chat(THREAD, "anything")

    assert "step limit" in reply  # handled, not propagated


@pytest.mark.skipif(
    not __import__("os").environ.get("AGENT_POSTGRES_DSN"),
    reason="needs a live Postgres; set AGENT_POSTGRES_DSN to run",
)
def test_history_survives_a_rebuilt_graph(settings, patch_llm):
    """Durable persistence: a new process must resume an existing thread."""
    import os

    durable = settings.model_copy(
        update={"checkpointer": "postgres", "postgres_dsn": os.environ["AGENT_POSTGRES_DSN"]}
    )
    patch_llm(decisions=[decision(), decision()], answers=["one [1]", "two [1]"])

    Agent(durable).chat(THREAD, "first")
    rebuilt = Agent(durable)  # fresh graph, same store
    rebuilt.chat(THREAD, "second")

    state = rebuilt.graph.get_state({"configurable": {"thread_id": THREAD}}).values
    assert len(state["messages"]) == 4


def test_memory_saver_can_be_injected(settings, patch_llm):
    """Sanity check on the seam the postgres test above exercises for real."""
    patch_llm(decisions=[decision(), decision()], answers=["one [1]", "two [1]"])
    shared = MemorySaver()

    Agent(settings, checkpointer=shared).chat(THREAD, "first")
    rebuilt = Agent(settings, checkpointer=shared)
    rebuilt.chat(THREAD, "second")

    state = rebuilt.graph.get_state({"configurable": {"thread_id": THREAD}}).values
    assert len(state["messages"]) == 4
