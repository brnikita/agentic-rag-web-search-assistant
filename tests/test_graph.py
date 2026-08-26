"""Graph wiring tests, with the chat model faked out — no API key needed."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

import agent.graph as graph_module


class _FakeChat(GenericFakeChatModel):
    """Fake model that calls the docs tool once, then answers."""

    def bind_tools(self, tools, **kwargs):  # noqa: D102 - mirrors BaseChatModel
        return self


def _fake_llm(responses):
    return _FakeChat(messages=iter(responses))


def test_graph_runs_tool_then_answers(settings, monkeypatch, _warm_embeddings):
    import agent.tools.docs_search as docs_search
    from agent.retrieval.ingest import ingest

    monkeypatch.setattr(docs_search, "get_settings", lambda: settings)
    ingest(settings, reset=True)

    tool_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_internal_docs", "args": {"query": "rollback"}, "id": "call_1"}
        ],
    )
    final = AIMessage(content="Use `make rollback ENV=production` (docs/deployment.md).")
    monkeypatch.setattr(graph_module, "build_llm", lambda s: _fake_llm([tool_call, final]))

    graph = graph_module.build_graph(settings)
    result = graph.invoke(
        {"messages": [HumanMessage(content="How do I roll back?")]},
        config={"configurable": {"thread_id": "t1"}},
    )

    kinds = [m.type for m in result["messages"]]
    assert kinds == ["human", "ai", "tool", "ai"]
    assert "make rollback" in result["messages"][-1].content


def test_graph_ends_without_tool_calls(settings, monkeypatch):
    final = AIMessage(content="Hello.")
    monkeypatch.setattr(graph_module, "build_llm", lambda s: _fake_llm([final]))

    graph = graph_module.build_graph(settings)
    result = graph.invoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "t2"}},
    )

    assert [m.type for m in result["messages"]] == ["human", "ai"]


def test_checkpointer_preserves_thread_history(settings, monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "build_llm",
        lambda s: _fake_llm([AIMessage(content="one"), AIMessage(content="two")]),
    )
    graph = graph_module.build_graph(settings)
    config = {"configurable": {"thread_id": "t3"}}

    graph.invoke({"messages": [HumanMessage(content="first")]}, config=config)
    result = graph.invoke({"messages": [HumanMessage(content="second")]}, config=config)

    assert [m.content for m in result["messages"]] == ["first", "one", "second", "two"]
