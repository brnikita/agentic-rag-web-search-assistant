"""The citation contract: every [n] must point at real evidence."""

from __future__ import annotations

import pytest

from agent.citations import cited_indices, validate_citations
from agent.graph import build_graph
from agent.nodes import format_sources
from agent.prompts import NO_SOURCES
from agent.state import Document
from helpers import decision


def docs(n: int) -> list[Document]:
    return [
        Document(
            id=f"int-{i}",
            source="internal",
            title=f"Doc {i}",
            url=None,
            content="body",
            score=1.0,
        )
        for i in range(n)
    ]


def test_valid_citations_pass_through():
    text = "Refunds take 14 days [1] and rate limits apply [2]."
    assert validate_citations(text, docs(2)) == text


def test_out_of_range_citation_is_stripped():
    assert validate_citations("Claim [3].", docs(2)) == "Claim."


def test_grouped_citation_keeps_only_valid_members():
    assert validate_citations("Claim [1, 3].", docs(2)) == "Claim [1]."


def test_all_citations_stripped_when_no_documents():
    assert validate_citations("Invented [1][2].", []) == "Invented."


def test_text_without_citations_is_untouched():
    text = "I don't have sources for that."
    assert validate_citations(text, docs(2)) == text


def test_stripping_does_not_leave_double_spaces():
    assert "  " not in validate_citations("A [9] B [1] C [9] D", docs(1))


@pytest.mark.parametrize(
    ("text", "expected"),
    [("[1]", {1}), ("[1][2]", {1, 2}), ("[1, 2]", {1, 2}), ("none", set())],
)
def test_cited_indices(text, expected):
    assert cited_indices(text) == expected


def test_format_sources_numbers_from_one():
    rendered = format_sources(docs(2))
    assert rendered.startswith("[1] Doc 0")
    assert "[2] Doc 1" in rendered


def test_format_sources_reports_emptiness():
    assert format_sources([]) == NO_SOURCES


def test_graph_strips_invented_citations(settings, patch_llm):
    """End to end: a hallucinated marker never reaches the user."""
    patch_llm(
        decisions=[decision(rag=True)],
        answers=["Refunds take 14 days [1]. Also, unrelated fact [9]."],
    )
    graph = build_graph(settings)
    config = {"configurable": {"thread_id": "t1"}}

    from langchain_core.messages import HumanMessage

    graph.invoke({"messages": [HumanMessage(content="refund policy")]}, config)
    answer = graph.get_state(config).values["messages"][-1].content

    assert "[1]" in answer
    assert "[9]" not in answer


def test_graph_keeps_citations_that_match_evidence(settings, patch_llm):
    patch_llm(decisions=[decision(rag=True)], answers=["A [1] B [2] C [3]"])
    graph = build_graph(settings)
    config = {"configurable": {"thread_id": "t2"}}

    from langchain_core.messages import HumanMessage

    graph.invoke({"messages": [HumanMessage(content="refund policy")]}, config)
    state = graph.get_state(config).values

    # The mock corpus yields 3 documents, so all three markers are legitimate.
    assert cited_indices(state["messages"][-1].content) <= set(
        range(1, len(state["documents"]) + 1)
    )
    assert "[3]" in state["messages"][-1].content
