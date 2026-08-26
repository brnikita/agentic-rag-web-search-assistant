"""Every branch of every conditional edge."""

from __future__ import annotations

import pytest
from langgraph.graph import END

from agent.routing import (
    route_after_analyze,
    route_after_breaker,
    route_after_rag,
    route_after_synthesize,
)


@pytest.mark.parametrize(
    ("needs_rag", "needs_web", "expected"),
    [
        (True, False, "rag_search"),
        (True, True, "rag_search"),  # rag first, web appended after
        (False, True, "web_search"),
        (False, False, "check_breaker"),  # small talk needs no evidence
    ],
)
def test_route_after_analyze(needs_rag, needs_web, expected):
    assert route_after_analyze({"needs_rag": needs_rag, "needs_web": needs_web}) == expected


@pytest.mark.parametrize(
    ("needs_web", "expected"), [(True, "web_search"), (False, "check_breaker")]
)
def test_route_after_rag(needs_web, expected):
    assert route_after_rag({"needs_web": needs_web}) == expected


@pytest.mark.parametrize(("tripped", "expected"), [(True, "fallback"), (False, "synthesize")])
def test_route_after_breaker(tripped, expected):
    assert route_after_breaker({"breaker_tripped": tripped}) == expected


@pytest.mark.parametrize(("tripped", "expected"), [(True, "fallback"), (False, END)])
def test_route_after_synthesize(tripped, expected):
    assert route_after_synthesize({"breaker_tripped": tripped}) == expected


def test_routing_tolerates_missing_keys():
    # analyze failing before it writes flags must not crash the edge function.
    assert route_after_analyze({}) == "check_breaker"
    assert route_after_breaker({}) == "synthesize"
