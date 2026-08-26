"""Mock tool behaviour: scoring, limits, retries, timeouts, failure injection."""

from __future__ import annotations

import time

import pytest

from agent.tools import ToolError, guarded, mock_rag_search, mock_web_search
from agent.tools.base import FAIL_NEXT, call_with_timeout


def test_rag_ranks_matching_document_first():
    docs = mock_rag_search("what is the refund policy?", top_k=3)
    assert docs[0]["title"] == "Refund Policy v3"
    assert docs[0]["score"] > docs[-1]["score"]


def test_web_ranks_matching_document_first():
    docs = mock_web_search("latest news about refund law", top_k=3)
    assert docs[0]["title"] == "Consumer refund law: 2026 update"


def test_top_k_is_honoured():
    assert len(mock_rag_search("refund", top_k=1)) == 1
    assert len(mock_rag_search("refund", top_k=99)) == 3


def test_sources_and_urls_differ_by_tool():
    rag = mock_rag_search("refund", top_k=1)[0]
    web = mock_web_search("refund", top_k=1)[0]
    assert rag["source"] == "internal" and rag["url"] is None
    assert web["source"] == "web" and web["url"].startswith("https://")


def test_ranking_is_deterministic():
    # No term matches, so every score ties — ordering must still be stable.
    first = [d["id"] for d in mock_rag_search("zzz", top_k=3)]
    second = [d["id"] for d in mock_rag_search("zzz", top_k=3)]
    assert first == second


def test_injected_failure_is_retried_then_succeeds(settings):
    FAIL_NEXT["rag"] = 2  # fewer than retry_attempts=3
    docs = guarded(mock_rag_search, "refund", 3, settings)
    assert docs and FAIL_NEXT["rag"] == 0


def test_retries_are_exhausted_then_raised(settings):
    FAIL_NEXT["rag"] = 5  # more than retry_attempts
    with pytest.raises(ToolError, match="simulated rag outage"):
        guarded(mock_rag_search, "refund", 3, settings)
    assert FAIL_NEXT["rag"] == 5 - settings.retry_attempts


def test_retry_count_follows_settings(settings):
    settings = settings.model_copy(update={"retry_attempts": 1})
    FAIL_NEXT["web"] = 5
    with pytest.raises(ToolError):
        guarded(mock_web_search, "news", 3, settings)
    assert FAIL_NEXT["web"] == 4  # exactly one attempt consumed


def test_timeout_raises_tool_error():
    def slow(query: str, top_k: int):
        time.sleep(1.0)
        return []

    with pytest.raises(ToolError, match="timed out"):
        call_with_timeout(slow, 0.05, "q", 1)


def test_timeout_is_retried_by_guarded(settings):
    settings = settings.model_copy(update={"tool_timeout_s": 0.05, "retry_attempts": 2})
    calls = {"n": 0}

    def slow(query: str, top_k: int):
        calls["n"] += 1
        time.sleep(1.0)
        return []

    with pytest.raises(ToolError, match="timed out"):
        guarded(slow, "q", 1, settings)
    assert calls["n"] == 2  # timeout feeds the retry path
