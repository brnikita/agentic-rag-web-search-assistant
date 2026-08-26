"""Mode selection, and the real adapters' contract.

The live paths need keys and a built index, so they are gated. What is always
tested is the wiring: that mode selection picks the right callable, and that
misconfiguration fails loudly at construction rather than silently per turn.
"""

from __future__ import annotations

import os

import pytest

from agent.tools import (
    clear_tool_cache,
    get_rag_search,
    get_web_search,
    mock_rag_search,
    mock_web_search,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_tool_cache()
    yield
    clear_tool_cache()


def test_mock_mode_selects_the_mocks(settings):
    assert get_rag_search(settings) is mock_rag_search
    assert get_web_search(settings) is mock_web_search


def test_real_web_search_requires_a_key(settings, monkeypatch):
    """A missing key is a config error, not a transient fault to retry."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    real = settings.model_copy(update={"tool_mode": "real"})

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        get_web_search(real)


def test_real_rag_is_memoised(settings, monkeypatch):
    import agent.tools as tools_module

    calls = {"n": 0}

    def fake_builder(_settings):
        calls["n"] += 1
        return mock_rag_search

    monkeypatch.setattr(tools_module, "build_real_rag_search", fake_builder)
    real = settings.model_copy(update={"tool_mode": "real"})

    get_rag_search(real)
    get_rag_search(real)

    assert calls["n"] == 1  # the ONNX model loads once, not per turn


def test_settings_are_not_used_as_a_cache_key(settings, monkeypatch):
    """Settings is a pydantic model and unhashable; keying on it would raise."""
    import agent.tools as tools_module

    monkeypatch.setattr(tools_module, "build_real_rag_search", lambda _s: mock_rag_search)
    real = settings.model_copy(update={"tool_mode": "real"})

    assert get_rag_search(real) is mock_rag_search  # no TypeError


@pytest.mark.skipif(
    not os.environ.get("TAVILY_API_KEY"), reason="needs TAVILY_API_KEY for a live search"
)
def test_real_web_search_returns_cited_documents(settings):
    real = settings.model_copy(update={"tool_mode": "real"})
    docs = get_web_search(real)("what is langgraph", 3)

    assert docs
    assert all(d["source"] == "web" and d["url"] for d in docs)


@pytest.mark.skipif(
    not (PROJECT_CHROMA := os.environ.get("AGENT_VECTOR_STORE_DIR")),
    reason="needs an ingested Chroma index; set AGENT_VECTOR_STORE_DIR to run",
)
def test_real_rag_search_returns_internal_documents(settings):
    real = settings.model_copy(update={"tool_mode": "real", "vector_store_dir": PROJECT_CHROMA})
    docs = get_rag_search(real)("deployment rollback", 3)

    assert docs
    assert all(d["source"] == "internal" and d["score"] > 0 for d in docs)
