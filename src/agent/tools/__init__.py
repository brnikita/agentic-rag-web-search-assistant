"""Tool selection.

`get_rag_search` / `get_web_search` are the single place that decides which
implementation the graph runs against. v1 ships mocks; M6 adds the real
Tavily and Chroma adapters behind `settings.tool_mode`.
"""

from __future__ import annotations

from agent.config import Settings, get_settings
from agent.tools.base import FAIL_NEXT, SearchFn, ToolError, guarded, reset_failures
from agent.tools.rag import mock_rag_search
from agent.tools.web import mock_web_search


def get_rag_search(settings: Settings | None = None) -> SearchFn:
    settings = settings or get_settings()
    return mock_rag_search


def get_web_search(settings: Settings | None = None) -> SearchFn:
    settings = settings or get_settings()
    return mock_web_search


__all__ = [
    "FAIL_NEXT",
    "SearchFn",
    "ToolError",
    "get_rag_search",
    "get_web_search",
    "guarded",
    "mock_rag_search",
    "mock_web_search",
    "reset_failures",
]
