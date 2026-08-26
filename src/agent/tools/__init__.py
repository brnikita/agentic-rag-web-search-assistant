"""Tool selection.

The single place that decides which implementation the graph runs against.
Both modes satisfy `SearchFn`, so nodes never learn which one they got.

Real builders are memoised because each opens an expensive resource — an ONNX
embedding model, an HTTP client. They are keyed on the settings fields that
actually affect construction rather than on the Settings object itself, which
is a pydantic model and therefore unhashable.
"""

from __future__ import annotations

from agent.config import Settings, get_settings
from agent.tools.base import FAIL_NEXT, SearchFn, ToolError, guarded, reset_failures
from agent.tools.rag import build_real_rag_search, mock_rag_search
from agent.tools.web import build_real_web_search, mock_web_search

_rag_cache: dict[tuple, SearchFn] = {}
_web_cache: dict[tuple, SearchFn] = {}


def clear_tool_cache() -> None:
    """Drop memoised real tools. Call when settings change (tests, reconfig)."""
    _rag_cache.clear()
    _web_cache.clear()


def get_rag_search(settings: Settings | None = None) -> SearchFn:
    settings = settings or get_settings()
    if settings.tool_mode != "real":
        return mock_rag_search
    key = (str(settings.vector_store_dir), settings.collection_name, settings.embedding_model)
    if key not in _rag_cache:
        _rag_cache[key] = build_real_rag_search(settings)
    return _rag_cache[key]


def get_web_search(settings: Settings | None = None) -> SearchFn:
    settings = settings or get_settings()
    if settings.tool_mode != "real":
        return mock_web_search
    key = (settings.web_top_k,)
    if key not in _web_cache:
        _web_cache[key] = build_real_web_search(settings)
    return _web_cache[key]


__all__ = [
    "FAIL_NEXT",
    "SearchFn",
    "ToolError",
    "clear_tool_cache",
    "get_rag_search",
    "get_web_search",
    "guarded",
    "mock_rag_search",
    "mock_web_search",
    "reset_failures",
]
