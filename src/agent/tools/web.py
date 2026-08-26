"""Public web search. Mock implementation for v1.

Mirrors `rag.py` exactly apart from `source="web"` and real-looking URLs, so a
single node implementation drives both.
"""

from __future__ import annotations

import os

from agent.config import Settings
from agent.state import Document
from agent.tools.base import SearchFn, ToolError, maybe_fail, score_corpus

_FAKE_CORPUS: list[dict[str, str]] = [
    {
        "title": "LangGraph 0.2 release notes",
        "url": "https://example.com/langgraph/release-notes",
        "content": (
            "The latest LangGraph release adds durable checkpointing and improved "
            "conditional edge routing for stateful agents."
        ),
    },
    {
        "title": "Consumer refund law: 2026 update",
        "url": "https://example.com/news/refund-law-2026",
        "content": (
            "Recent legislation extends the statutory refund window for digital goods "
            "to 30 days, exceeding most published company policies."
        ),
    },
    {
        "title": "State of vector databases",
        "url": "https://example.com/blog/vector-databases",
        "content": (
            "A survey of current vector database options, covering pgvector, Qdrant "
            "and Chroma, with benchmarks for recall and query latency."
        ),
    },
]


def mock_web_search(query: str, top_k: int) -> list[Document]:
    """Search the fake web corpus. Raises ToolError when failure is injected."""
    maybe_fail("web")
    return score_corpus(query, _FAKE_CORPUS, source="web", id_prefix="web", with_urls=True)[:top_k]


def build_real_web_search(settings: Settings) -> SearchFn:
    """Tavily-backed web search.

    The key is checked once, here, rather than per turn: a missing key is a
    configuration error, not a transient tool fault the breaker should absorb.
    """
    if not os.environ.get("TAVILY_API_KEY"):
        raise ValueError("AGENT_TOOL_MODE=real requires TAVILY_API_KEY")

    from langchain_tavily import TavilySearch

    client = TavilySearch(max_results=settings.web_top_k)

    def search(query: str, top_k: int) -> list[Document]:
        try:
            payload = client.invoke({"query": query})
        except Exception as exc:
            raise ToolError(f"web search failed: {exc}") from exc
        results = payload.get("results", []) if isinstance(payload, dict) else []
        return [
            Document(
                id=f"web-{index}",
                source="web",
                title=str(result.get("title", "untitled")),
                url=result.get("url"),
                content=str(result.get("content", "")),
                score=round(float(result.get("score", 0.0)), 4),
            )
            for index, result in enumerate(results[:top_k])
        ]

    return search
