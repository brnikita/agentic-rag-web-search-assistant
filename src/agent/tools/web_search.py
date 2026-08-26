"""Public web search tool, backed by Tavily."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch

from agent.config import Settings, get_settings


def build_web_search_tool(settings: Settings | None = None) -> BaseTool:
    """Build the Tavily search tool.

    Kept as a factory rather than a module-level singleton so importing the
    package does not require a Tavily key (tests and docs-only runs).
    """
    settings = settings or get_settings()

    kwargs: dict = {
        "max_results": settings.web_search_max_results,
        "name": "web_search",
        "description": (
            "Search the public web for current events, third-party products, and "
            "anything not covered by the internal documentation. Returns result "
            "snippets with their URLs, which you should cite."
        ),
    }
    if settings.tavily_api_key:
        kwargs["tavily_api_key"] = settings.tavily_api_key

    return TavilySearch(**kwargs)
