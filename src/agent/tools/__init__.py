"""Tool registry.

`build_tools()` is the single place that decides what the agent can do; the
graph binds whatever it returns. Add a tool here and it is wired everywhere.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agent.config import Settings, get_settings
from agent.tools.docs_search import search_internal_docs
from agent.tools.web_search import build_web_search_tool


def build_tools(settings: Settings | None = None) -> list[BaseTool]:
    """All tools available to the agent.

    Web search is omitted when no Tavily key is configured, so the agent
    degrades to docs-only instead of failing every turn on a missing key.
    """
    settings = settings or get_settings()

    tools: list[BaseTool] = [search_internal_docs]
    if settings.tavily_api_key:
        tools.append(build_web_search_tool(settings))
    return tools


__all__ = ["build_tools", "build_web_search_tool", "search_internal_docs"]
