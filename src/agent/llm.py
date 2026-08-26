"""Chat model construction."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from agent.config import Settings, get_settings


def build_llm(settings: Settings | None = None) -> ChatAnthropic:
    """Build the Claude chat model from settings.

    Adaptive thinking lets Claude decide how much to reason per turn; it is on
    by default and is what the current Claude models expect (the older
    fixed `budget_tokens` form is rejected by them).
    """
    settings = settings or get_settings()

    kwargs: dict = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
    }
    if settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key
    if settings.thinking == "adaptive":
        kwargs["thinking"] = {"type": "adaptive"}

    return ChatAnthropic(**kwargs)
