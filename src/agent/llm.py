"""Chat model construction."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from agent.config import Settings, get_settings
from agent.schemas import RoutingDecision


def build_llm(settings: Settings | None = None) -> BaseChatModel:
    """The synthesis model.

    No `thinking` parameter: Anthropic requires temperature=1 when extended
    thinking is enabled, and routing is configured at temperature 0 for
    reproducibility.
    """
    settings = settings or get_settings()
    return ChatAnthropic(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def build_router(llm: BaseChatModel):
    """The routing model, constrained to emit a RoutingDecision."""
    return llm.with_structured_output(RoutingDecision)
