"""Chat model construction.

Two providers behind one factory. The graph never learns which one it got —
`analyze` and `synthesize` just call `build_llm(settings)`.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from agent.config import Settings, get_settings
from agent.schemas import RoutingDecision


def build_llm(settings: Settings | None = None) -> BaseChatModel:
    """The chat model for both routing and synthesis.

    A missing key raises here rather than inside a node, where the node's own
    exception handling would swallow it into a generic fallback message and
    hide the real cause.
    """
    settings = settings or get_settings()
    key_var = settings.api_key_var()
    if not os.environ.get(key_var):
        raise ValueError(f"AGENT_LLM_PROVIDER={settings.llm_provider} requires {key_var} to be set")

    model = settings.resolved_model()

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=settings.temperature,
            max_completion_tokens=settings.max_tokens,
        )

    from langchain_anthropic import ChatAnthropic

    # No `thinking`: Anthropic requires temperature=1 when it is enabled, and
    # routing runs at temperature 0.
    return ChatAnthropic(
        model=model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def build_router(llm: BaseChatModel):
    """The routing model, constrained to emit a RoutingDecision."""
    return llm.with_structured_output(RoutingDecision)
