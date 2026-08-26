"""Provider selection: defaults, overrides, and credential checks.

Live calls are gated on a key being present; the wiring itself is always
tested, since that is what breaks silently when a provider is added.
"""

from __future__ import annotations

import os

import pytest

from agent.config import DEFAULT_MODELS
from agent.llm import build_llm, build_router


def test_default_provider_is_anthropic(settings):
    assert settings.llm_provider == "anthropic"
    assert settings.resolved_model() == DEFAULT_MODELS["anthropic"]


def test_each_provider_has_its_own_default_model(settings):
    openai = settings.model_copy(update={"llm_provider": "openai"})
    assert openai.resolved_model() == "gpt-5.6-luna"
    assert openai.resolved_model() != settings.resolved_model()


def test_explicit_model_overrides_the_provider_default(settings):
    pinned = settings.model_copy(update={"llm_provider": "openai", "model_name": "gpt-4"})
    assert pinned.resolved_model() == "gpt-4"


def test_each_provider_reads_its_own_key_var(settings):
    assert settings.api_key_var() == "ANTHROPIC_API_KEY"
    assert settings.model_copy(update={"llm_provider": "openai"}).api_key_var() == "OPENAI_API_KEY"


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_missing_key_fails_loudly(settings, monkeypatch, provider):
    """Must raise at construction, not inside a node.

    A node would catch it and emit a generic fallback message, hiding the fact
    that the real problem is an unset environment variable.
    """
    target = settings.model_copy(update={"llm_provider": provider})
    monkeypatch.delenv(target.api_key_var(), raising=False)

    with pytest.raises(ValueError, match=target.api_key_var()):
        build_llm(target)


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
def test_openai_model_answers(settings):
    llm = build_llm(settings.model_copy(update={"llm_provider": "openai"}))
    assert llm.invoke("Reply with the single word OK").content.strip().upper().startswith("OK")


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
def test_openai_router_emits_a_routing_decision(settings):
    from agent.schemas import RoutingDecision

    llm = build_llm(settings.model_copy(update={"llm_provider": "openai"}))
    decision = build_router(llm).invoke("What is our refund policy?")

    assert isinstance(decision, RoutingDecision)
    assert decision.needs_rag is True  # an internal-policy question
