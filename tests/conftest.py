from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import Settings, get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Keep the cached singleton from leaking configuration between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Deterministic settings: mock tools, isolated vector store."""
    return Settings(
        tool_mode="mock",
        docs_dir=PROJECT_ROOT / "docs",
        vector_store_dir=tmp_path / "chroma",
        collection_name="test_docs",
    )


@pytest.fixture(autouse=True)
def _reset_injected_failures():
    """Injected tool failures are module-level; never let one leak into the next test."""
    from agent.tools.base import reset_failures

    reset_failures()
    yield
    reset_failures()


@pytest.fixture
def patch_llm(monkeypatch):
    """Replace both LLM calls in `agent.nodes`. Returns the fake chat model."""

    def _patch(*, decisions=(), answers=()):
        import agent.nodes as nodes_module
        from helpers import FakeChat, FakeRouter

        chat = FakeChat(answers)
        router = FakeRouter(decisions)
        monkeypatch.setattr(nodes_module, "build_llm", lambda settings: chat)
        monkeypatch.setattr(nodes_module, "build_router", lambda llm: router)
        return chat

    return _patch
