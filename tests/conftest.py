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
