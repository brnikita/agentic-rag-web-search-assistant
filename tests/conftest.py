from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent.config import Settings, get_settings
from agent.retrieval.store import get_embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a throwaway vector store over the real docs corpus."""
    return Settings(
        anthropic_api_key="test-key",
        tavily_api_key=None,
        docs_dir=PROJECT_ROOT / "docs",
        vector_store_dir=tmp_path / "chroma",
        collection_name="test_docs",
    )


@pytest.fixture(scope="session")
def _warm_embeddings():
    # Downloads the ONNX model once; the first call is slow.
    get_embeddings()


@pytest.fixture(autouse=True)
def _cleanup(tmp_path: Path):
    yield
    shutil.rmtree(tmp_path / "chroma", ignore_errors=True)
