"""Central configuration, loaded from the environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Every knob the agent reads. Override any of these in `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM -------------------------------------------------------------
    anthropic_api_key: str | None = None
    model: str = "claude-opus-5"
    max_tokens: int = 8000
    # "adaptive" turns on adaptive thinking; "off" disables it entirely.
    thinking: str = "adaptive"

    # --- Web search ------------------------------------------------------
    tavily_api_key: str | None = None
    web_search_max_results: int = 5

    # --- Internal docs search -------------------------------------------
    docs_dir: Path = Field(default=PROJECT_ROOT / "docs")
    vector_store_dir: Path = Field(default=PROJECT_ROOT / ".chroma")
    collection_name: str = "internal_docs"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    docs_search_k: int = 4

    # --- Graph -----------------------------------------------------------
    recursion_limit: int = 25


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call `get_settings.cache_clear()` in tests."""
    return Settings()
