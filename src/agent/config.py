"""Settings, loaded from the environment / `.env` with an `AGENT_` prefix.

Provider API keys are deliberately absent: the `AGENT_` prefix would rename
`ANTHROPIC_API_KEY` to `AGENT_ANTHROPIC_API_KEY`, and both `ChatAnthropic` and
`TavilySearch` already read their own unprefixed variables from the environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM -------------------------------------------------------------
    model_name: str = "claude-sonnet-4-6"
    # 0 for reproducible routing. Note this rules out extended thinking:
    # Anthropic requires temperature=1 whenever thinking is enabled.
    temperature: float = 0.0
    max_tokens: int = 8000

    # --- Circuit breakers -------------------------------------------------
    max_tool_calls_per_turn: int = 4
    max_tool_calls_per_thread: int = 40
    max_error_count: int = 2
    tool_timeout_s: float = 10.0
    retry_attempts: int = 3

    # --- Retrieval --------------------------------------------------------
    tool_mode: Literal["mock", "real"] = "mock"
    rag_top_k: int = 4
    web_top_k: int = 3

    # --- Real-mode retrieval (unused when tool_mode == "mock") -----------
    docs_dir: Path = Field(default=PROJECT_ROOT / "docs")
    vector_store_dir: Path = Field(default=PROJECT_ROOT / ".chroma")
    collection_name: str = "internal_docs"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Graph ------------------------------------------------------------
    checkpointer: Literal["memory", "postgres"] = "memory"
    postgres_dsn: str | None = None
    recursion_limit: int = 25


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call `get_settings.cache_clear()` in tests."""
    return Settings()
