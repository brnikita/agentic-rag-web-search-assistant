"""Structured-output models for LLM calls."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    """What `analyze` asks the LLM to produce for each user turn."""

    rewritten_query: str = Field(
        description="Standalone search query resolving pronouns from chat history"
    )
    needs_rag: bool = Field(description="True if internal company docs may answer this")
    needs_web: bool = Field(
        description="True only if the answer needs public, recent, or external information"
    )
    reasoning: str = Field(description="One sentence")
