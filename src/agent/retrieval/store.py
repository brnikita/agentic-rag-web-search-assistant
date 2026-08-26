"""Chroma vector store backed by local FastEmbed embeddings.

Embeddings run on-device (ONNX, no API key), so the docs corpus never leaves
the machine and the store works offline once the model is cached.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from agent.config import Settings, get_settings
from agent.retrieval.embeddings import FastEmbedEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Local embedding model. Cached: loading the ONNX model is expensive."""
    settings = get_settings()
    return FastEmbedEmbeddings(model_name=settings.embedding_model)


def get_vector_store(settings: Settings | None = None) -> Chroma:
    """Open (or create) the persistent Chroma collection.

    Safe to call when the store is empty; searches simply return nothing until
    `agent-ingest` has run.
    """
    settings = settings or get_settings()
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.vector_store_dir),
    )
