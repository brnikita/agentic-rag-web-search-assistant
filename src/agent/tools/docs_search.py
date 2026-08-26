"""Internal docs search tool (RAG over the local vector store)."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.tools import tool

from agent.config import get_settings
from agent.retrieval.store import get_vector_store

_EMPTY = (
    "The internal documentation index is empty. Run `uv run agent-ingest` to build it, "
    "then retry. Do not treat this as evidence that the topic is undocumented."
)


def _format(documents: list[tuple[Document, float]]) -> str:
    """Render hits as a citable block. Scores are Chroma distances (lower = closer)."""
    parts = []
    for doc, score in documents:
        source = doc.metadata.get("source", "unknown")
        parts.append(f'<doc source="{source}" score="{score:.4f}">\n{doc.page_content}\n</doc>')
    return "\n\n".join(parts)


@tool(parse_docstring=True)
def search_internal_docs(query: str) -> str:
    """Search the organisation's internal documentation.

    Authoritative for our own products, architecture, policies, and processes.
    Returns the most relevant passages with their source paths, which you should
    cite. Prefer natural-language questions over keywords.

    Args:
        query: The question or topic to look up in the internal docs.
    """
    settings = get_settings()
    store = get_vector_store(settings)
    hits = store.similarity_search_with_score(query, k=settings.docs_search_k)
    if not hits:
        return _EMPTY
    return _format(hits)
