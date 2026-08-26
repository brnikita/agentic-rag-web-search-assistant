"""Tool contract tests. These hit the local vector store but never the network."""

from __future__ import annotations

from pathlib import Path

import agent.tools.docs_search as docs_search
from agent.retrieval.ingest import ingest, load_documents
from agent.tools import build_tools
from agent.tools.docs_search import search_internal_docs


def test_registry_omits_web_search_without_key(settings):
    names = {t.name for t in build_tools(settings)}
    assert names == {"search_internal_docs"}


def test_registry_includes_web_search_with_key(settings):
    settings = settings.model_copy(update={"tavily_api_key": "tvly-test"})
    names = {t.name for t in build_tools(settings)}
    assert names == {"search_internal_docs", "web_search"}


def test_load_documents_uses_relative_sources(settings):
    docs = load_documents(settings.docs_dir)
    assert docs, "sample corpus should not be empty"
    assert all(not Path(d.metadata["source"]).is_absolute() for d in docs)
    assert {"architecture.md", "deployment.md"} <= {d.metadata["source"] for d in docs}


def test_docs_search_returns_cited_passages(settings, monkeypatch, _warm_embeddings):
    monkeypatch.setattr(docs_search, "get_settings", lambda: settings)
    assert ingest(settings, reset=True) > 0

    result = search_internal_docs.invoke({"query": "When are production deploys blocked?"})

    assert "source=" in result
    assert "deployment.md" in result
    assert "Friday" in result


def test_docs_search_reports_empty_index(settings, monkeypatch, _warm_embeddings):
    monkeypatch.setattr(docs_search, "get_settings", lambda: settings)
    result = search_internal_docs.invoke({"query": "anything"})
    assert "empty" in result.lower()
