# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --extra dev                  # install (creates .venv)
uv run agent-ingest --reset          # rebuild the docs vector store
uv run agent-chat                    # interactive terminal chat
uv run langgraph dev                 # LangGraph Studio on :2024

uv run pytest                        # full suite
uv run pytest tests/test_graph.py::test_graph_runs_tool_then_answers   # single test
uv run ruff check . --fix && uv run ruff format .
uv run mypy
```

Everything runs through `uv run`; there is no `make`/`tox` layer. The whole test
suite runs offline — no `ANTHROPIC_API_KEY` or `TAVILY_API_KEY` required.

## Architecture

A ReAct loop: `START → agent → (tools → agent)* → END`. The `agent` node calls
Claude with tools bound; `tools_condition` routes back to `tools` whenever the
reply carries tool calls, and to `END` otherwise.

The pieces that matter, and how they connect:

- **`tools/__init__.py: build_tools()` is the single source of truth for the
  agent's capabilities.** The graph binds whatever it returns — to add a tool,
  write it under `tools/` and register it there. Nothing else changes.
- **`build_graph()` takes a `Settings` and returns a fresh compiled graph.**
  Nothing is a module-level singleton, which is what lets tests build isolated
  graphs against a temp vector store. Keep it that way.
- **`config.py: get_settings()` is an `lru_cache`d singleton** read from `.env`.
  Tests clear it via an autouse fixture. Every module takes `settings` as an
  optional argument and falls back to `get_settings()` — preserve that pattern
  so tests can inject.
- **The system prompt is prepended per LLM call, not stored in state**, so
  editing `prompts.py` takes effect on existing checkpointed threads.

### The two tools

`search_internal_docs` is local RAG: `docs/` → chunked by
`retrieval/ingest.py` → embedded by a local ONNX model (`retrieval/embeddings.py`)
→ persisted in Chroma under `.chroma/`. It is a plain `@tool`, so it reads
settings from the cached singleton; tests monkeypatch `docs_search.get_settings`
to redirect it.

`web_search` wraps Tavily and is built by a **factory**, not defined at module
level — importing the package must not require a Tavily key. `build_tools()`
omits it entirely when the key is absent, so the agent degrades to docs-only
rather than failing every turn.

The asymmetry between the two (decorator vs. factory) is deliberate: a tool
needing per-instance config gets a factory, a self-contained one gets `@tool`.

## Things that will bite you

- **`langgraph dev` and LangGraph Platform inject their own checkpointer**, and
  compiling with one already attached is an error there. That is why
  `make_graph()` — the entry point named in `langgraph.json` — passes
  `checkpointer=False`. `build_graph()` otherwise defaults to an in-process
  `MemorySaver`; swap it for a Postgres/SQLite saver for anything real.
- **Re-ingesting without `--reset` leaves stale chunks.** Chroma has no
  upsert-by-source primitive here, so edited docs accumulate duplicates.
- **The model is `claude-opus-5` with `thinking={"type": "adaptive"}`.** The
  older fixed `budget_tokens` form is *rejected* by current Claude models — do
  not reintroduce it. Set `THINKING=off` in `.env` to disable.
- **The first embedding call downloads an ONNX model** (~130 MB, cached
  thereafter). The session-scoped `_warm_embeddings` fixture exists to pay that
  once; depend on it in any test that touches the vector store.
- **`mypy` targets 3.12 while the project targets 3.11.** numpy's stubs use
  3.12-only `type` statements that mypy refuses to parse under a lower target.
  Ruff's `UP` rules still enforce 3.11 compatibility on our own code.

## Testing approach

`test_graph.py` fakes the chat model — a `GenericFakeChatModel` subclass whose
`bind_tools` returns `self` — and monkeypatches `graph_module.build_llm`. That
exercises real graph routing, the real `ToolNode`, and the real checkpointer
with scripted `AIMessage`s. Follow this rather than mocking the graph itself;
the routing is the part worth testing.
