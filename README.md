# provectus-agentic-task

A LangGraph agent boilerplate with two tools: **public web search** (Tavily) and
**internal docs search** (local RAG over a Chroma vector store).

## Setup

```bash
uv sync --extra dev
cp .env.example .env      # add ANTHROPIC_API_KEY, optionally TAVILY_API_KEY
uv run agent-ingest       # embed docs/ into the vector store
```

Only `ANTHROPIC_API_KEY` is required. Without `TAVILY_API_KEY` the agent still
runs — the web search tool is simply not registered, and it answers from the
internal docs alone. Embeddings run locally (ONNX via `fastembed`), so no third
key is needed and the docs corpus never leaves the machine.

## Run

```bash
uv run agent-chat         # interactive terminal chat
uv run langgraph dev      # LangGraph Studio at http://localhost:2024
```

## Develop

```bash
uv run pytest             # full suite; no API key needed
uv run ruff check . --fix
uv run ruff format .
uv run mypy
```

## Adding your own docs

Drop `.md`, `.txt`, or `.rst` files anywhere under `docs/` and re-run
`uv run agent-ingest --reset`. The `--reset` flag drops the existing collection
first — without it, edited files leave stale chunks behind.

## Adding a tool

Write it in `src/agent/tools/`, then register it in `build_tools()` in
`src/agent/tools/__init__.py`. The graph binds whatever that function returns,
so nothing else needs to change.
