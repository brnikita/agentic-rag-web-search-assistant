# Agentic RAG + Web Search Assistant

A multi-turn LangGraph agent that routes each question to internal-docs RAG,
public web search, both, or neither — then answers with numbered citations.

The LLM never picks tools itself. A router node emits a structured decision and
explicit graph edges act on it, so the tool path is deterministic and directly
assertable in tests.

```
START ─▶ analyze ─┬─▶ rag_search ─┬─▶ web_search ─┐
                  │               └───────────────┼─▶ check_breaker ─┬─▶ synthesize ─▶ END
                  ├─▶ web_search ─────────────────┘                  └─▶ fallback   ─▶ END
                  └─────────────────────────────────────────────────▶ (no tools)
```

## Setup

```bash
uv sync --extra dev
cp .env.example .env      # add ANTHROPIC_API_KEY
```

Tools are **mocked by default** — deterministic in-memory corpora, no keys and
no infrastructure. The whole test suite runs offline.

## Run

```bash
uv run agent-chat         # interactive chat
uv run langgraph dev      # LangGraph Studio at http://localhost:2024
```

## Real tools

`AGENT_TOOL_MODE=real` swaps the mocks for a Chroma vector store over `docs/`
and Tavily web search, behind the same interface.

```bash
uv run agent-ingest --reset          # embed docs/ (local ONNX, no API key)
AGENT_TOOL_MODE=real uv run agent-chat
```

Web search additionally needs `TAVILY_API_KEY`; RAG needs no key at all.

## Develop

```bash
uv run pytest             # offline, no keys required
uv run ruff check . --fix && uv run ruff format .
uv run mypy
```

## Configuration

All settings take an `AGENT_` prefix — see `.env.example`. Provider keys
(`ANTHROPIC_API_KEY`, `TAVILY_API_KEY`) are deliberately *not* prefixed; the
SDKs read them directly.

| Setting | Default | Purpose |
| --- | --- | --- |
| `AGENT_TOOL_MODE` | `mock` | `mock` or `real` |
| `AGENT_MODEL_NAME` | `claude-sonnet-4-6` | Routing and synthesis |
| `AGENT_MAX_TOOL_CALLS_PER_TURN` | `4` | Circuit breaker, per turn |
| `AGENT_MAX_TOOL_CALLS_PER_THREAD` | `40` | Circuit breaker, thread lifetime |
| `AGENT_MAX_ERROR_COUNT` | `2` | Tool errors before giving up |
| `AGENT_CHECKPOINTER` | `memory` | `memory` or `postgres` |

## Failure behaviour

Every node catches its own exceptions and returns a state delta, so state is
always written and the graph never dies mid-turn.

| Failure | Response |
| --- | --- |
| Transient tool error | Retried with exponential backoff |
| Tool still failing | Recorded; the graph continues on partial evidence |
| Router LLM error | Safe default — internal docs only |
| Too many tool calls or errors | Breaker trips → honest fallback message |
| Synthesis LLM error | Breaker trips → fallback |
| Routing loop | `GraphRecursionError` caught and reported |
