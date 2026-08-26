# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --extra dev
uv run pytest                                    # offline; no API keys needed
uv run pytest tests/test_graph.py::test_rag_only_path   # single test
uv run ruff check . --fix && uv run ruff format .
uv run mypy

uv run agent-chat                                # interactive chat (mock tools)
uv run langgraph dev                             # Studio on :2024
uv run agent-ingest --reset                      # build the Chroma index for real mode
AGENT_TOOL_MODE=real uv run agent-chat
```

Everything runs through `uv run`. **The suite must stay offline and key-free** —
that is the whole point of the mock tools. Anything needing a key or a database
is `pytest.mark.skipif`-gated.

## Architecture

The LLM never selects tools. `analyze` emits a structured `RoutingDecision`
and explicit edges act on it:

```
analyze → [rag_search] → [web_search] → check_breaker → synthesize | fallback → END
```

This is the central design choice, and it is why tests can assert the exact
node path via `stream_mode="updates"` rather than hoping the model called the
right tool.

- **`tools/__init__.py`** is the only place that decides which implementation
  runs. `get_rag_search()` / `get_web_search()` return a `SearchFn`; nodes never
  learn whether they got a mock or the real thing.
- **`build_graph(settings)` binds nodes to a `Settings` via `functools.partial`.**
  Nothing reaches through a module-level singleton, which is what gives tests
  full isolation. Keep it that way.
- **Every node catches its own exceptions** and returns a state delta instead of
  raising. State is always written, the checkpoint stays consistent, and a bad
  turn routes to `fallback` rather than killing the graph.
- **Edge functions in `routing.py` are pure** and tolerate missing keys, because
  `analyze` can fail before it writes its flags.

## Things that will bite you

- **`documents` and `errors` are per-turn, but the checkpointer persists them.**
  They use the `append_or_reset` reducer and `analyze` writes the `RESET`
  sentinel at the start of every turn. **Do not switch them to `operator.add`,
  and do not try to clear them with `update_state(config, {"documents": []})`** —
  that write goes *through* the reducer, so `add(existing, [])` returns the list
  unchanged and evidence silently accumulates across turns. `tests/test_state.py`
  and `test_persistence.py::test_documents_do_not_leak_between_turns` guard this.
- **`total_tool_calls` must never be reset** — it is the thread-lifetime budget.
  Only `tool_calls` resets per turn.
- **`check_breaker` uses `>=`**, so the turn that *reaches* a budget is the one
  that trips, not the turn after it.
- **Routing errors are prefixed `analyze:` and excluded from the breaker's error
  count.** `analyze` already degraded to a safe default; that is not a tool
  outage. Preserve the prefix if you touch that code.
- **`Settings` is a pydantic model and unhashable.** Never use it as an
  `lru_cache` key — it raises `TypeError` at the first real-mode call, and
  neither ruff nor mypy catches it. Memoise on the specific fields instead
  (see `tools/__init__.py`).
- **`llm.py` serves two providers behind one factory** (`AGENT_LLM_PROVIDER`).
  Model defaults live in `config.DEFAULT_MODELS`, credentials in
  `config.API_KEY_VARS`; add a provider by extending both plus the branch in
  `build_llm`. A missing key raises there, *not* inside a node — a node would
  catch it and emit a generic fallback, hiding an unset env var.
- **OpenAI uses `max_completion_tokens`, Anthropic uses `max_tokens`.**
- **The model takes no `thinking` parameter (Anthropic).** Anthropic requires
  `temperature=1` when extended thinking is on, and routing runs at
  `temperature=0`. The two are mutually exclusive.
- **`ChatAnthropic` kwargs need the pydantic mypy plugin** (enabled in
  `pyproject.toml`). Without it mypy derives `__init__` from field aliases and
  reports every unset optional as a missing required argument.
- **Chroma returns a distance (lower is closer); `Document.score` is relevance
  (higher is better).** `build_real_rag_search` inverts it. Passing it through
  ranks the citation block backwards.
- **`mypy` targets 3.12 while the project targets 3.11**, because numpy's stubs
  use 3.12-only `type` statements. Ruff's `UP` rules still hold our code to 3.11.

## Testing approach

`tests/helpers.py` holds scripted fakes for the two LLM calls — each takes a
list of results and hands them out in order, raising if the graph consumes more
than the test scripted. Put an `Exception` in the list to drive a failure path.
The `patch_llm` fixture installs both.

Prefer asserting the **node path** and the resulting state over mocking the
graph. Tool failures are driven through `FAIL_NEXT` in `tools/base.py`, reset
by an autouse fixture — it is module-level state, so never let it leak.
