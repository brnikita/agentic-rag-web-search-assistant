"""Shared plumbing for the search tools: errors, timeouts, retries, injection.

Both tools expose the same signature — `(query: str, top_k: int) -> list[Document]`
— so the graph nodes are identical apart from which callable they hand to
`guarded()`.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.config import Settings
from agent.state import Document

SearchFn = Callable[[str, int], list[Document]]


class ToolError(Exception):
    """A search failed in a way worth retrying."""


# Test hook: number of forced consecutive failures per tool. Tests set these
# to drive the retry and circuit-breaker paths deterministically.
FAIL_NEXT: dict[str, int] = {"rag": 0, "web": 0}


def reset_failures() -> None:
    """Clear every injected failure. Call between tests."""
    for key in FAIL_NEXT:
        FAIL_NEXT[key] = 0


def maybe_fail(name: str) -> None:
    """Raise if a failure has been injected for `name`, consuming one."""
    if FAIL_NEXT.get(name, 0) > 0:
        FAIL_NEXT[name] -= 1
        raise ToolError(f"simulated {name} outage")


def call_with_timeout(fn: SearchFn, timeout_s: float, query: str, top_k: int) -> list[Document]:
    """Run `fn` and give up after `timeout_s`.

    A thread is used rather than `signal.alarm`, which is POSIX-only. Python
    cannot kill a running thread, so on timeout the worker is abandoned rather
    than stopped — `shutdown(wait=False)` returns immediately instead of
    blocking until the straggler finishes, which would defeat the timeout.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(fn, query, top_k)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError as exc:
            future.cancel()
            raise ToolError(f"timed out after {timeout_s}s") from exc
    finally:
        pool.shutdown(wait=False)


def guarded(fn: SearchFn, query: str, top_k: int, settings: Settings) -> list[Document]:
    """Run a search with a per-attempt timeout and exponential-backoff retries.

    Retry limits are read from `settings` at call time rather than baked into a
    decorator at import, so tests can vary them.
    """
    retryer = Retrying(
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_exponential(min=0.2, max=2),
        retry=retry_if_exception_type(ToolError),
        reraise=True,
    )
    return retryer(call_with_timeout, fn, settings.tool_timeout_s, query, top_k)


def score_corpus(
    query: str,
    corpus: list[dict[str, str]],
    *,
    source: str,
    id_prefix: str,
    with_urls: bool,
) -> list[Document]:
    """Score a fake corpus by naive term overlap, best first.

    Deterministic: ties break on id, so repeated runs return the same order.
    """
    terms = query.lower().split()
    documents: list[Document] = []
    for index, entry in enumerate(corpus):
        haystack = f"{entry['title']} {entry['content']}".lower()
        hits = sum(term in haystack for term in terms)
        documents.append(
            Document(
                id=f"{id_prefix}-{index}",
                source=source,  # type: ignore[typeddict-item]
                title=entry["title"],
                url=entry["url"] if with_urls else None,
                content=entry["content"],
                score=round(hits / max(len(terms), 1), 2),
            )
        )
    return sorted(documents, key=lambda d: (-d["score"], d["id"]))
