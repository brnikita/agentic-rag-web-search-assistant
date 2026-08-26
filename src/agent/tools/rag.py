"""Internal-docs search. Mock implementation for v1.

Deterministic and in-memory so the graph can be exercised without a vector
store. `real_rag_search` (M6) swaps in behind the same signature.
"""

from __future__ import annotations

from agent.state import Document
from agent.tools.base import maybe_fail, score_corpus

_FAKE_CORPUS: list[dict[str, str]] = [
    {
        "title": "Refund Policy v3",
        "url": "",
        "content": (
            "Refunds are issued within 14 days of purchase. Customers must supply the "
            "original order id. Refunds on annual plans are prorated from the "
            "cancellation date."
        ),
    },
    {
        "title": "Onboarding Guide",
        "url": "",
        "content": (
            "New employees must complete security training in their first week. "
            "Laptop provisioning is handled by IT and takes two business days."
        ),
    },
    {
        "title": "API Rate Limits",
        "url": "",
        "content": (
            "Internal APIs allow 1000 requests per minute per service. Exceeding the "
            "limit returns HTTP 429 with a Retry-After header."
        ),
    },
]


def mock_rag_search(query: str, top_k: int) -> list[Document]:
    """Search the fake internal corpus. Raises ToolError when failure is injected."""
    maybe_fail("rag")
    return score_corpus(query, _FAKE_CORPUS, source="internal", id_prefix="int", with_urls=False)[
        :top_k
    ]
