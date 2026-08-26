"""Post-processing for the §8 citation contract.

The synthesis prompt asks for `[n]` markers indexed against `state["documents"]`,
but a prompt is not a guarantee. A marker pointing past the end of the evidence
list is worse than no marker: it looks like a real source to the reader and
cannot be checked. Out-of-range markers are stripped rather than renumbered —
renumbering would silently re-point a citation at a document the model never
saw.
"""

from __future__ import annotations

import re

import structlog

from agent.state import Document

log = structlog.get_logger()

# [1] or grouped forms like [1, 2] / [1,2]
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def validate_citations(text: str, documents: list[Document]) -> str:
    """Remove `[n]` markers that do not point at a real document."""
    valid = range(1, len(documents) + 1)

    def replace(match: re.Match[str]) -> str:
        numbers = [int(part) for part in match.group(1).split(",")]
        kept = [n for n in numbers if n in valid]
        if len(kept) != len(numbers):
            log.warning(
                "invalid_citation_stripped",
                marker=match.group(0),
                documents=len(documents),
            )
        if not kept:
            return ""
        return "[" + ", ".join(str(n) for n in kept) + "]"

    cleaned = CITATION.sub(replace, text)
    # Stripping a marker can leave a double space or a space before punctuation.
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return re.sub(r" +([.,;:!?])", r"\1", cleaned)


def cited_indices(text: str) -> set[int]:
    """Every document index the text refers to."""
    return {int(part) for match in CITATION.finditer(text) for part in match.group(1).split(",")}
