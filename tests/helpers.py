"""Test doubles for the two LLM calls the graph makes.

Scripted rather than mocked: each fake takes a list of results and hands them
out in order, raising if a test scripts fewer than the graph consumes. An
`Exception` in the list is raised instead of returned, which is how the failure
paths are driven.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agent.schemas import RoutingDecision


class FakeRouter:
    """Stands in for `llm.with_structured_output(RoutingDecision)`."""

    def __init__(self, decisions):
        self._decisions = list(decisions)

    def invoke(self, messages, **kwargs):
        if not self._decisions:
            raise AssertionError("router called more times than the test scripted")
        result = self._decisions.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeChat:
    """Stands in for the synthesis model. Records the messages it was sent."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.calls: list = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        if not self._answers:
            raise AssertionError("synthesize called more times than the test scripted")
        result = self._answers.pop(0)
        if isinstance(result, Exception):
            raise result
        return AIMessage(content=result)


def decision(*, rag: bool = True, web: bool = False, query: str = "rewritten") -> RoutingDecision:
    return RoutingDecision(rewritten_query=query, needs_rag=rag, needs_web=web, reasoning="because")
