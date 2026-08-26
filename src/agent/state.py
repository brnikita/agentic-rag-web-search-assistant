"""Graph state definition."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Conversation state threaded through every node.

    `add_messages` is a reducer: nodes return only the *new* messages and
    LangGraph appends them (matching on id, so a message can be updated).
    """

    messages: Annotated[list[AnyMessage], add_messages]
