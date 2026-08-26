"""Graph wiring: a ReAct-style loop over the tool registry.

    START -> agent -> (tools -> agent)* -> END

`agent` calls Claude with the tools bound. `tools_condition` routes to the
`tools` node whenever the reply contains tool calls, and to END otherwise, so
the loop runs until Claude answers without asking for a tool.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import Settings, get_settings
from agent.llm import build_llm
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from agent.tools import build_tools


def build_graph(settings: Settings | None = None, *, checkpointer=None):
    """Compile the agent graph.

    Args:
        settings: Overrides the process-wide settings (useful in tests).
        checkpointer: Persistence backend. Defaults to an in-process
            `MemorySaver`; swap for a Postgres/SQLite saver in production.
            Pass `False` to compile without one — LangGraph Platform and
            `langgraph dev` inject their own, and compiling with a checkpointer
            already attached is an error there.
    """
    settings = settings or get_settings()
    tools = build_tools(settings)
    llm = build_llm(settings).bind_tools(tools)

    def agent(state: AgentState) -> dict:
        # The system prompt is prepended per call rather than stored in state,
        # so editing prompts.py takes effect on existing threads too.
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    if checkpointer is False:
        return builder.compile()
    return builder.compile(checkpointer=checkpointer or MemorySaver())


def make_graph():
    """Entry point for `langgraph dev` / LangGraph Platform (see langgraph.json).

    The platform supplies its own checkpointer, hence `checkpointer=False`.
    """
    return build_graph(checkpointer=False)
