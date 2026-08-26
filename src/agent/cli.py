"""Interactive terminal chat against the compiled graph.

Run with `uv run agent-chat`. Streams tool calls and the final answer; the
conversation shares one thread id, so the checkpointer keeps context.
"""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.config import get_settings
from agent.graph import build_graph


def _render(message) -> None:
    if isinstance(message, ToolMessage):
        preview = str(message.content).replace("\n", " ")[:160]
        print(f"  \033[2m<- {message.name}: {preview}...\033[0m")
    elif isinstance(message, AIMessage):
        for call in message.tool_calls:
            print(f"  \033[2m-> {call['name']}({call['args']})\033[0m")
        if text := message.text():
            print(f"\n{text}\n")


def main() -> None:
    settings = get_settings()
    graph = build_graph(settings)
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": settings.recursion_limit,
    }

    print(f"Agent ready ({settings.model}). Ctrl-C or an empty line to quit.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return

        for chunk in graph.stream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            stream_mode="values",
        ):
            _render(chunk["messages"][-1])


if __name__ == "__main__":
    main()
