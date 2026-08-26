"""Interactive terminal chat. Run with `uv run agent-chat`."""

from __future__ import annotations

import logging
import uuid

from agent.config import get_settings
from agent.logging import configure_logging
from agent.runtime import Agent


def main() -> None:
    settings = get_settings()
    # Console rendering here, not JSON: this is a human-facing session.
    configure_logging(level=logging.WARNING, json_output=False)

    agent = Agent(settings)
    thread_id = str(uuid.uuid4())

    print(
        f"Agent ready ({settings.resolved_model()} via {settings.llm_provider}, "
        f"tools={settings.tool_mode}). "
        "Ctrl-C or an empty line to quit.\n"
    )
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        print(f"\n{agent.chat(thread_id, question)}\n")


if __name__ == "__main__":
    main()
