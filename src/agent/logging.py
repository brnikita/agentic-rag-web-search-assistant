"""structlog configuration.

JSON to stdout so `thread_id` and `turn_id` stay machine-greppable when the
agent runs as a service.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: int = logging.INFO, *, json_output: bool = True) -> None:
    renderer = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
