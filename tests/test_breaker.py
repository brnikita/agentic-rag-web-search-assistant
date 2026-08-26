"""Circuit-breaker thresholds."""

from __future__ import annotations

from agent.nodes import check_breaker


def test_does_not_trip_when_healthy(settings):
    state = {"tool_calls": 1, "total_tool_calls": 1, "errors": []}
    assert check_breaker(state, settings) == {"breaker_tripped": False}


def test_trips_on_per_turn_tool_call_limit(settings):
    state = {"tool_calls": settings.max_tool_calls_per_turn, "total_tool_calls": 5, "errors": []}
    assert check_breaker(state, settings)["breaker_tripped"] is True


def test_trips_on_thread_lifetime_limit(settings):
    state = {
        "tool_calls": 1,
        "total_tool_calls": settings.max_tool_calls_per_thread,
        "errors": [],
    }
    assert check_breaker(state, settings)["breaker_tripped"] is True


def test_trips_on_error_threshold(settings):
    state = {"tool_calls": 2, "total_tool_calls": 2, "errors": ["rag: boom", "web: boom"]}
    assert check_breaker(state, settings)["breaker_tripped"] is True


def test_routing_errors_alone_do_not_trip(settings):
    """`analyze` already degraded to a safe default; that is not a tool outage."""
    state = {
        "tool_calls": 1,
        "total_tool_calls": 1,
        "errors": ["analyze: rate limited", "analyze: rate limited"],
    }
    assert check_breaker(state, settings)["breaker_tripped"] is False


def test_thresholds_follow_settings(settings):
    state = {"tool_calls": 2, "total_tool_calls": 2, "errors": []}
    assert check_breaker(state, settings)["breaker_tripped"] is False
    tighter = settings.model_copy(update={"max_tool_calls_per_turn": 2})
    assert check_breaker(state, tighter)["breaker_tripped"] is True


def test_tolerates_empty_state(settings):
    assert check_breaker({}, settings) == {"breaker_tripped": False}
