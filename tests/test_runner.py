"""Tests for the T13 retry/backoff + max-steps wrapper. No network, no ALFWorld."""

from __future__ import annotations

import pytest

from tripwire.agent.groq_client import (
    MaxStepsExceededError,
    RetryExhaustedError,
    call_with_retry,
    enforce_max_steps,
)


class _FakeStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"fake error {status_code}")
        self.status_code = status_code


def test_retries_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    attempts = iter([_FakeStatusError(429), _FakeStatusError(500), "ok"])

    def flaky():
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    result, retry_count = call_with_retry(flaky, max_retries=5, sleep=sleeps.append)

    assert result == "ok"
    assert retry_count == 2  # two retries before success
    assert len(sleeps) == 2  # one sleep per retried attempt


def test_raises_after_exhausting_retries(monkeypatch):
    def always_fails():
        raise _FakeStatusError(503)

    with pytest.raises(RetryExhaustedError):
        call_with_retry(always_fails, max_retries=2, sleep=lambda _: None)


def test_non_retryable_error_propagates_immediately(monkeypatch):
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise _FakeStatusError(400)

    with pytest.raises(RetryExhaustedError):
        call_with_retry(bad_request, max_retries=5, sleep=lambda _: None)
    assert calls["n"] == 1  # never retried a non-retryable error


def test_backoff_delays_increase(monkeypatch):
    sleeps: list[float] = []
    attempts = iter([_FakeStatusError(429)] * 3 + ["ok"])

    def flaky():
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    call_with_retry(
        flaky,
        max_retries=5,
        backoff_base_seconds=1.0,
        backoff_max_seconds=30.0,
        sleep=sleeps.append,
    )

    assert len(sleeps) == 3
    assert sleeps[0] < sleeps[1] < sleeps[2]


def test_max_steps_cap_stops_loop():
    max_steps = 5
    for step in range(max_steps):
        enforce_max_steps(step, max_steps)  # steps 0..4 are fine

    with pytest.raises(MaxStepsExceededError):
        enforce_max_steps(max_steps, max_steps)
