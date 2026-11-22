"""Retry/backoff wrapper around Groq API calls (T13 core logic).

Schema-independent: this wraps the API-call layer, not the trajectory logger, so it can be
built and tested before docs/log_schema.md (T03) exists.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class RetryExhaustedError(RuntimeError):
    """Raised when a wrapped call still fails after max_retries attempts."""


class MaxStepsExceededError(RuntimeError):
    """Raised when an episode loop exceeds its configured max_steps cap."""


def is_retryable(exc: Exception) -> bool:
    """Whether an exception looks like a Groq 429 (rate limit) or 5xx (server) error.

    Groq's SDK raises errors exposing a ``status_code`` attribute; fall back to inspecting
    the exception's string form so this also works against plain mocks/fakes in tests.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code == 429 or 500 <= status_code < 600
    text = str(exc)
    return "429" in text or any(f"{code}" in text for code in range(500, 600))


def call_with_retry(
    fn: Callable[[], T],
    max_retries: int = 5,
    backoff_base_seconds: float = 1.0,
    backoff_max_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` with exponential backoff on retryable errors.

    Retries up to ``max_retries`` times, doubling the delay each attempt (capped at
    ``backoff_max_seconds``, with jitter), and re-raises as :class:`RetryExhaustedError`
    once retries are exhausted. Non-retryable exceptions propagate immediately.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below, not swallowed
            if not is_retryable(exc) or attempt >= max_retries:
                raise RetryExhaustedError(
                    f"Call failed after {attempt} retr{'y' if attempt == 1 else 'ies'}: {exc}"
                ) from exc
            delay = min(backoff_base_seconds * (2**attempt), backoff_max_seconds)
            delay += random.uniform(0, delay * 0.1)
            sleep(delay)
            attempt += 1


def enforce_max_steps(step: int, max_steps: int) -> None:
    """Raise :class:`MaxStepsExceededError` once ``step`` reaches ``max_steps``."""
    if step >= max_steps:
        raise MaxStepsExceededError(f"Episode exceeded max_steps cap ({max_steps})")
