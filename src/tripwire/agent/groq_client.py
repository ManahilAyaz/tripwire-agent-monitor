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
) -> tuple[T, int]:
    """Call ``fn`` with exponential backoff on retryable errors.

    Retries up to ``max_retries`` times, doubling the delay each attempt (capped at
    ``backoff_max_seconds``, with jitter), and re-raises as :class:`RetryExhaustedError`
    once retries are exhausted. Non-retryable exceptions propagate immediately.

    Returns ``(result, attempts)`` — ``attempts`` is how many retries were needed (0 if the
    first call succeeded), which the caller logs as docs/log_schema.md's per-step
    ``retry_count``.
    """
    attempt = 0
    while True:
        try:
            return fn(), attempt
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


def make_groq_llm(model: str, log_rate_limits: bool = True) -> Callable[[list[dict]], str]:
    """Build a ``llm(messages) -> reply_text`` callable against the Groq API.

    Reads ``GROQ_API_KEY`` from the environment only (never hard-coded), same contract as
    ``scripts/t01_sanity_test.py::make_groq_llm`` — kept here instead since this is the
    Groq-facing module and other callers (``run_episode``, the batch runner) need it too.

    When ``log_rate_limits`` is true (the default), each call prints Groq's own
    ``x-ratelimit-*`` response headers — the live, authoritative remaining-quota numbers for
    this key, straight from the server for the request just made. This is what tells you how
    much of today's budget is left without checking the console.groq.com dashboard by hand.
    """
    import os

    from openai import OpenAI

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Set it before calling make_groq_llm().")
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

    def llm(messages: list[dict]) -> str:
        raw_response = client.chat.completions.with_raw_response.create(
            model=model, messages=messages, temperature=0.2, max_tokens=300
        )
        if log_rate_limits:
            limits = {k: v for k, v in raw_response.headers.items() if "ratelimit" in k.lower()}
            if limits:
                print(f"[groq rate limits] {limits}")
        response = raw_response.parse()
        message = response.choices[0].message
        if not message.content:
            raise ValueError("The model returned an empty response.")
        return message.content

    return llm
