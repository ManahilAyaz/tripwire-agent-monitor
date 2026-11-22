"""The real Tripwire ReAct agent (T12 parser, T13 episode loop).

Kept separate from scripts/t01_sanity_test.py (Manahil's merged, frozen T01 sanity check) per
her T03 reply: one owned copy of the parser here, instead of two implementations that could
drift apart.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable

from tripwire.agent.groq_client import RetryExhaustedError, call_with_retry
from tripwire.trajectory.schema import EpisodeSummaryRecord, StepRecord

THOUGHT_RE = re.compile(r"Thought:\s*(.*)", re.IGNORECASE)
ACTION_RE = re.compile(r"Action:\s*(.*)", re.IGNORECASE)

SYSTEM = """
You are an agent solving household tasks in ALFWorld.
Respond using exactly this format:
Thought: Brief explanation of your next step
Action: One valid ALFWorld command
Valid actions include:
go to X, open X, close X, take X from Y,
put X in/on Y, use X, clean X with Y,
heat X with Y, cool X with Y, look, inventory.
Do not call tools or functions.
Do not return JSON.
"""


def parse_reply(text: str) -> tuple[str, str, bool]:
    """Parse a raw LLM reply into (thought, action, parse_ok).

    `parse_ok` is False when no `Action:` line was found, in which case `action` falls back
    to "look" — this flag is what lets the logger tell a real "look" choice apart from a
    model formatting failure (docs/log_schema.md's `parse_ok` step field).
    """
    thought_match = THOUGHT_RE.search(text)
    action_match = ACTION_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""
    parse_ok = action_match is not None
    action = action_match.group(1).strip() if action_match else "look"
    return thought, action, parse_ok


def _now_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def run_episode(
    reset_fn: Callable[[], tuple[str, str, str]],
    step_fn: Callable[[str], tuple[str, bool, bool]],
    llm_call: Callable[[list[dict]], str],
    episode_id: str,
    model: str,
    max_steps: int,
    max_retries: int = 5,
    backoff_base_seconds: float = 1.0,
    backoff_max_seconds: float = 30.0,
) -> tuple[list[StepRecord], EpisodeSummaryRecord]:
    """Run one episode and return (steps, summary) — pure data, no filesystem writes.

    `reset_fn`/`step_fn` are injected (normally `env.alfworld_env.reset_episode`/`step_episode`
    bound to a live env) so this is testable with fakes and never imports ALFWorld itself.
    """
    start_timestamp = _now_iso()
    try:
        initial_observation, task_input, task_id = reset_fn()
    except Exception:  # noqa: BLE001 - env reset failed, record as an error episode
        end_timestamp = _now_iso()
        return [], EpisodeSummaryRecord(
            episode_id=episode_id,
            task_id="unknown",
            task_input="",
            initial_observation="",
            num_steps=0,
            outcome="error",
            model=model,
            max_steps=max_steps,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{initial_observation}\nGoal: {task_input}"},
    ]
    steps: list[StepRecord] = []
    outcome = "max_steps"

    for step_num in range(1, max_steps + 1):
        try:
            reply, retry_count = call_with_retry(
                lambda: llm_call(messages),
                max_retries=max_retries,
                backoff_base_seconds=backoff_base_seconds,
                backoff_max_seconds=backoff_max_seconds,
            )
        except RetryExhaustedError:
            outcome = "error"
            break

        thought, action, parse_ok = parse_reply(reply)

        try:
            observation, done, won = step_fn(action)
        except Exception:  # noqa: BLE001 - env step crashed mid-episode
            outcome = "error"
            break

        timestamp = _now_iso()
        steps.append(
            StepRecord(
                episode_id=episode_id,
                step=step_num,
                timestamp=timestamp,
                thought=thought,
                action=action,
                observation=observation,
                parse_ok=parse_ok,
                retry_count=retry_count,
            )
        )
        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {"role": "user", "content": f"Observation: {observation}\nContinue solving the goal."}
        )

        if done:
            outcome = "success" if won else "failure"
            break

    end_timestamp = _now_iso()
    summary = EpisodeSummaryRecord(
        episode_id=episode_id,
        task_id=task_id,
        task_input=task_input,
        initial_observation=initial_observation,
        num_steps=len(steps),
        outcome=outcome,
        model=model,
        max_steps=max_steps,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )
    return steps, summary
