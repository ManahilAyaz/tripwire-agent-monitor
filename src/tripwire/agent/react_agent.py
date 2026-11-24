"""The real Tripwire ReAct agent (T12 parser, T13 episode loop).

Kept separate from scripts/t01_sanity_test.py (Manahil's merged, frozen T01 sanity check) per
her T03 reply: one owned copy of the parser here, instead of two implementations that could
drift apart.
"""

from __future__ import annotations

import difflib
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

Every turn lists "Valid actions:" - the exact commands the environment will currently
accept. Your Action must be copied from that list, not invented or reworded. If the exact
action you want is not listed, you likely need a different action first (e.g. "go to X"
before you can interact with X).

Action rules (breaking any of these causes "Nothing happens."):
- Exactly ONE command - never two actions joined by "and", "then", or a comma, and never a
  sentence describing a plan.
- Pick the action exactly as written in "Valid actions:" - same words, same object numbers.
- If your last action's observation was "Nothing happens.", that action was not understood
  or not currently possible - try something different next, don't repeat the same one.

Example:
Valid actions: go to countertop 1, go to fridge 1, look, inventory
Thought: I have not been to the countertop yet, I should go there first.
Action: go to countertop 1

Valid actions: take mug 1 from countertop 1, go to fridge 1, look
Thought: The mug is right here on the countertop, I should pick it up.
Action: take mug 1 from countertop 1

Do not call tools or functions.
Do not return JSON.
"""


def parse_reply(text: str) -> tuple[str, str, bool]:
    """Parse a raw LLM reply into (thought, action, parse_ok).

    `parse_ok` is False when no `Action:` line was found, in which case `action` falls back
    to "look" — this flag is what lets the logger tell a real "look" choice apart from a
    model formatting failure (docs/log_schema.md's `parse_ok` step field).

    Trailing punctuation (periods, etc.) is stripped from the action regardless of what the
    prompt says, since a weaker model's habitual trailing "." would otherwise make an
    otherwise-correct ALFWorld command fail to match and silently return "Nothing happens.".
    """
    thought_match = THOUGHT_RE.search(text)
    action_match = ACTION_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""
    parse_ok = action_match is not None
    action = action_match.group(1).strip() if action_match else "look"
    action = action.rstrip(".!?,;: ")
    return thought, action, parse_ok


def map_to_admissible_action(raw_action: str, admissible_commands: list[str]) -> str:
    """Map a possibly-malformed action to the closest ALFWorld-admissible command.

    ALFWorld only executes an exact string match from `infos["admissible_commands"]` - a
    weaker model's output (wrong object number, invented compound action, stray wording)
    would otherwise silently return "Nothing happens." forever, no matter how many retries.
    Falls back to `raw_action` unchanged if there is nothing to match against (or already an
    exact match), so this never makes a correct action worse.
    """
    if not admissible_commands or raw_action in admissible_commands:
        return raw_action
    match = difflib.get_close_matches(raw_action, admissible_commands, n=1, cutoff=0.0)
    return match[0] if match else raw_action


def _now_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def run_episode(
    reset_fn: Callable[[], tuple[str, str, str, list[str]]],
    step_fn: Callable[[str], tuple[str, bool, bool, list[str]]],
    llm_call: Callable[[list[dict]], str],
    episode_id: str,
    model: str,
    max_steps: int,
    max_retries: int = 5,
    backoff_base_seconds: float = 1.0,
    backoff_max_seconds: float = 30.0,
    history_window_steps: int = 10,
) -> tuple[list[StepRecord], EpisodeSummaryRecord]:
    """Run one episode and return (steps, summary) — pure data, no filesystem writes.

    `reset_fn`/`step_fn` are injected (normally `env.alfworld_env.reset_episode`/`step_episode`
    bound to a live env) so this is testable with fakes and never imports ALFWorld itself. Both
    also return ALFWorld's `admissible_commands` list (the only commands it will currently
    accept) so the model is prompted with them and any reply that isn't an exact match gets
    mapped to the closest one via `map_to_admissible_action`, instead of silently doing nothing.

    `history_window_steps` bounds how many of the most recent step-exchanges are kept in the
    prompt sent to `llm_call` (the system message and initial observation are always kept).
    Without this, the full episode history gets resent every step, so a 30-step episode's token
    usage grows roughly with the triangular sum of its own length — two such episodes were
    enough to exhaust Groq's 200K-tokens/day free-tier budget for qwen/qwen3.8-27b outright.
    """
    start_timestamp = _now_iso()
    try:
        initial_observation, task_input, task_id, admissible_commands = reset_fn()
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
        {
            "role": "user",
            "content": (
                f"{initial_observation}\nGoal: {task_input}\n"
                f"Valid actions: {', '.join(admissible_commands)}"
            ),
        },
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

        thought, raw_action, parse_ok = parse_reply(reply)
        action = map_to_admissible_action(raw_action, admissible_commands)

        try:
            observation, done, won, admissible_commands = step_fn(action)
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
                raw_action=raw_action,
            )
        )
        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Observation: {observation}\n"
                    f"Valid actions: {', '.join(admissible_commands)}\n"
                    "Continue solving the goal."
                ),
            }
        )
        max_messages = 2 + 2 * history_window_steps
        if len(messages) > max_messages:
            messages = messages[:2] + messages[-2 * history_window_steps :]

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
