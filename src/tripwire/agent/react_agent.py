"""The real Tripwire ReAct agent (T12 parser, T13 episode loop).

Kept separate from scripts/t01_sanity_test.py (Manahil's merged, frozen T01 sanity check) per
her T03 reply: one owned copy of the parser here, instead of two implementations that could
drift apart.
"""

from __future__ import annotations

import difflib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tripwire.agent.groq_client import RetryExhaustedError, call_with_retry
from tripwire.trajectory.schema import EpisodeSummaryRecord, StepRecord

THOUGHT_RE = re.compile(r"Thought:\s*(.*)", re.IGNORECASE)
ACTION_RE = re.compile(r"Action:\s*(.*)", re.IGNORECASE)
THINK_PREFIX_RE = re.compile(r"^>?\s*think:\s*", re.IGNORECASE)
GO_TO_RE = re.compile(r"^go to (.+)$", re.IGNORECASE)
TAKE_RE = re.compile(r"^take (.+?) from (.+)$", re.IGNORECASE)
RELEASE_RE = re.compile(r"^(?:put|move) (.+?) (?:in|on|to) (.+)$", re.IGNORECASE)

PROMPT_VERSION = "react_v3"

# configs/prompts/ lives at the repo root, not relative to this installed package - resolved
# from this file's own location so it works regardless of the caller's cwd (scripts invoke
# run_episode/run_batch from the repo root, but nothing guarantees that).
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "configs" / "prompts"


def load_task_examples(task_type: str) -> str:
    """Load the 2 task-type-matched few-shot examples (configs/prompts/react_alfworld_examples.json).

    These are verbatim ReAct-repo walkthroughs (see that file's `_citation`), not anything
    from our own pilot/full corpus — the whole point is to demonstrate the *format* (ReAct's
    "think: ..." reasoning steps, interleaved with real actions) on a different task instance.
    """
    data = json.loads((_PROMPTS_DIR / "react_alfworld_examples.json").read_text(encoding="utf-8"))
    examples = data["examples"].get(task_type, data["examples"]["put"])
    return "\n\n".join(f"=== Example {i} ===\n{text}" for i, text in enumerate(examples, 1))


def load_system_prompt(task_type: str) -> str:
    """Build the full system prompt (configs/prompts/react_v3.txt) for one task type."""
    template = (_PROMPTS_DIR / f"{PROMPT_VERSION}.txt").read_text(encoding="utf-8")
    return template.format(task_examples=load_task_examples(task_type))


def parse_reply(text: str) -> tuple[str, str, bool]:
    """Parse a raw LLM reply into (thought, action, parse_ok).

    `parse_ok` is False when no `Action:` line was found, in which case `action` falls back
    to "look" — this flag is what lets the logger tell a real "look" choice apart from a
    model formatting failure (docs/log_schema.md's `parse_ok` step field).

    A `Thought:` line is optional (react_v3's prompt no longer requires one — reasoning
    normally happens via a "think: ..." action instead, see `split_think_action`); when
    present it's still captured here for backward compatibility.

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


def split_think_action(action: str) -> tuple[bool, str]:
    """Split a "think: <reasoning>" action into (True, reasoning); anything else is (False, action).

    Tolerates the model echoing the few-shot examples' literal "> " transcript prefix (e.g.
    "> think: ...") since that's exactly what it's shown in `configs/prompts/react_v3.txt`.
    """
    match = THINK_PREFIX_RE.match(action)
    if match:
        return True, action[match.end() :].strip()
    return False, action


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


def _update_state(location: str, holding: str, executed_action: str) -> tuple[str, str]:
    """Track "where am I / what am I holding" from the executed action alone (not by parsing
    observation text, which varies too much to match reliably) - "go to X" updates location,
    "take X from Y" picks it up, "put X in/on Y" / "move X to Y" sets it back down."""
    go_to = GO_TO_RE.match(executed_action)
    if go_to:
        location = go_to.group(1)
    take = TAKE_RE.match(executed_action)
    if take:
        holding = take.group(1)
    release = RELEASE_RE.match(executed_action)
    if release:
        holding = "nothing"
    return location, holding


def _now_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def run_episode(
    reset_fn: Callable[[], tuple[str, str, str, list[str], str]],
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
    bound to a live env) so this is testable with fakes and never imports ALFWorld itself.
    `reset_fn` also returns ALFWorld's `admissible_commands` (so the model is prompted with
    them and any reply that isn't an exact match gets mapped to the closest one via
    `map_to_admissible_action`) and `task_type` (which few-shot examples to prompt with, see
    `load_task_examples`).

    ReAct-style "think: ..." actions (`split_think_action`) are reasoning steps, not
    environment actions: they're never sent to `step_fn`, the agent just gets back a fixed
    "OK." and another turn, and they don't count toward `max_steps`/ALFWorld's own step cap -
    only `env_step_num` (real actions) does. Two in a row is the most the agent gets before a
    third consecutive "think:" is overridden into a real "look" action instead, so a model
    that tries to stall by only ever "thinking" still can't prevent the episode from
    progressing (and therefore from ever reaching `max_steps`).

    `history_window_steps` bounds how many of the most recent turns are kept in the prompt
    sent to `llm_call` (the system message and initial observation are always kept). Without
    this, the full episode history gets resent every step, so a 30-step episode's token usage
    grows roughly with the triangular sum of its own length — two such episodes were enough to
    exhaust Groq's 200K-tokens/day free-tier budget for qwen/qwen3.8-27b outright.
    """
    start_timestamp = _now_iso()
    try:
        initial_observation, task_input, task_id, admissible_commands, task_type = reset_fn()
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
            task_type="unknown",
            prompt_version=PROMPT_VERSION,
        )

    system_prompt = load_system_prompt(task_type)
    location, holding = "unknown", "nothing"
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{initial_observation}\nGoal: {task_input}\n"
                f"You are at: {location}. You are holding: {holding}.\n"
                f"Valid actions: {', '.join(admissible_commands)}"
            ),
        },
    ]
    steps: list[StepRecord] = []
    outcome = "max_steps"
    turn_index = 0
    env_step_num = 0
    consecutive_thinks = 0

    while env_step_num < max_steps:
        turn_index += 1
        done_flag, won_flag = False, False
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
        is_think, think_text = split_think_action(raw_action)
        think_capped = is_think and consecutive_thinks >= 2
        if think_capped:
            # Capped: force a real action instead of a 3rd consecutive think, so the agent
            # can't stall the episode forever just by "thinking". "look" is always admissible
            # and side-effect-free, so it's a safe deterministic fallback here rather than
            # fuzzy-matching nonsense reasoning text against the admissible-commands list.
            # raw_action/thought still keep the model's actual (overridden) reply, same
            # transparency as any other raw_action/action mismatch.
            is_think = False

        if is_think:
            consecutive_thinks += 1
            action = ""
            observation = "OK."
            step_thought = think_text
        else:
            consecutive_thinks = 0
            action = "look" if think_capped else map_to_admissible_action(
                raw_action, admissible_commands
            )
            step_thought = think_text if think_capped else thought
            try:
                observation, done_flag, won_flag, admissible_commands = step_fn(action)
            except Exception:  # noqa: BLE001 - env step crashed mid-episode
                outcome = "error"
                break
            env_step_num += 1
            location, holding = _update_state(location, holding, action)

        timestamp = _now_iso()
        steps.append(
            StepRecord(
                episode_id=episode_id,
                step=turn_index,
                timestamp=timestamp,
                thought=step_thought,
                action=action,
                observation=observation,
                parse_ok=parse_ok,
                retry_count=retry_count,
                raw_action=raw_action,
                is_think=is_think,
            )
        )
        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Observation: {observation}\n"
                    f"You are at: {location}. You are holding: {holding}.\n"
                    f"Valid actions: {', '.join(admissible_commands)}\n"
                    "Continue solving the goal."
                ),
            }
        )
        max_messages = 2 + 2 * history_window_steps
        if len(messages) > max_messages:
            messages = messages[:2] + messages[-2 * history_window_steps :]

        if not is_think and done_flag:
            outcome = "success" if won_flag else "failure"
            break

    end_timestamp = _now_iso()
    summary = EpisodeSummaryRecord(
        episode_id=episode_id,
        task_id=task_id,
        task_input=task_input,
        initial_observation=initial_observation,
        num_steps=env_step_num,
        outcome=outcome,
        model=model,
        max_steps=max_steps,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        task_type=task_type,
        prompt_version=PROMPT_VERSION,
    )
    return steps, summary
