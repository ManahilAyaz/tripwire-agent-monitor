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
PUNCT_RE = re.compile(r"[.,!?;:]")
FILLER_WORDS = {"over", "the", "a", "an", "please", "now"}
PUT_AS_MOVE_RE = re.compile(r"^put\s+(.+?)\s+(?:in/on|in|on)\s+(.+)$", re.IGNORECASE)

PROMPT_VERSION = "react_v3.1"

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


GOAL_LINE_RE = re.compile(r"Your task is to:\s*(.*?)\s*\n", re.IGNORECASE)


def example_task_inputs() -> set[str]:
    """The exact goal text of every few-shot example task (configs/prompts/react_alfworld_examples.json).

    These are real ALFWorld training-split tasks, not synthetic ones - so a pilot/full batch
    that also samples from the train split could by chance re-draw one of them, which would
    mean testing the agent on a task it was already directly shown solved in its own prompt.
    `runner.batch_runner.avoid_example_tasks` uses this set to resample instead.
    """
    data = json.loads((_PROMPTS_DIR / "react_alfworld_examples.json").read_text(encoding="utf-8"))
    task_inputs = set()
    for examples in data["examples"].values():
        for text in examples:
            match = GOAL_LINE_RE.search(text)
            if match:
                task_inputs.add(match.group(1).strip())
    return task_inputs


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


def normalize_action(text: str) -> str:
    """Lowercase, strip punctuation, and drop filler words that don't change the action's
    meaning ("over", "the", "a", "an", "please", "now") - so "go over to the fridge 1" and
    "go to fridge 1" normalize to the same text instead of being treated as different actions.
    Object names, numbers, and the prepositions ALFWorld's own vocabulary uses ("to", "from",
    "in/on", "with") are left alone - this is phrasing cleanup, not paraphrasing.
    """
    text = PUNCT_RE.sub("", text.lower())
    return " ".join(word for word in text.split() if word not in FILLER_WORDS)


def rewrite_put_as_move(raw_action: str) -> str | None:
    """Rewrite ReAct's "put X in/on Y" / "put X in Y" / "put X on Y" phrasing to ALFWorld's
    actual placement verb, "move X to Y" - or None if `raw_action` isn't a "put ..." action.

    The few-shot examples (adapted from ysymyth/ReAct) originally taught this "put" phrasing,
    but current ALFWorld's admissible commands use "move X to Y" instead (confirmed from real
    pilot_v4 trajectories, e.g. "You move the bread 2 to the countertop 3."). Since
    `map_to_admissible_action` restricts matches to the same first-word verb, a "put"-verb
    raw_action could never match a "move"-verb admissible command no matter how close the rest
    of the text was - this rewrite exists specifically to bridge that one known verb mismatch.
    """
    match = PUT_AS_MOVE_RE.match(raw_action.strip())
    if not match:
        return None
    return f"move {match.group(1)} to {match.group(2)}"


def map_to_admissible_action(
    raw_action: str, admissible_commands: list[str]
) -> tuple[str, str, str, float]:
    """Map a possibly-malformed action to the closest ALFWorld-admissible command.

    Returns `(action, mapping, normalized_action, similarity)`:
    - `mapping="exact"` (similarity 1.0): `raw_action` was already admissible, or became
      identical to an admissible command once both are normalized (harmless phrasing
      difference, e.g. "go over to the fridge 1" vs "go to fridge 1") - the real admissible
      command is sent, not the normalized text.
    - `mapping="alias"` (similarity 1.0): `raw_action` used the old ReAct "put X in/on Y"
      phrasing (see `rewrite_put_as_move`); rewriting it to "move X to Y" and normalizing gave
      an exact admissible match. Logged separately from `"exact"` since the raw text's own verb
      ("put") genuinely isn't the command's verb ("move"), unlike a harmless phrasing tweak.
    - `mapping="fuzzy"`: no exact/normalized-exact/alias match, but the closest *same-verb*
      admissible command (matched on the first word - "take", "go", "heat", etc., compared
      after normalization) scored >= 0.8 similarity on normalized text.
    - `mapping="none"`: no admissible command shares `raw_action`'s verb, or none scored high
      enough - `raw_action` is returned unchanged and sent to ALFWorld as-is. `similarity` is
      still the best same-verb candidate's score when one existed, 0.0 otherwise, so a
      near-miss is visible even when it wasn't accepted.

    The same-verb restriction matters: plain string similarity alone can match "take tomato 4
    from fridge 1" to "examine fridge 1" just because they share enough characters (fridge, 1,
    overall shape), even though they're completely different action types - worse than doing
    nothing, since it quietly substitutes the wrong *kind* of action instead of surfacing the
    mismatch. Restricting candidates to the same first word first, then requiring a high
    similarity among those, keeps a fuzzy match from ever crossing into a different verb - which
    is exactly why the "put"/"move" alias check above has to happen as its own step, before
    verb-matching, rather than relying on the fuzzy path to bridge it.
    """
    normalized_raw = normalize_action(raw_action)
    if not admissible_commands:
        return raw_action, "none", normalized_raw, 0.0
    if raw_action in admissible_commands:
        return raw_action, "exact", normalized_raw, 1.0

    normalized_commands = {c: normalize_action(c) for c in admissible_commands}
    for command, normalized_command in normalized_commands.items():
        if normalized_command == normalized_raw:
            return command, "exact", normalized_raw, 1.0

    alias_rewrite = rewrite_put_as_move(raw_action)
    if alias_rewrite is not None:
        normalized_alias = normalize_action(alias_rewrite)
        for command, normalized_command in normalized_commands.items():
            if normalized_command == normalized_alias:
                return command, "alias", normalized_alias, 1.0

    verb = normalized_raw.split(" ", 1)[0] if normalized_raw else ""
    same_verb_commands = {
        c: nc for c, nc in normalized_commands.items() if nc.split(" ", 1)[0] == verb
    }
    if same_verb_commands:
        best_command, best_score = None, 0.0
        for command, normalized_command in same_verb_commands.items():
            score = difflib.SequenceMatcher(None, normalized_raw, normalized_command).ratio()
            if score > best_score:
                best_command, best_score = command, score
        if best_score >= 0.8:
            return best_command, "fuzzy", normalized_raw, round(best_score, 3)
        return raw_action, "none", normalized_raw, round(best_score, 3)
    return raw_action, "none", normalized_raw, 0.0


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

    # No "You are at X / holding Y" line here on purpose: that's not part of standard ReAct,
    # and telling the agent its own state back to correct for a model that's losing track of
    # it is really the future correction layer's job (FYP-II), not this baseline agent's.
    system_prompt = load_system_prompt(task_type)
    messages = [
        {"role": "system", "content": system_prompt},
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
    turn_index = 0
    env_step_num = 0
    consecutive_thinks = 0

    while env_step_num < max_steps:
        turn_index += 1
        done_flag, won_flag = False, False
        # captured before step_fn below reassigns admissible_commands to the *next* turn's
        # list - this is what the model was actually shown (the "Valid actions:" line) and
        # what map_to_admissible_action matched against this turn, so it's what gets logged.
        current_admissible_commands = list(admissible_commands)
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
            mapping, normalized_action, similarity = "none", "", 0.0
            observation = "OK."
            step_thought = think_text
        else:
            consecutive_thinks = 0
            if think_capped:
                action, mapping, normalized_action, similarity = "look", "none", "", 0.0
            else:
                action, mapping, normalized_action, similarity = map_to_admissible_action(
                    raw_action, current_admissible_commands
                )
            step_thought = think_text if think_capped else thought
            try:
                observation, done_flag, won_flag, admissible_commands = step_fn(action)
            except Exception:  # noqa: BLE001 - env step crashed mid-episode
                outcome = "error"
                break
            env_step_num += 1

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
                mapping=mapping,
                normalized_action=normalized_action,
                similarity=similarity,
                is_think=is_think,
                admissible_actions=current_admissible_commands,
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
