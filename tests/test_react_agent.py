"""Tests for the T12 reply parser and the T13 episode loop (run_episode), using fakes instead
of a real ALFWorld env/Groq call — no network, no ALFWorld, runs on Windows."""

from __future__ import annotations

import pytest

from tripwire.agent.react_agent import (
    PROMPT_VERSION,
    example_task_inputs,
    load_system_prompt,
    map_to_admissible_action,
    parse_reply,
    run_episode,
    split_think_action,
)


def test_parses_well_formed_reply():
    thought, action, parse_ok = parse_reply(
        "Thought: The potato may be in the fridge.\nAction: go to fridge 1"
    )
    assert thought == "The potato may be in the fridge."
    assert action == "go to fridge 1"
    assert parse_ok is True


def test_falls_back_to_look_when_no_action_line():
    thought, action, parse_ok = parse_reply("Thought: I'm not sure what to do next.")
    assert action == "look"
    assert parse_ok is False


def test_falls_back_to_look_on_empty_reply():
    thought, action, parse_ok = parse_reply("")
    assert thought == ""
    assert action == "look"
    assert parse_ok is False


def test_strips_trailing_punctuation_from_action():
    thought, action, parse_ok = parse_reply(
        "Thought: I should heat the cup.\nAction: heat cup 1 with coffeemachine 1."
    )
    assert action == "heat cup 1 with coffeemachine 1"
    assert parse_ok is True


def _fake_reset(
    initial_observation="You are in a room.",
    task_input="do a thing",
    task_id="task_abc",
    admissible_commands=None,
    task_type="put",
):
    commands = admissible_commands if admissible_commands is not None else []
    return lambda: (initial_observation, task_input, task_id, commands, task_type)


def _scripted_step(*results):
    """results: (observation, done, won) or (observation, done, won, admissible_commands)
    tuples, one per call to step_fn - the 3-tuple form implies admissible_commands=[]."""
    it = iter(results)

    def step(action):
        result = next(it)
        return result if len(result) == 4 else (*result, [])

    return step


def _scripted_llm(*replies):
    it = iter(replies)
    return lambda messages: next(it)


def test_run_episode_success_records_steps_and_outcome():
    steps, summary = run_episode(
        reset_fn=_fake_reset(),
        step_fn=_scripted_step(
            ("You see a mug.", False, False),
            ("You picked up the mug.", True, True),
        ),
        llm_call=_scripted_llm(
            "Thought: look around\nAction: look",
            "Thought: take it\nAction: take mug 1",
        ),
        episode_id="ep_0001",
        model="qwen/qwen3.8-27b",
        max_steps=5,
    )

    assert len(steps) == 2
    assert [s.step for s in steps] == [1, 2]
    assert steps[0].parse_ok is True
    assert steps[0].retry_count == 0
    assert summary.outcome == "success"
    assert summary.num_steps == 2
    assert summary.task_id == "task_abc"
    assert summary.model == "qwen/qwen3.8-27b"


def test_run_episode_done_without_win_is_failure():
    steps, summary = run_episode(
        reset_fn=_fake_reset(),
        step_fn=_scripted_step(("Nothing happens.", True, False)),
        llm_call=_scripted_llm("Thought: give up\nAction: look"),
        episode_id="ep_0002",
        model="qwen/qwen3.8-27b",
        max_steps=5,
    )

    assert summary.outcome == "failure"
    assert summary.num_steps == 1


def test_run_episode_hits_max_steps_without_done():
    steps, summary = run_episode(
        reset_fn=_fake_reset(),
        step_fn=_scripted_step(*[("still going", False, False)] * 3),
        llm_call=_scripted_llm(*["Thought: keep trying\nAction: look"] * 3),
        episode_id="ep_0003",
        model="qwen/qwen3.8-27b",
        max_steps=3,
    )

    assert summary.outcome == "max_steps"
    assert summary.num_steps == 3


def test_run_episode_api_error_after_retries_becomes_error_outcome():
    def always_fails(messages):
        raise RuntimeError("429 rate limited")

    steps, summary = run_episode(
        reset_fn=_fake_reset(),
        step_fn=_scripted_step(("unused", False, False)),
        llm_call=always_fails,
        episode_id="ep_0004",
        model="qwen/qwen3.8-27b",
        max_steps=5,
        max_retries=1,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
    )

    assert summary.outcome == "error"
    assert len(steps) == 0


def test_run_episode_env_crash_mid_episode_becomes_error_outcome():
    def crashing_step(action):
        raise RuntimeError("ALFWorld crashed")

    steps, summary = run_episode(
        reset_fn=_fake_reset(),
        step_fn=crashing_step,
        llm_call=_scripted_llm("Thought: try\nAction: look"),
        episode_id="ep_0005",
        model="qwen/qwen3.8-27b",
        max_steps=5,
    )

    assert summary.outcome == "error"
    assert len(steps) == 0


def test_run_episode_reset_failure_becomes_error_with_zero_steps():
    def crashing_reset():
        raise RuntimeError("ALFWorld reset failed")

    steps, summary = run_episode(
        reset_fn=crashing_reset,
        step_fn=lambda action: pytest.fail("step_fn should never be called"),
        llm_call=lambda messages: pytest.fail("llm_call should never be called"),
        episode_id="ep_0006",
        model="qwen/qwen3.8-27b",
        max_steps=5,
    )

    assert steps == []
    assert summary.outcome == "error"
    assert summary.num_steps == 0
    assert summary.task_id == "unknown"


def test_run_episode_captures_nonzero_retry_count_after_flaky_llm_call():
    attempts = iter([RuntimeError("500 server error"), "Thought: ok\nAction: take mug 1"])

    def flaky_llm(messages):
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    steps, summary = run_episode(
        reset_fn=_fake_reset(),
        step_fn=_scripted_step(("You picked up the mug.", True, True)),
        llm_call=flaky_llm,
        episode_id="ep_0007",
        model="qwen/qwen3.8-27b",
        max_steps=5,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
    )

    assert steps[0].retry_count == 1
    assert steps[0].parse_ok is True
    assert summary.outcome == "success"


def test_run_episode_bounds_message_history_to_window():
    seen_lengths = []

    def recording_llm(messages):
        seen_lengths.append(len(messages))
        return "Thought: keep going\nAction: look"

    steps, summary = run_episode(
        reset_fn=_fake_reset(),
        step_fn=_scripted_step(*[("still going", False, False)] * 6),
        llm_call=recording_llm,
        episode_id="ep_0009",
        model="qwen/qwen3.8-27b",
        max_steps=6,
        history_window_steps=2,
    )

    assert summary.num_steps == 6
    # system + initial observation + at most 2 step-exchanges (4 messages) = 6, never the full
    # un-truncated history (which would reach 2 + 2*6 = 14 by the last step)
    assert max(seen_lengths) <= 6
    assert seen_lengths[-1] == 6


def test_map_to_admissible_action_passes_through_exact_match():
    admissible = ["go to fridge 1", "look"]
    assert map_to_admissible_action("go to fridge 1", admissible) == "go to fridge 1"


def test_map_to_admissible_action_maps_to_closest_command():
    admissible = ["take mug 1 from countertop 1", "go to fridge 1", "look"]
    mapped = map_to_admissible_action("take the mug 1 from the countertop 1.", admissible)
    assert mapped == "take mug 1 from countertop 1"


def test_map_to_admissible_action_passes_through_when_no_admissible_commands():
    assert map_to_admissible_action("do something weird", []) == "do something weird"


def test_run_episode_maps_invalid_action_and_logs_raw_action_separately():
    steps, summary = run_episode(
        reset_fn=_fake_reset(admissible_commands=["go to fridge 1", "look"]),
        step_fn=_scripted_step(("You arrive at fridge 1.", False, False, ["open fridge 1"])),
        llm_call=_scripted_llm("Thought: check the fridge\nAction: go over to the fridge 1"),
        episode_id="ep_0010",
        model="qwen/qwen3.8-27b",
        max_steps=1,
    )

    assert steps[0].raw_action == "go over to the fridge 1"
    assert steps[0].action == "go to fridge 1"


def test_run_episode_raw_action_equals_action_when_already_admissible():
    steps, summary = run_episode(
        reset_fn=_fake_reset(admissible_commands=["look"]),
        step_fn=_scripted_step(("You see nothing.", False, False)),
        llm_call=_scripted_llm("Thought: look around\nAction: look"),
        episode_id="ep_0011",
        model="qwen/qwen3.8-27b",
        max_steps=1,
    )

    assert steps[0].raw_action == "look"
    assert steps[0].action == "look"


def test_run_episode_includes_valid_actions_in_prompt():
    seen_messages = []

    def recording_llm(messages):
        seen_messages.append(messages[-1]["content"])
        return "Thought: ok\nAction: look"

    run_episode(
        reset_fn=_fake_reset(admissible_commands=["look", "go to fridge 1"]),
        step_fn=_scripted_step(("obs", False, False, ["go to fridge 1"])),
        llm_call=recording_llm,
        episode_id="ep_0012",
        model="qwen/qwen3.8-27b",
        max_steps=1,
    )

    assert "Valid actions: look, go to fridge 1" in seen_messages[0]


def test_split_think_action_detects_think_prefix():
    is_think, content = split_think_action("think: I should check the fridge first.")
    assert is_think is True
    assert content == "I should check the fridge first."


def test_split_think_action_tolerates_literal_transcript_prefix():
    # The few-shot examples show "> think: ..." - the model sometimes echoes the "> " too.
    is_think, content = split_think_action("> think: now I have the mug.")
    assert is_think is True
    assert content == "now I have the mug."


def test_split_think_action_passes_through_real_action_unchanged():
    is_think, content = split_think_action("go to fridge 1")
    assert is_think is False
    assert content == "go to fridge 1"


def test_load_system_prompt_fills_in_task_type_matched_examples():
    prompt = load_system_prompt("heat")
    assert "=== Example 1 ===" in prompt
    assert "=== Example 2 ===" in prompt
    # the heat-bucket example's own task text, not another bucket's
    assert "heat" in prompt.lower()


def test_load_system_prompt_falls_back_to_put_for_unknown_task_type():
    # same content as the real "put" bucket, not an error
    assert load_system_prompt("some_unseen_type") == load_system_prompt("put")


def test_example_task_inputs_returns_all_12_fewshot_task_goals():
    task_inputs = example_task_inputs()
    assert len(task_inputs) == 12  # 6 buckets x 2 examples each
    assert "put some spraybottle on toilet." in task_inputs


def test_run_episode_logs_think_step_without_calling_env():
    env_calls = []

    def step(action):
        env_calls.append(action)
        return "You picked up the mug.", True, True, []

    steps, summary = run_episode(
        reset_fn=_fake_reset(admissible_commands=["take mug 1", "look"]),
        step_fn=step,
        llm_call=_scripted_llm(
            "Action: think: I should take the mug.",
            "Action: take mug 1",
        ),
        episode_id="ep_0013",
        model="qwen/qwen3.8-27b",
        max_steps=5,
    )

    assert len(steps) == 2
    assert steps[0].is_think is True
    assert steps[0].action == ""
    assert steps[0].observation == "OK."
    # parse_reply strips trailing punctuation from the whole Action value (by design, for real
    # commands) - a cosmetic side effect on think text too, not a functional issue.
    assert steps[0].thought == "I should take the mug"
    assert steps[1].is_think is False
    assert env_calls == ["take mug 1"]  # the think turn never reached step_fn
    assert summary.num_steps == 1  # the think turn doesn't count as a real action
    assert summary.outcome == "success"


def test_run_episode_think_turns_dont_count_toward_max_steps():
    steps, summary = run_episode(
        reset_fn=_fake_reset(admissible_commands=["look"]),
        step_fn=_scripted_step(("still here", False, False)),
        llm_call=_scripted_llm(
            "Action: think: thinking once.",
            "Action: look",
        ),
        episode_id="ep_0014",
        model="qwen/qwen3.8-27b",
        max_steps=1,  # only 1 real action allowed - the think turn must not consume it
    )

    assert summary.num_steps == 1
    assert summary.outcome == "max_steps"
    assert [s.is_think for s in steps] == [True, False]


def test_run_episode_caps_consecutive_thinks_and_forces_a_real_action():
    steps, summary = run_episode(
        reset_fn=_fake_reset(admissible_commands=["look"]),
        step_fn=_scripted_step(("still here", False, False)),
        llm_call=_scripted_llm(
            "Action: think: one.",
            "Action: think: two.",
            "Action: think: three - trying to stall again.",
        ),
        episode_id="ep_0015",
        model="qwen/qwen3.8-27b",
        max_steps=1,
    )

    assert [s.is_think for s in steps] == [True, True, False]
    # the 3rd think is overridden into a real action instead of being honored
    assert steps[2].action == "look"
    assert steps[2].raw_action == "think: three - trying to stall again"
    assert summary.num_steps == 1
    assert summary.outcome == "max_steps"


def test_run_episode_summary_has_task_type_and_prompt_version():
    _, summary = run_episode(
        reset_fn=_fake_reset(task_type="cool", admissible_commands=["look"]),
        step_fn=_scripted_step(("o", True, True)),
        llm_call=_scripted_llm("Action: look"),
        episode_id="ep_0016",
        model="qwen/qwen3.8-27b",
        max_steps=5,
    )

    assert summary.task_type == "cool"
    assert summary.prompt_version == PROMPT_VERSION


def test_run_episode_reset_failure_summary_has_unknown_task_type():
    def crashing_reset():
        raise RuntimeError("ALFWorld reset failed")

    _, summary = run_episode(
        reset_fn=crashing_reset,
        step_fn=lambda action: pytest.fail("step_fn should never be called"),
        llm_call=lambda messages: pytest.fail("llm_call should never be called"),
        episode_id="ep_0018",
        model="qwen/qwen3.8-27b",
        max_steps=5,
    )

    assert summary.task_type == "unknown"
    assert summary.prompt_version == PROMPT_VERSION


def test_run_episode_logs_parse_ok_false_when_reply_has_no_action_line():
    steps, summary = run_episode(
        reset_fn=_fake_reset(),
        step_fn=_scripted_step(("Nothing happens.", True, False)),
        llm_call=_scripted_llm("Thought: I'm confused, not sure what to do."),
        episode_id="ep_0008",
        model="qwen/qwen3.8-27b",
        max_steps=5,
    )

    assert steps[0].action == "look"
    assert steps[0].parse_ok is False
    assert summary.outcome == "failure"
