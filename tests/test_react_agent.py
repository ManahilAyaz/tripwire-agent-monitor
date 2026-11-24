"""Tests for the T12 reply parser and the T13 episode loop (run_episode), using fakes instead
of a real ALFWorld env/Groq call — no network, no ALFWorld, runs on Windows."""

from __future__ import annotations

import pytest

from tripwire.agent.react_agent import parse_reply, run_episode


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


def _fake_reset(initial_observation="You are in a room.", task_input="do a thing", task_id="task_abc"):
    return lambda: (initial_observation, task_input, task_id)


def _scripted_step(*results):
    """results: list of (observation, done, won) tuples, one per call to step_fn."""
    it = iter(results)
    return lambda action: next(it)


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
