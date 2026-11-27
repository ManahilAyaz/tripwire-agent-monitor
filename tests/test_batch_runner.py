"""Tests for the T14 resumable batch collector. Fakes run_episode_fn, no ALFWorld/Groq."""

from __future__ import annotations

import json

import pytest

from tripwire.runner.batch_runner import (
    avoid_example_tasks,
    is_episode_done,
    run_batch,
    sequential_game_order,
)
from tripwire.trajectory.logger import TrajectoryLogger, read_jsonl
from tripwire.trajectory.schema import EpisodeSummaryRecord, StepRecord


class FakeTextworldEnv:
    """Stands in for TextWorld's `TextworldBatchGymEnv` - just the two attributes
    `sequential_game_order` depends on (`gamefiles`, `_gamefiles_iterator`)."""

    def __init__(self, gamefiles):
        self.gamefiles = list(gamefiles)
        self._gamefiles_iterator = iter(self.gamefiles)  # ALFWorld's own (restart-unsafe) default


def _summary(episode_id, outcome, num_steps=1):
    return EpisodeSummaryRecord(
        episode_id=episode_id,
        task_id="task_abc",
        task_input="do a thing",
        initial_observation="You are in a room.",
        num_steps=num_steps,
        outcome=outcome,
        model="qwen/qwen3.8-27b",
        max_steps=5,
        start_timestamp="2026-11-22T10:00:00.000Z",
        end_timestamp="2026-11-22T10:00:05.000Z",
        task_type="put",
        prompt_version="react_v3",
    )


def _one_step(episode_id, step=1):
    return StepRecord(
        episode_id=episode_id,
        step=step,
        timestamp="2026-11-22T10:00:00.000Z",
        thought="t",
        action="look",
        observation="o",
    )


def _write_episode_file(output_dir, episode_id, outcome, with_summary=True):
    with TrajectoryLogger(output_dir, episode_id) as logger:
        logger.log_step(_one_step(episode_id))
        if with_summary:
            logger.log_episode_summary(_summary(episode_id, outcome))
    return output_dir / f"{episode_id}.jsonl"


def test_avoid_example_tasks_passes_through_a_clean_task():
    reset_fn = avoid_example_tasks(
        lambda: ("obs", "put the mug in the sink.", "task_x"),
        excluded_task_inputs={"put some spraybottle on toilet."},
    )
    assert reset_fn() == ("obs", "put the mug in the sink.", "task_x")


def test_avoid_example_tasks_resamples_past_an_excluded_task():
    results = iter(
        [
            ("obs", "put some spraybottle on toilet.", "task_example"),
            ("obs2", "put the mug in the sink.", "task_clean"),
        ]
    )
    reset_fn = avoid_example_tasks(
        lambda: next(results),
        excluded_task_inputs={"put some spraybottle on toilet."},
    )
    assert reset_fn() == ("obs2", "put the mug in the sink.", "task_clean")


def test_avoid_example_tasks_gives_up_after_max_resamples():
    reset_fn = avoid_example_tasks(
        lambda: ("obs", "put some spraybottle on toilet.", "task_example"),
        excluded_task_inputs={"put some spraybottle on toilet."},
        max_resamples=3,
    )
    with pytest.raises(RuntimeError):
        reset_fn()


def test_is_episode_done_true_for_real_outcomes(tmp_path):
    for outcome in ("success", "failure", "max_steps"):
        path = _write_episode_file(tmp_path, f"ep_{outcome}", outcome)
        assert is_episode_done(path) is True


def test_is_episode_done_false_for_error_outcome(tmp_path):
    path = _write_episode_file(tmp_path, "ep_err", "error")
    assert is_episode_done(path) is False


def test_is_episode_done_false_for_unfinished_file_with_no_summary(tmp_path):
    path = _write_episode_file(tmp_path, "ep_partial", outcome=None, with_summary=False)
    assert is_episode_done(path) is False


def test_is_episode_done_false_when_file_missing(tmp_path):
    assert is_episode_done(tmp_path / "ep_missing.jsonl") is False


def test_run_batch_skips_already_done_episode(tmp_path):
    _write_episode_file(tmp_path, "ep_0001", "success")
    calls = []

    def fake_run_episode(reset_fn, step_fn, llm_call, episode_id, *args):
        calls.append(episode_id)
        return [_one_step(episode_id)], _summary(episode_id, "success")

    run_batch(
        reset_fn=lambda: ("obs", "goal", "task"),
        step_fn=lambda a: ("o", True, True),
        llm_call=lambda m: "Thought: x\nAction: look",
        model="qwen/qwen3.8-27b",
        output_dir=tmp_path,
        num_episodes=1,
        max_steps=5,
        run_episode_fn=fake_run_episode,
    )

    assert calls == []  # ep_0001 was already done, never re-run


def test_run_batch_retries_error_outcome_episode_once_then_succeeds(tmp_path):
    _write_episode_file(tmp_path, "ep_0001", "error")
    calls = []

    def fake_run_episode(reset_fn, step_fn, llm_call, episode_id, *args):
        calls.append(episode_id)
        return [_one_step(episode_id)], _summary(episode_id, "success")

    summaries = run_batch(
        reset_fn=lambda: ("obs", "goal", "task"),
        step_fn=lambda a: ("o", True, True),
        llm_call=lambda m: "Thought: x\nAction: look",
        model="qwen/qwen3.8-27b",
        output_dir=tmp_path,
        num_episodes=1,
        max_steps=5,
        run_episode_fn=fake_run_episode,
    )

    assert calls == ["ep_0001"]  # old error file deleted, retried once, succeeded
    assert summaries[0].outcome == "success"
    lines = read_jsonl(tmp_path / "ep_0001.jsonl")
    assert lines[-1]["outcome"] == "success"


def test_run_batch_deletes_unfinished_partial_file_before_retrying(tmp_path):
    _write_episode_file(tmp_path, "ep_0001", outcome=None, with_summary=False)

    def fake_run_episode(reset_fn, step_fn, llm_call, episode_id, *args):
        return [_one_step(episode_id)], _summary(episode_id, "success")

    run_batch(
        reset_fn=lambda: ("obs", "goal", "task"),
        step_fn=lambda a: ("o", True, True),
        llm_call=lambda m: "Thought: x\nAction: look",
        model="qwen/qwen3.8-27b",
        output_dir=tmp_path,
        num_episodes=1,
        max_steps=5,
        run_episode_fn=fake_run_episode,
    )

    lines = read_jsonl(tmp_path / "ep_0001.jsonl")
    assert len(lines) == 2  # one step + one summary, not the stale leftover step too
    assert lines[-1]["outcome"] == "success"


def test_run_batch_gives_up_after_one_retry_within_same_run(tmp_path):
    calls = []

    def always_errors(reset_fn, step_fn, llm_call, episode_id, *args):
        calls.append(episode_id)
        return [], _summary(episode_id, "error", num_steps=0)

    summaries = run_batch(
        reset_fn=lambda: ("obs", "goal", "task"),
        step_fn=lambda a: ("o", True, True),
        llm_call=lambda m: "Thought: x\nAction: look",
        model="qwen/qwen3.8-27b",
        output_dir=tmp_path,
        num_episodes=2,
        max_steps=5,
        run_episode_fn=always_errors,
    )

    # exactly 2 attempts per episode (initial + one retry), across 2 episodes = 4 calls total
    assert calls == ["ep_0001", "ep_0001", "ep_0002", "ep_0002"]
    assert [s.outcome for s in summaries] == ["error", "error"]
    # the error file is still written, so a later invocation can retry it again
    assert (tmp_path / "ep_0001.jsonl").exists()
    assert is_episode_done(tmp_path / "ep_0001.jsonl") is False


def test_run_batch_stops_early_after_consecutive_error_streak(tmp_path):
    calls = []

    def always_errors(reset_fn, step_fn, llm_call, episode_id, *args):
        calls.append(episode_id)
        return [], _summary(episode_id, "error", num_steps=0)

    summaries = run_batch(
        reset_fn=lambda: ("obs", "goal", "task"),
        step_fn=lambda a: ("o", True, True),
        llm_call=lambda m: "Thought: x\nAction: look",
        model="qwen/qwen3.8-27b",
        output_dir=tmp_path,
        num_episodes=10,
        max_steps=5,
        max_consecutive_errors=2,
        run_episode_fn=always_errors,
    )

    # 2 attempts (initial + retry) per episode, for exactly 2 episodes before stopping — not
    # all 10 episodes' worth of wasted attempts
    assert calls == ["ep_0001", "ep_0001", "ep_0002", "ep_0002"]
    assert len(summaries) == 2


def test_run_batch_success_resets_the_consecutive_error_streak(tmp_path):
    outcomes = iter(["error", "error", "success", "error", "error"])

    def scripted(reset_fn, step_fn, llm_call, episode_id, *args):
        outcome = next(outcomes)
        return [], _summary(episode_id, outcome, num_steps=0)

    summaries = run_batch(
        reset_fn=lambda: ("obs", "goal", "task"),
        step_fn=lambda a: ("o", True, True),
        llm_call=lambda m: "Thought: x\nAction: look",
        model="qwen/qwen3.8-27b",
        output_dir=tmp_path,
        num_episodes=3,
        max_steps=5,
        max_consecutive_errors=2,
        run_episode_fn=scripted,
    )

    # ep_0001: error, error (retry) -> still error, counted. ep_0002: success (streak resets).
    # ep_0003: error, error (retry) -> error again, streak hits 2 -> would stop, but there's no
    # ep_0004 anyway since num_episodes=3. All 3 episodes run.
    assert len(summaries) == 3
    assert [s.outcome for s in summaries] == ["error", "success", "error"]


def test_sequential_game_order_rejects_env_missing_expected_attributes():
    class BareEnv:
        pass

    with pytest.raises(AttributeError):
        sequential_game_order(BareEnv(), order_path="x", cursor_path="y")


def test_sequential_game_order_creates_order_file_shuffled_from_gamefiles(tmp_path):
    env = FakeTextworldEnv(["a", "b", "c", "d", "e"])
    order_path = tmp_path / "order.json"

    order = sequential_game_order(env, order_path=order_path, cursor_path=tmp_path / "cursor.txt")

    assert sorted(order) == ["a", "b", "c", "d", "e"]
    assert json.loads(order_path.read_text(encoding="utf-8")) == order


def test_sequential_game_order_never_repeats_within_one_process(tmp_path):
    env = FakeTextworldEnv(["a", "b", "c"])
    sequential_game_order(env, order_path=tmp_path / "order.json", cursor_path=tmp_path / "cursor.txt")

    drawn = [next(env._gamefiles_iterator) for _ in range(3)]

    assert len(set(drawn)) == 3  # no repeats
    with pytest.raises(StopIteration):
        next(env._gamefiles_iterator)  # only 3 games existed - exhausted, not wrapped around


def test_sequential_game_order_resumes_from_persisted_cursor_across_a_new_env(tmp_path):
    order_path = tmp_path / "order.json"
    cursor_path = tmp_path / "cursor.txt"

    env_1 = FakeTextworldEnv(["a", "b", "c", "d"])
    sequential_game_order(env_1, order_path=order_path, cursor_path=cursor_path)
    first_two = [next(env_1._gamefiles_iterator) for _ in range(2)]

    # Simulate a process restart: a brand new env (same underlying game pool), same paths.
    env_2 = FakeTextworldEnv(["a", "b", "c", "d"])
    sequential_game_order(env_2, order_path=order_path, cursor_path=cursor_path)
    next_two = [next(env_2._gamefiles_iterator) for _ in range(2)]

    assert set(first_two).isdisjoint(next_two)  # env_2 never redraws what env_1 already drew


def test_sequential_game_order_loads_identical_order_on_resume_even_if_gamefiles_reordered(tmp_path):
    order_path = tmp_path / "order.json"
    cursor_path = tmp_path / "cursor.txt"

    env_1 = FakeTextworldEnv(["a", "b", "c"])
    first_order = sequential_game_order(env_1, order_path=order_path, cursor_path=cursor_path)

    # A fresh env whose own `gamefiles` attribute happens to come back in a different order -
    # sequential_game_order must ignore it and reuse the saved order, not reshuffle.
    env_2 = FakeTextworldEnv(["c", "a", "b"])
    second_order = sequential_game_order(env_2, order_path=order_path, cursor_path=cursor_path)

    assert second_order == first_order


def test_sequential_game_order_advances_cursor_before_yielding(tmp_path):
    # The cursor file must already reflect a game as "consumed" the instant it's handed out,
    # not after - so a crash immediately after this point can never cause a repeat.
    env = FakeTextworldEnv(["a", "b"])
    cursor_path = tmp_path / "cursor.txt"
    sequential_game_order(env, order_path=tmp_path / "order.json", cursor_path=cursor_path)

    next(env._gamefiles_iterator)

    assert cursor_path.read_text(encoding="utf-8") == "1"


def test_run_batch_writes_valid_jsonl_for_fresh_episode(tmp_path):
    def fake_run_episode(reset_fn, step_fn, llm_call, episode_id, *args):
        return [_one_step(episode_id, step=1), _one_step(episode_id, step=2)], _summary(
            episode_id, "success", num_steps=2
        )

    summaries = run_batch(
        reset_fn=lambda: ("obs", "goal", "task"),
        step_fn=lambda a: ("o", True, True),
        llm_call=lambda m: "Thought: x\nAction: look",
        model="qwen/qwen3.8-27b",
        output_dir=tmp_path,
        num_episodes=1,
        max_steps=5,
        run_episode_fn=fake_run_episode,
    )

    lines = read_jsonl(tmp_path / "ep_0001.jsonl")
    assert len(lines) == 3
    assert [l["type"] for l in lines] == ["step", "step", "episode_summary"]
    assert len(summaries) == 1
