"""Tests for the T14 resumable batch collector. Fakes run_episode_fn, no ALFWorld/Groq."""

from __future__ import annotations

from tripwire.runner.batch_runner import is_episode_done, run_batch
from tripwire.trajectory.logger import TrajectoryLogger, read_jsonl
from tripwire.trajectory.schema import EpisodeSummaryRecord, StepRecord


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
