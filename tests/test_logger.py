"""Tests for the T12 logger, matching docs/log_schema.md v1.0 (PR #5)."""

from __future__ import annotations

from tripwire.trajectory.logger import TrajectoryLogger, load_prefix, read_jsonl
from tripwire.trajectory.schema import SCHEMA_VERSION, EpisodeSummaryRecord, StepRecord


def _write_sample_episode(tmp_path, episode_id="ep_test_0001"):
    with TrajectoryLogger(tmp_path, episode_id) as logger:
        logger.log_step(
            StepRecord(
                episode_id=episode_id,
                step=1,
                timestamp="2026-11-22T10:00:00.000Z",
                thought="I should find the mug first.",
                action="go to countertop 1",
                observation="You arrive at countertop 1.",
            )
        )
        logger.log_step(
            StepRecord(
                episode_id=episode_id,
                step=2,
                timestamp="2026-11-22T10:00:05.000Z",
                thought="The mug is on the countertop.",
                action="take mug 1",
                observation="You pick up the mug.",
                parse_ok=False,
                retry_count=1,
            )
        )
        logger.log_episode_summary(
            EpisodeSummaryRecord(
                episode_id=episode_id,
                task_id="task_abc",
                task_input="put a clean mug in coffeemachine",
                initial_observation="You are in the middle of a room.",
                num_steps=2,
                outcome="success",
                model="qwen/qwen3.8-27b",
                max_steps=30,
                start_timestamp="2026-11-22T10:00:00.000Z",
                end_timestamp="2026-11-22T10:00:05.000Z",
                task_type="clean",
                prompt_version="react_v3",
            )
        )
        return logger.path


def test_logs_steps_and_summary_as_valid_jsonl(tmp_path):
    out_path = _write_sample_episode(tmp_path)
    lines = read_jsonl(out_path)

    assert len(lines) == 3
    assert lines[0]["step"] == 1
    assert lines[0]["type"] == "step"
    assert lines[0]["parse_ok"] is True
    assert lines[1]["retry_count"] == 1
    assert lines[1]["parse_ok"] is False
    assert lines[2]["type"] == "episode_summary"
    assert lines[2]["schema_version"] == SCHEMA_VERSION
    assert lines[2]["outcome"] == "success"


def test_output_file_named_by_episode_id(tmp_path):
    with TrajectoryLogger(tmp_path, "ep_0042") as logger:
        pass
    assert logger.path.name == "ep_0042.jsonl"
    assert logger.path.parent == tmp_path


def test_load_prefix_truncates_steps_and_drops_outcome(tmp_path):
    out_path = _write_sample_episode(tmp_path)

    prefix = load_prefix(out_path, k=1)

    assert prefix["task_input"] == "put a clean mug in coffeemachine"
    assert prefix["initial_observation"] == "You are in the middle of a room."
    assert [s["step"] for s in prefix["steps"]] == [1]
    assert "outcome" not in prefix
    for step in prefix["steps"]:
        assert "outcome" not in step


def test_load_prefix_with_k_past_episode_end_returns_all_steps(tmp_path):
    out_path = _write_sample_episode(tmp_path)

    prefix = load_prefix(out_path, k=99)

    assert [s["step"] for s in prefix["steps"]] == [1, 2]
    assert "outcome" not in prefix
