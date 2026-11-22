"""Tests for the T12 logger scaffold. Pure file I/O, no ALFWorld/schema-final dependency."""

from __future__ import annotations

from tripwire.trajectory.logger import TrajectoryLogger, read_jsonl
from tripwire.trajectory.schema import EpisodeSummaryRecord, StepRecord


def test_logs_steps_and_summary_as_valid_jsonl(tmp_path):
    episode_id = "ep_test_0001"

    with TrajectoryLogger(tmp_path, episode_id) as logger:
        logger.log_step(
            StepRecord(
                episode_id=episode_id,
                task_id="task_abc",
                step=0,
                timestamp="2026-11-22T10:00:00Z",
                thought="I should find the mug first.",
                action="go to countertop 1",
                observation="You arrive at countertop 1.",
            )
        )
        logger.log_step(
            StepRecord(
                episode_id=episode_id,
                task_id="task_abc",
                step=1,
                timestamp="2026-11-22T10:00:05Z",
                thought="The mug is on the countertop.",
                action="take mug 1",
                observation="You pick up the mug.",
                retry_count=1,
            )
        )
        logger.log_episode_summary(
            EpisodeSummaryRecord(
                episode_id=episode_id,
                task_id="task_abc",
                task_input="put a clean mug in coffeemachine",
                num_steps=2,
                outcome="success",
                start_timestamp="2026-11-22T10:00:00Z",
                end_timestamp="2026-11-22T10:00:05Z",
            )
        )
        out_path = logger.path

    lines = read_jsonl(out_path)

    assert len(lines) == 3
    assert lines[0]["step"] == 0
    assert lines[0]["action"] == "go to countertop 1"
    assert lines[1]["retry_count"] == 1
    assert lines[2]["type"] == "episode_summary"
    assert lines[2]["outcome"] == "success"


def test_output_file_named_by_episode_id(tmp_path):
    with TrajectoryLogger(tmp_path, "ep_0042") as logger:
        pass
    assert logger.path.name == "ep_0042.jsonl"
    assert logger.path.parent == tmp_path
