"""Keeps the examples in docs/log_schema.md honest: they must match the documented keys and types."""
import json
import re
from pathlib import Path

DOC = (Path(__file__).parent.parent / "docs" / "log_schema.md").read_text(encoding="utf-8")
STEP_KEYS = {"type", "episode_id", "step", "timestamp", "thought", "action", "observation", "parse_ok", "retry_count", "raw_action", "mapping", "is_think"}
SUMMARY_KEYS = {"type", "schema_version", "episode_id", "task_id", "task_input", "initial_observation",
                "num_steps", "outcome", "model", "max_steps", "start_timestamp", "end_timestamp",
                "task_type", "prompt_version"}
OUTCOMES = {"success", "failure", "max_steps", "error"}
TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def lines(tag):
    block = re.search(rf"```{tag}\n(.*?)```", DOC, re.S).group(1)
    return [json.loads(x) for x in block.strip().splitlines()]


def test_step_examples_match_schema():
    steps = lines("jsonl-step")
    assert len(steps) >= 2
    for i, s in enumerate(steps, 1):
        assert set(s) == STEP_KEYS
        assert s["type"] == "step" and s["step"] == i
        assert isinstance(s["parse_ok"], bool) and isinstance(s["retry_count"], int) and s["retry_count"] >= 0
        assert TS.match(s["timestamp"])


def test_summary_example_matches_schema():
    (s,) = lines("jsonl-summary")
    assert set(s) == SUMMARY_KEYS
    assert s["type"] == "episode_summary" and s["schema_version"] == "1.3"
    assert s["outcome"] in OUTCOMES
    assert TS.match(s["start_timestamp"]) and TS.match(s["end_timestamp"])


def test_all_outcomes_are_documented():
    for o in OUTCOMES:
        assert f"`{o}`" in DOC


def test_step_and_summary_share_episode_id():
    assert {s["episode_id"] for s in lines("jsonl-step")} == {lines("jsonl-summary")[0]["episode_id"]}