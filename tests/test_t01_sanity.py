import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from t01_sanity_test import parse_reply, run_sanity


def test_parse_reply_normal():
    assert parse_reply("Thought: find the mug\nAction: go to cabinet 1") == ("find the mug", "go to cabinet 1")


def test_parse_reply_missing_action_falls_back_to_look():
    assert parse_reply("I am confused")[1] == "look"


def test_run_sanity_respects_max_steps_and_records_fields():
    out = run_sanity(lambda a: ("ok", False), lambda m: "Thought: t\nAction: look", "obs", "goal", max_steps=3)
    assert len(out["steps"]) == 3
    assert set(out["steps"][0]) == {"t", "thought", "action", "observation"}


def test_run_sanity_stops_when_done():
    out = run_sanity(lambda a: ("done", True), lambda m: "Thought: t\nAction: look", "obs", "goal", max_steps=5)
    assert len(out["steps"]) == 1


def test_output_is_json_serialisable():
    out = run_sanity(lambda a: ("ok", False), lambda m: "Thought: t\nAction: look", "obs", "goal", max_steps=2)
    assert json.loads(json.dumps(out))["goal"] == "goal"
