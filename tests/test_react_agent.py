"""Tests for the T12 reply parser, including the fallback case Manahil asked for in T03."""

from __future__ import annotations

from tripwire.agent.react_agent import parse_reply


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
