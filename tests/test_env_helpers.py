"""Tests for the pure helpers in env/alfworld_env.py. No ALFWorld import, runs on Windows."""

from __future__ import annotations

from tripwire.env.alfworld_env import gamefile_to_task_id, split_observation_and_goal


def test_splits_room_description_from_goal():
    text = (
        "You are in the middle of a room. Looking quickly around you, you see a fridge 1, "
        "a microwave 1, a sidetable 1.Your task is to: heat some potato and put it in sidetable."
    )
    initial_observation, task_input = split_observation_and_goal(text)
    assert initial_observation == (
        "You are in the middle of a room. Looking quickly around you, you see a fridge 1, "
        "a microwave 1, a sidetable 1."
    )
    assert task_input == "heat some potato and put it in sidetable."


def test_split_with_no_goal_marker_returns_whole_text_and_empty_goal():
    initial_observation, task_input = split_observation_and_goal("You are in a room.")
    assert initial_observation == "You are in a room."
    assert task_input == ""


def test_gamefile_to_task_id_uses_last_two_path_components():
    path = (
        "/root/.cache/alfworld/json_2.1.1/train/"
        "pick_heat_then_place_in_recep-Potato-None-SideTable-10/trial_T20190909/game.tw-pddl"
    )
    assert gamefile_to_task_id(path) == (
        "pick_heat_then_place_in_recep-Potato-None-SideTable-10/trial_T20190909"
    )


def test_gamefile_to_task_id_handles_windows_style_backslashes():
    path = (
        "C:\\alfworld\\json_2.1.1\\train\\"
        "pick_and_place_simple-Mug-None-CoffeeMachine-1\\trial_T20190908\\game.tw-pddl"
    )
    assert gamefile_to_task_id(path) == (
        "pick_and_place_simple-Mug-None-CoffeeMachine-1/trial_T20190908"
    )
