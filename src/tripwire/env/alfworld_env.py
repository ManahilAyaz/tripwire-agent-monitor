"""ALFWorld environment wrapper (T13), exposing the real `infos` dict.

ALFWorld's own `env.step`/`env.reset` return `infos` as a dict of lists (TextWorld's batched
convention), not a list of per-env dicts: for batch_size=1, read `infos["won"][0]` and
`infos["extra.gamefile"][0]`, never `infos[0]["won"]`.

`alfworld` is imported lazily, inside `make_env` only, so this module still imports (and its
two pure helpers stay testable) on a machine without ALFWorld installed, e.g. native Windows.
"""

from __future__ import annotations

from typing import Any

from tripwire.paths import resolve_config_path


def make_env(config_path: str) -> Any:
    """Build a single-instance (batch_size=1) ALFWorld environment from a config file."""
    import yaml
    from alfworld.agents.environment import get_environment

    with open(resolve_config_path(config_path), "r") as file:
        cfg = yaml.safe_load(file)
    return get_environment(cfg["env"]["type"])(cfg, train_eval="train").init_env(batch_size=1)


def split_observation_and_goal(text: str) -> tuple[str, str]:
    """Split ALFWorld's initial observation text into (room description, goal).

    ALFWorld's reset text is "<room description>Your task is to: <goal>." — this is the same
    split `t01_sanity_test.py::make_alfworld` uses for `goal`, done here so `initial_observation`
    is also captured separately, per docs/log_schema.md's summary-line field.
    """
    marker = "Your task is to:"
    if marker in text:
        initial_observation, goal = text.split(marker, 1)
        return initial_observation.strip(), goal.strip()
    return text.strip(), ""


def gamefile_to_task_id(gamefile_path: str) -> str:
    """Derive a human-readable task id from ALFWorld's `infos["extra.gamefile"]` path.

    Matches ALFWorld's own convention (`alfworld/agents/eval/evaluate_dagger.py`): the last two
    path components joined, e.g. "pick_heat_then_place_in_recep-Potato-None-SideTable-10/trial_...".
    """
    parts = gamefile_path.replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-3:-1])


def reset_episode(env: Any) -> tuple[str, str, str, list[str]]:
    """Reset the env for a new episode.

    Returns (initial_observation, task_input, task_id, admissible_commands) - the last is
    ALFWorld's own list of the only commands it will actually accept right now, which the
    agent uses to avoid inventing invalid actions (see react_agent.map_to_admissible_action).
    """
    obs, infos = env.reset()
    initial_observation, task_input = split_observation_and_goal(obs[0])
    task_id = gamefile_to_task_id(infos["extra.gamefile"][0])
    admissible_commands = infos["admissible_commands"][0]
    return initial_observation, task_input, task_id, admissible_commands


def step_episode(env: Any, action: str) -> tuple[str, bool, bool, list[str]]:
    """Take one action. Returns (observation, done, won, admissible_commands) - the last is
    ALFWorld's valid-commands list for the *next* step."""
    obs, _scores, dones, infos = env.step([action])
    admissible_commands = infos["admissible_commands"][0]
    return obs[0], bool(dones[0]), bool(infos["won"][0]), admissible_commands
