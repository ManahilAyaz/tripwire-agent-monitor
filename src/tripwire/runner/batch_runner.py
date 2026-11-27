"""Resumable batch collector for pilot trajectories (T14).

Must survive Groq free-tier interruptions: skips episodes already done, deletes and redoes
partial/errored ones instead of appending to stale data, and never lets one persistently
broken episode hang a whole run.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

from tripwire.agent.react_agent import run_episode
from tripwire.paths import resolve_output_dir
from tripwire.trajectory.logger import TrajectoryLogger, read_jsonl
from tripwire.trajectory.schema import EpisodeSummaryRecord

# "error" is a valid, schema-compliant outcome when written, but the batch runner treats it
# like an unfinished episode so the next run retries it — resumability exists specifically to
# survive Groq free-tier interruptions, and silently losing episodes toward the pilot target
# would defeat that.
DONE_OUTCOMES = {"success", "failure", "max_steps"}

# Fixed, arbitrary seed for the one-time shuffle in `sequential_game_order` — not meant to be
# varied per run (see that function's docstring for why).
TASK_ORDER_SEED = 20240115


def sequential_game_order(
    env: Any,
    order_path: Path | str,
    cursor_path: Path | str,
    seed: int = TASK_ORDER_SEED,
) -> list[str]:
    """Make `env.reset()` draw games strictly without replacement, in a fixed order that
    survives a process restart (T27 fix for episodes repeating the same ALFWorld task).

    Root cause this replaces: ALFWorld's `env.reset()` delegates game selection to
    TextWorld's `TextworldBatchGymEnv`, which already samples *without* replacement from a
    shuffled cycle over all game files — but that cycle is seeded with a hardcoded `1234`
    private to the `TextworldBatchGymEnv` instance, unrelated to this project's own
    `random.seed()`/`--seed`. Every time a Kaggle session restart forced a fresh `env` to be
    built, that cycle restarted at position 0 of the *same* fixed shuffle, so episodes run
    right after a restart could redraw a game an earlier process invocation (this run, or
    even the separate pilot run) had already used — not a random collision, a deterministic
    replay. See `data/raw/full/EXCLUDED.md` for the 11 episodes this caused in the T27 corpus.

    Fix: own the order and the resume position ourselves, on disk, instead of relying on
    `env`'s internal (restart-unsafe) cycle:
    - `order_path` holds the full game-file list, shuffled once with `seed` the first time
      this runs, then loaded unchanged on every later invocation (same file -> same order,
      forever, regardless of how many times the process restarts).
    - `cursor_path` holds how many games have been handed out so far. It's written *before*
      each game is handed out (not after), so even a crash immediately after this function
      hands out a game leaves the cursor already past it — the unsafe direction would be
      under-counting (risking a repeat); over-counting just skips one never-attempted game,
      which is harmless.
    - `env._gamefiles_iterator` is monkey-patched to walk `order` starting at the resumed
      cursor. This relies on `TextworldBatchGymEnv`'s current internal attribute names
      (`gamefiles`, `_gamefiles_iterator`) — confirmed against ALFWorld's `AlfredTWEnv.init_env`
      (`alfworld/agents/environment/alfred_tw_env.py`), which returns `textworld.gym.make(...)`
      directly with no further gym.Wrapper layered on top, so these are reachable as plain
      attributes on whatever `tripwire.env.alfworld_env.make_env` returns. If a future
      ALFWorld/TextWorld upgrade renames these, the `AttributeError` below will say so loudly
      instead of silently falling back to the old repeat-prone behavior.

    Called once right after `env = make_env(...)`, before any `reset_fn()` calls. Safe to call
    on every invocation (first run creates `order_path`/`cursor_path`; every later one resumes
    from them). Returns the loaded/created order, mainly so tests can inspect it.
    """
    if not hasattr(env, "gamefiles") or not hasattr(env, "_gamefiles_iterator"):
        raise AttributeError(
            "env has no 'gamefiles'/'_gamefiles_iterator' attribute - TextWorld's internal "
            "TextworldBatchGymEnv API this monkey-patch depends on may have changed (see "
            "this function's docstring). Falling back silently would reintroduce the "
            "repeated-task bug this exists to fix, so this raises instead."
        )

    # .resolve() (not just Path(...)) collapses any ".." components before this touches the
    # filesystem - same system-boundary pattern as tripwire.paths.resolve_output_dir, applied
    # here too since order_path/cursor_path are passed in directly rather than always routed
    # through that helper.
    order_path = Path(order_path).expanduser().resolve()
    cursor_path = Path(cursor_path).expanduser().resolve()

    if order_path.exists():
        order = json.loads(order_path.read_text(encoding="utf-8"))
    else:
        order = list(env.gamefiles)
        random.Random(seed).shuffle(order)
        order_path.parent.mkdir(parents=True, exist_ok=True)
        order_path.write_text(json.dumps(order), encoding="utf-8")

    cursor = int(cursor_path.read_text(encoding="utf-8")) if cursor_path.exists() else 0

    def _advancing_iterator():
        nonlocal cursor
        for gamefile in order[cursor:]:
            cursor += 1
            cursor_path.write_text(str(cursor), encoding="utf-8")
            yield gamefile

    env._gamefiles_iterator = _advancing_iterator()
    return order


def avoid_example_tasks(
    reset_fn: Callable[[], tuple],
    excluded_task_inputs: set[str],
    max_resamples: int = 50,
) -> Callable[[], tuple]:
    """Wrap a `reset_fn` so it never returns one of the few-shot example tasks.

    `reset_fn`'s 2nd element is always `task_input` (see `env.alfworld_env.reset_episode`'s
    `(initial_observation, task_input, task_id, admissible_commands, task_type)`). ALFWorld's
    `env.reset()` samples from the configured split on every call; since the react_v3 few-shot
    examples (`react_agent.example_task_inputs`) are themselves real ALFWorld train-split
    tasks, there's a real chance of re-sampling one of them, which would mean testing the agent
    on a task it was directly shown solved in its own prompt. Resamples up to `max_resamples`
    times before giving up loudly (`RuntimeError`) instead of silently letting a contaminated
    episode through.
    """

    def wrapped():
        for _ in range(max_resamples):
            result = reset_fn()
            if result[1] not in excluded_task_inputs:
                return result
        raise RuntimeError(
            f"Could not sample a non-example task after {max_resamples} resets - the env's "
            "task pool may be unexpectedly small, or excluded_task_inputs is too broad."
        )

    return wrapped


def is_episode_done(path: Path) -> bool:
    """Whether `path` holds a finished episode that should be skipped on the next run."""
    if not path.exists():
        return False
    try:
        records = read_jsonl(path)
    except (OSError, ValueError):
        return False
    summary = next((r for r in records if r.get("type") == "episode_summary"), None)
    return summary is not None and summary.get("outcome") in DONE_OUTCOMES


def run_batch(
    reset_fn: Callable[[], tuple[str, str, str]],
    step_fn: Callable[[str], tuple[str, bool, bool]],
    llm_call: Callable[[list[dict]], str],
    model: str,
    output_dir: str | Path,
    num_episodes: int,
    max_steps: int,
    max_retries: int = 5,
    backoff_base_seconds: float = 1.0,
    backoff_max_seconds: float = 30.0,
    seed: int = 0,
    history_window_steps: int = 10,
    max_consecutive_errors: int = 3,
    run_episode_fn: Callable[..., tuple[list, EpisodeSummaryRecord]] = run_episode,
) -> list[EpisodeSummaryRecord]:
    """Collect up to `num_episodes` pilot trajectories into `output_dir`.

    `run_episode_fn` defaults to the real `run_episode` and is overridable in tests. Episode
    ids are a fixed `ep_0001..ep_{num_episodes:04d}` sequence, so a second invocation with the
    same `num_episodes` naturally resumes rather than starting new episodes.

    `max_consecutive_errors` stops the whole run early once that many episodes in a row come
    back `error` even after their one in-run retry. A streak that long means something
    systemic is broken (e.g. Groq's daily quota exhausted) rather than one flaky episode, and
    ploughing through the rest of `num_episodes` the same way would just waste time for no new
    data — the already-collected episodes (written to `output_dir` as they finish, same as
    always) are still there to zip up and download.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    output_dir = resolve_output_dir(str(output_dir))
    summaries: list[EpisodeSummaryRecord] = []
    consecutive_errors = 0

    for i in range(1, num_episodes + 1):
        episode_id = f"ep_{i:04d}"
        path = output_dir / f"{episode_id}.jsonl"

        if is_episode_done(path):
            continue
        if path.exists():
            path.unlink()  # partial or errored from a previous run — start this episode fresh

        steps, summary = run_episode_fn(
            reset_fn,
            step_fn,
            llm_call,
            episode_id,
            model,
            max_steps,
            max_retries,
            backoff_base_seconds,
            backoff_max_seconds,
            history_window_steps,
        )

        if summary.outcome == "error":
            # One immediate retry within this run, so a single transient failure doesn't cost
            # the episode — but no more than one, so a persistently broken episode can't hang
            # this run forever. If it errors again, its file is still written (a valid
            # schema-compliant "error" summary) and left for the next invocation to retry.
            steps, summary = run_episode_fn(
                reset_fn,
                step_fn,
                llm_call,
                episode_id,
                model,
                max_steps,
                max_retries,
                backoff_base_seconds,
                backoff_max_seconds,
                history_window_steps,
            )

        with TrajectoryLogger(output_dir, episode_id) as logger:
            for step in steps:
                logger.log_step(step)
            logger.log_episode_summary(summary)

        summaries.append(summary)

        if summary.outcome == "error":
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(
                    f"Stopping early after {consecutive_errors} episodes in a row ended in "
                    f"'error' (even after a retry) — likely a systemic issue (e.g. Groq's "
                    f"daily quota exhausted), not one-off flakiness. Collected "
                    f"{len(summaries)}/{num_episodes} episodes this invocation."
                )
                break
        else:
            consecutive_errors = 0

    return summaries
