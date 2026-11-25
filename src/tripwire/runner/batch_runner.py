"""Resumable batch collector for pilot trajectories (T14).

Must survive Groq free-tier interruptions: skips episodes already done, deletes and redoes
partial/errored ones instead of appending to stale data, and never lets one persistently
broken episode hang a whole run.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from tripwire.agent.react_agent import run_episode
from tripwire.paths import resolve_output_dir
from tripwire.trajectory.logger import TrajectoryLogger, read_jsonl
from tripwire.trajectory.schema import EpisodeSummaryRecord

# "error" is a valid, schema-compliant outcome when written, but the batch runner treats it
# like an unfinished episode so the next run retries it — resumability exists specifically to
# survive Groq free-tier interruptions, and silently losing episodes toward the pilot target
# would defeat that.
DONE_OUTCOMES = {"success", "failure", "max_steps"}


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
