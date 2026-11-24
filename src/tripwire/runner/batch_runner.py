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
    run_episode_fn: Callable[..., tuple[list, EpisodeSummaryRecord]] = run_episode,
) -> list[EpisodeSummaryRecord]:
    """Collect up to `num_episodes` pilot trajectories into `output_dir`.

    `run_episode_fn` defaults to the real `run_episode` and is overridable in tests. Episode
    ids are a fixed `ep_0001..ep_{num_episodes:04d}` sequence, so a second invocation with the
    same `num_episodes` naturally resumes rather than starting new episodes.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    output_dir = resolve_output_dir(str(output_dir))
    summaries: list[EpisodeSummaryRecord] = []

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

    return summaries
