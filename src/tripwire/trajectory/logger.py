"""Trajectory logger (T12) — JSONL file I/O matching docs/log_schema.md v1.0 (PR #5).

No ALFWorld/Groq coupling: this only writes/reads the JSONL files, so it doesn't depend on
the live agent runner.
"""

from __future__ import annotations

import json
from pathlib import Path

from tripwire.trajectory.schema import EpisodeSummaryRecord, StepRecord


class TrajectoryLogger:
    """Writes one JSONL line per step plus one episode-summary line, per episode file."""

    def __init__(self, output_dir: str | Path, episode_id: str):
        self._path = Path(output_dir) / f"{episode_id}.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def log_step(self, record: StepRecord) -> None:
        self._write(record.to_dict())

    def log_episode_summary(self, record: EpisodeSummaryRecord) -> None:
        self._write(record.to_dict())

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TrajectoryLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict]:
    """Read back a trajectory file as a list of dicts, for tests/validation."""
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_prefix(path: str | Path, k: int) -> dict:
    """Load what a monitor is allowed to see: task input, initial observation, steps 1..k.

    Per docs/log_schema.md's no-leak rule, this is the one function T17/T26/T29 call instead
    of writing their own truncation: `outcome` and every other summary-only key are never
    included in the result, no matter how large `k` is or whether the episode has finished.
    """
    records = read_jsonl(path)
    summary = next((r for r in records if r.get("type") == "episode_summary"), None)
    steps = [r for r in records if r.get("type") == "step" and r["step"] <= k]
    return {
        "task_input": summary["task_input"] if summary else None,
        "initial_observation": summary["initial_observation"] if summary else None,
        "steps": steps,
    }
