"""PROVISIONAL trajectory logger (T12 scaffold) — pure JSONL file I/O.

No ALFWorld/Groq coupling, so this can be built and tested before docs/log_schema.md (T03)
lands. Once the real schema is merged, revise StepRecord/EpisodeSummaryRecord in schema.py;
this file's I/O functions should not need to change.
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
