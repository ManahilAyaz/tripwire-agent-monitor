"""Trajectory record types, matching docs/log_schema.md v1.0 (PR #5, w1/t03-log-schema).

PR #5 is real (reviewed with Manahil, open on GitHub) but not yet merged to main — if the
merged schema ends up different, update these to match it exactly (T12 note: "follow
docs/log_schema.md exactly").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SCHEMA_VERSION = "1.0"


@dataclass
class StepRecord:
    """One externally-observable agent step. No hidden states.

    `parse_ok` is False when the agent's reply had no valid `Action:` line and the parser
    (`agent/react_agent.py::parse_reply`) substituted "look" — without this flag a parsing
    failure is indistinguishable from the agent genuinely choosing "look", which would pollute
    Aqsa's repeated-action/stagnation signals.
    """

    episode_id: str
    step: int
    timestamp: str
    thought: str
    action: str
    observation: str
    parse_ok: bool = True
    retry_count: int = 0
    type: str = "step"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeSummaryRecord:
    """One summary line per episode, written only when the episode finishes.

    Holds `outcome` — per the no-leak rule, monitors never see this field; only
    `trajectory.logger.load_prefix` and evaluation code may read it.
    """

    episode_id: str
    task_id: str
    task_input: str
    initial_observation: str
    num_steps: int
    outcome: str
    model: str
    max_steps: int
    start_timestamp: str
    end_timestamp: str
    type: str = "episode_summary"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)
