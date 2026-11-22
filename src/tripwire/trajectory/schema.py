"""PROVISIONAL trajectory record types — pending T03 (docs/log_schema.md).

Field names here follow Artifacts/schema_proposal_kashmala_draft.md and are NOT final. Once
docs/log_schema.md is merged, update these to match it exactly (T12 note: "follow
docs/log_schema.md exactly").

`thought` is included because Manahil's T01 prototype (agent/executor.py in
tripwire_starter.zip) has the agent emit a ReAct "Thought: ... / Action: ..." pair every step
and already captures `thought` alongside `action`/`observation` per step — so it is externally
observable in the real agent, not a hidden state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class StepRecord:
    """One externally-observable agent step. No hidden states."""

    episode_id: str
    task_id: str
    step: int
    timestamp: str
    thought: str
    action: str
    observation: str
    retry_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeSummaryRecord:
    """One summary line per episode: task, steps, final outcome."""

    episode_id: str
    task_id: str
    task_input: str
    num_steps: int
    outcome: str
    start_timestamp: str
    end_timestamp: str
    type: str = "episode_summary"

    def to_dict(self) -> dict:
        return asdict(self)
