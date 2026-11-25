"""Trajectory record types, matching docs/log_schema.md v1.0 (PR #5, w1/t03-log-schema).

PR #5 is real (reviewed with Manahil, open on GitHub) but not yet merged to main — if the
merged schema ends up different, update these to match it exactly (T12 note: "follow
docs/log_schema.md exactly").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SCHEMA_VERSION = "1.2"


@dataclass
class StepRecord:
    """One externally-observable agent step. No hidden states.

    `parse_ok` is False when the agent's reply had no valid `Action:` line and the parser
    (`agent/react_agent.py::parse_reply`) substituted "look" — without this flag a parsing
    failure is indistinguishable from the agent genuinely choosing "look", which would pollute
    Aqsa's repeated-action/stagnation signals.

    `raw_action` (v1.1) is what the model actually produced before mapping to the closest
    ALFWorld-admissible command (equal to `action` when no mapping was needed). Keeping both
    instead of only the corrected one preserves a real failure-mode signal — a model saying
    an invalid/invented action is itself something Tripwire exists to monitor, not noise to
    silently discard.

    `is_think` (v1.2) is True for a ReAct-style "think: ..." turn — reasoning the agent chose
    to do out loud instead of acting. These are never sent to ALFWorld (`action` is `""` and
    `observation` is the fixed stand-in `"OK."`) and don't count toward `max_steps`/ALFWorld's
    own step cap, which only bounds real environment actions
    (`agent/react_agent.py::run_episode`).
    """

    episode_id: str
    step: int
    timestamp: str
    thought: str
    action: str
    observation: str
    parse_ok: bool = True
    retry_count: int = 0
    raw_action: str = ""
    is_think: bool = False
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
    task_type: str
    prompt_version: str
    type: str = "episode_summary"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)
