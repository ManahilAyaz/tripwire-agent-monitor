"""Trajectory record types, matching docs/log_schema.md v1.0 (PR #5, w1/t03-log-schema).

PR #5 is real (reviewed with Manahil, open on GitHub) but not yet merged to main — if the
merged schema ends up different, update these to match it exactly (T12 note: "follow
docs/log_schema.md exactly").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "1.5"


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

    `mapping` (v1.3) says how `action` was derived from `raw_action`: `"exact"` (already an
    admissible command, or became identical to one after normalization - see `normalize_action`
    and v1.4 below), `"alias"` (v1.5 - `raw_action` used ReAct's original "put X in/on Y"
    phrasing, which was rewritten to ALFWorld's actual "move X to Y" verb before matching, and
    that rewrite was exactly admissible - see `rewrite_put_as_move`), `"fuzzy"` (closest
    same-verb admissible command, similarity >= 0.8 on normalized text - see
    `map_to_admissible_action`), or `"none"` (no close-enough same-verb match, so `raw_action`
    was sent to ALFWorld unchanged - or this step is a think turn, where no mapping was
    attempted at all). Restricting fuzzy matches to the same first-word verb as `raw_action`
    exists because plain string-similarity matching could otherwise map a "take" action to an
    unrelated "examine" command just for sharing enough characters, which is worse than sending
    the model's literal (wrong) text and getting "Nothing happens.". `"alias"` is logged
    separately from `"exact"` even though both are fully-confident matches, because the raw
    text's verb genuinely differs from the executed command's verb - a real pilot_v4 spot-check
    found 340/741 (46%) of all `"none"` steps were exactly this "put" vs "move" mismatch, often
    repeated many times in a row without the model ever discovering "move" - worth keeping
    visible as its own category rather than folding into "exact".

    `normalized_action`/`similarity` (v1.4): `normalized_action` is `raw_action` lowercased,
    punctuation-stripped, with filler words ("over", "the", "a", "an", "please", "now") removed
    - this is what harmless-phrasing matches (e.g. "go over to the fridge 1" vs "go to fridge
    1") are actually compared on, so a genuinely different object/number (e.g. "tomato 4" vs
    "tomato 1") still isn't treated as the same action. `similarity` is the matched (or best
    attempted) same-verb candidate's score: 1.0 for `"exact"`/`"alias"`, the real ratio for
    `"fuzzy"`, and for `"none"` either the best same-verb candidate's score (if one existed but
    fell short of 0.8) or 0.0 (no same-verb candidate at all, or no admissible commands, or a
    think/forced turn where no mapping was attempted).

    `admissible_actions` (v1.5) is the exact ALFWorld `admissible_commands` list offered this
    turn (what the "Valid actions:" line in the prompt showed) - logged so a mapping decision,
    especially `"alias"`/`"none"`, can be audited against what was actually available at that
    step instead of inferred after the fact from nearby steps.
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
    mapping: str = "none"
    normalized_action: str = ""
    similarity: float = 0.0
    is_think: bool = False
    admissible_actions: list[str] = field(default_factory=list)
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
