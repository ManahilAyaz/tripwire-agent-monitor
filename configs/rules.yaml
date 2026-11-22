"""Layer 1 deterministic rule: consecutive repeated actions (T10).

Scope
-----
FYP-I, detection only. This module flags a *suspicious pattern* (taxonomy
TX-01, signal SIG-01). A flag is NOT a confirmed episode failure: an agent can
repeat an action and still recover. There is no correction, recovery or
steering logic here.

What the rule does
------------------
It looks for a "run": neighbouring agent steps whose actions are equal after
normalization. A run is flagged when its length reaches ``threshold``.

* ``threshold`` counts TOTAL occurrences, the first one included.
  threshold = 3 means "the same action 3 times in a row" (first + 2 repeats).
* The default threshold 3 is PROVISIONAL. It was chosen by reasoning, not
  validated on data. Final values belong to T10/T19 once pilot data exists.

Normalization (one definition, also used in docs/taxonomy.md)
-------------------------------------------------------------
trim leading/trailing whitespace, collapse runs of whitespace into one space,
convert to lowercase:  "  Go to  Cabinet 1 " -> "go to cabinet 1".
Different arguments stay different: "go to cabinet 1" != "go to cabinet 2".

What ends (interrupts) a run
----------------------------
* a different action;
* a step that is not usable: missing/blank/non-string action, missing or
  invalid step number, a record of the wrong type, or ``parse_ok`` not true
  (when ``parse_ok`` is false the logger substituted ``look``; that action was
  not chosen by the agent, so it must neither extend nor start a run);
* a gap in the step numbers (e.g. step 3 followed by step 5). A run only
  continues when the next step number is exactly previous + 1.
Unusable records are listed in ``result.skipped`` with a reason.

Schema v1.0 handling (docs/trajectory log schema, agreed by the team)
---------------------------------------------------------------------
* ``type == "step"``            -> examined.
* ``type == "episode_summary"`` -> skipped silently. Nothing is read from it
                                   (in particular not ``outcome``).
* any other / missing type      -> unusable record (reported).
* ``step`` is 1-indexed and is used exactly as logged.
* ``retry_count`` is an API retry counter. It is ignored: one record is one
  agent action, however many retries the LLM call needed.
* ``thought``, ``observation``, ``timestamp`` are not used by this rule.
  (Whether ``thought`` may be used at all is a pending team/supervisor
  decision; see docs/candidate_signals.md.)

No leaks
--------
Monitors may only see a prefix of the trajectory (steps 1..k). The caller
must obtain that prefix from the team's ``load_prefix(path, k)`` (T12). This
module deliberately has NO file reading and NO prefix logic. It only looks at
the records it is given, never at ``outcome`` and never at anything after the
last record passed in. The result for a prefix is therefore the same as the
result the live monitor would have had at that moment.

Structure (kept separate on purpose)
------------------------------------
1. detect_repeated_actions(): core logic. Works on plain lists of actions and
   step numbers. Knows nothing about JSONL or field names.
2. extract_actions():         input adapter for schema v1.0 records. The only
   place that knows field names.
3. check_repeated_actions():  adapter + core in one call.
4. load_config():             reads configs/rules.yaml with strict validation.

Example
-------
    from Tripwire.monitors.rules import check_repeated_actions
    result = check_repeated_actions(prefix_records)   # records from load_prefix
    if result.flagged:
        print(result.first_alert_step, result.evidence)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

RULE_ID = "repeated_action"
TAXONOMY_ID = "TX-01"
SIGNAL_ID = "SIG-01"

# Provisional (not validated on data). Must equal configs/rules.yaml; a test checks this.
DEFAULT_THRESHOLD = 3
MIN_THRESHOLD = 2

# Schema v1.0 constants.
STEP_TYPE = "step"
SUMMARY_TYPE = "episode_summary"

# rules.py lives in <repo>/srs/Tripwire/monitors/, so the repo root is parents[3].
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "rules.yaml"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _validate_threshold(threshold: Any) -> None:
    """Threshold must be a real integer >= 2 (bool is rejected on purpose)."""
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < MIN_THRESHOLD:
        raise ValueError(f"threshold must be an integer >= {MIN_THRESHOLD}, got {threshold!r}")


@dataclass(frozen=True)
class RepeatedActionConfig:
    threshold: int = DEFAULT_THRESHOLD

    def __post_init__(self) -> None:
        _validate_threshold(self.threshold)


def load_config(path: Optional[Path] = None) -> RepeatedActionConfig:
    """Read the ``repeated_action`` section of configs/rules.yaml (strict).

    Raises ValueError when: the file is empty or not a mapping, the
    ``repeated_action`` section is missing or not a mapping, ``threshold`` is
    missing or invalid, or the section has keys other than ``threshold``.
    Other top-level sections (future rules) are allowed and ignored.
    """
    try:
        import yaml  # imported here so the detector itself does not need PyYAML
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise ImportError("PyYAML is needed to read rules.yaml: pip install pyyaml") from exc

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, Mapping):
        raise ValueError(f"{config_path}: top level must be a mapping")
    section = data.get(RULE_ID)
    if not isinstance(section, Mapping):
        raise ValueError(f"{config_path}: missing or invalid '{RULE_ID}' section")

    unknown = set(section) - {"threshold"}
    if unknown:
        raise ValueError(f"Unknown keys in '{RULE_ID}' config: {sorted(unknown)}")
    if "threshold" not in section:
        raise ValueError(f"'{RULE_ID}.threshold' is required")

    return RepeatedActionConfig(threshold=section["threshold"])


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkippedRecord:
    """A record that could not be used as an agent action."""

    record_index: int  # 0-based position in the list passed in
    step: Optional[int]  # logged step number if it was valid, else None
    reason: str


@dataclass(frozen=True)
class RepeatedActionRun:
    """One run of identical consecutive actions that reached the threshold."""

    action: str  # normalized action text that was compared
    run_length: int  # total occurrences, first one included
    start_step: int
    end_step: int
    alert_step: int  # the step at which run_length first reached the threshold
    step_indexes: tuple  # logged step numbers, not Python list indexes


@dataclass(frozen=True)
class RepeatedActionResult:
    flagged: bool
    threshold: int
    runs: tuple = ()
    max_run_length: int = 0  # longest run seen, flagged or not (0 if no usable step)
    num_steps: int = 0  # non-summary records examined, including unusable records
    evidence: tuple = ()
    skipped: tuple = ()
    rule_id: str = RULE_ID
    taxonomy_id: str = TAXONOMY_ID
    signal_id: str = SIGNAL_ID

    @property
    def first_alert_step(self) -> Optional[int]:
        """Earliest step at which this rule would have raised its alert."""
        return min((r.alert_step for r in self.runs), default=None)

    @property
    def summary(self) -> str:
        if self.flagged:
            return (
                "FLAGGED (suspicious pattern, not a confirmed failure): "
                f"{len(self.runs)} repeated-action run(s), first alert at step {self.first_alert_step}."
            )
        return f"Not flagged: longest run was {self.max_run_length} (threshold {self.threshold})."

    def to_dict(self) -> dict:
        """Plain dict (JSON friendly) for logging or later layers."""
        return {
            "rule_id": self.rule_id,
            "taxonomy_id": self.taxonomy_id,
            "signal_id": self.signal_id,
            "flagged": self.flagged,
            "threshold": self.threshold,
            "first_alert_step": self.first_alert_step,
            "max_run_length": self.max_run_length,
            "num_steps": self.num_steps,
            "runs": [
                {
                    "action": r.action,
                    "run_length": r.run_length,
                    "start_step": r.start_step,
                    "end_step": r.end_step,
                    "alert_step": r.alert_step,
                    "step_indexes": list(r.step_indexes),
                }
                for r in self.runs
            ],
            "evidence": list(self.evidence),
            "skipped": [
                {"record_index": s.record_index, "step": s.step, "reason": s.reason}
                for s in self.skipped
            ],
        }


# ---------------------------------------------------------------------------
# 1. Core detection logic (no knowledge of JSONL or field names)
# ---------------------------------------------------------------------------

def normalize_action(action: Any) -> Optional[str]:
    """Trim, collapse whitespace, lowercase. None if not a non-blank string."""
    if not isinstance(action, str):
        return None
    text = " ".join(action.split()).lower()  # split() with no argument trims and collapses
    return text or None


def _is_valid_step(step: Any) -> bool:
    """Schema v1.0: steps are 1-indexed integers (bool is not an integer here)."""
    return isinstance(step, int) and not isinstance(step, bool) and step >= 1


def _group_consecutive(actions: Sequence, steps: Sequence) -> list:
    """Split the trajectory into runs of equal normalized actions.

    Returns a list of (normalized_action, [step numbers]). Unusable items
    (bad action or bad step) and gaps in step numbering end the current run.
    """
    groups: list = []
    current_key: Optional[str] = None
    current_steps: list = []

    for action, step in zip(actions, steps):
        key = normalize_action(action)
        if key is None or not _is_valid_step(step):
            if current_steps:
                groups.append((current_key, current_steps))
            current_key, current_steps = None, []
            continue

        continues_run = (
            bool(current_steps) and key == current_key and step == current_steps[-1] + 1
        )
        if continues_run:
            current_steps.append(step)
        else:
            if current_steps:
                groups.append((current_key, current_steps))
            current_key, current_steps = key, [step]

    if current_steps:
        groups.append((current_key, current_steps))
    return groups


def detect_repeated_actions(
    actions: Iterable,
    threshold: int = DEFAULT_THRESHOLD,
    step_indexes: Optional[Sequence] = None,
) -> RepeatedActionResult:
    """Find runs of identical consecutive actions in an ordered list.

    actions      : ordered actions. Anything that is not a non-blank string is
                   unusable and ends any run.
    threshold    : total occurrences needed to flag a run (integer >= 2).
    step_indexes : step number of each action (1-indexed integers). If omitted
                   the steps are 1, 2, 3, ... An item that is not a positive
                   integer is unusable; a gap in the numbers ends a run.
    """
    _validate_threshold(threshold)
    if actions is None or isinstance(actions, (str, bytes)):
        raise TypeError("actions must be a sequence of action strings")
    action_list = list(actions)
    count = len(action_list)

    if step_indexes is None:
        steps = list(range(1, count + 1))
    else:
        steps = list(step_indexes)
        if len(steps) != count:
            raise ValueError(
                f"step_indexes has {len(steps)} items but actions has {count}; they must match"
            )

    groups = _group_consecutive(action_list, steps)
    max_run = max((len(run_steps) for _, run_steps in groups), default=0)

    runs = []
    evidence = []
    for key, run_steps in groups:
        if len(run_steps) < threshold:
            continue
        run = RepeatedActionRun(
            action=key,
            run_length=len(run_steps),
            start_step=run_steps[0],
            end_step=run_steps[-1],
            alert_step=run_steps[threshold - 1],
            step_indexes=tuple(run_steps),
        )
        runs.append(run)
        evidence.append(
            f"Action {key!r} occurred {run.run_length} times in a row at steps {run_steps}; "
            f"threshold {threshold} was reached at step {run.alert_step}."
        )

    return RepeatedActionResult(
        flagged=bool(runs),
        threshold=threshold,
        runs=tuple(runs),
        max_run_length=max_run,
        num_steps=count,
        evidence=tuple(evidence),
    )


# ---------------------------------------------------------------------------
# 2. Input adapter for schema v1.0 records (the only part that knows field names)
# ---------------------------------------------------------------------------

def _read_record(record: Any) -> tuple:
    """Return (action_or_None, step_or_None, reason_if_unusable) for one step record."""
    if not isinstance(record, Mapping):
        return None, None, "record is not a mapping"

    step = record.get("step")
    step = step if _is_valid_step(step) else None

    record_type = record.get("type")
    if record_type != STEP_TYPE:
        return None, step, f"unexpected record type {record_type!r} (expected {STEP_TYPE!r})"
    if step is None:
        return None, None, "'step' is missing or not a positive integer"

    parse_ok = record.get("parse_ok")
    if parse_ok is False:
        return None, step, "parse_ok is false (logger substituted a default action)"
    if parse_ok is not True:
        return None, step, "'parse_ok' is missing or not a boolean"

    if "action" not in record:
        return None, step, "missing 'action' field"
    action = record["action"]
    if not isinstance(action, str):
        return None, step, "'action' is not a string"
    if not action.strip():
        return None, step, "'action' is empty"
    return action, step, ""


def extract_actions(records: Iterable) -> tuple:
    """Turn schema v1.0 records into (actions, steps, skipped).

    actions : list, str where usable, None where not (None ends a run).
    steps   : list of int (or None), same length as actions.
    skipped : list of SkippedRecord explaining every None.

    ``episode_summary`` records are dropped silently and never read.
    """
    if records is None or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of records")

    actions: list = []
    steps: list = []
    skipped: list = []
    for record_index, record in enumerate(records):
        if isinstance(record, Mapping) and record.get("type") == SUMMARY_TYPE:
            continue  # not a step; do not look at anything inside it
        action, step, reason = _read_record(record)
        actions.append(action)
        steps.append(step)
        if action is None:
            skipped.append(SkippedRecord(record_index, step, reason))
    return actions, steps, skipped


# ---------------------------------------------------------------------------
# 3. Records in, result out
# ---------------------------------------------------------------------------

def check_repeated_actions(
    records: Iterable,
    config: Optional[RepeatedActionConfig] = None,
) -> RepeatedActionResult:
    """Run the T10 check on schema v1.0 records (a prefix from load_prefix)."""
    cfg = config or RepeatedActionConfig()
    actions, steps, skipped = extract_actions(records)
    result = detect_repeated_actions(actions, threshold=cfg.threshold, step_indexes=steps)
    return dataclasses.replace(result, skipped=tuple(skipped))
