"""Unit tests for the repeated-action rule (T10), written for Schema v1.0.

Run from the repository root (the folder that contains srs/, configs/, docs/, tests/):
    python -m pytest tests/test_rules.py -v

ASSUMPTION (documented in CHANGELOG.md): the package folder is srs/Tripwire/ and
srs/ is the import root, so the module is imported as Tripwire.monitors.rules.
If your team imports it differently (e.g. `from srs.Tripwire.monitors.rules import ...`),
change only the import block below.
"""

import json
import sys
from pathlib import Path

import pytest


def _find_repo_root() -> Path:
    """Walk upwards from this file until the folder holding configs/rules.yaml is found."""
    for folder in Path(__file__).resolve().parents:
        if (folder / "configs" / "rules.yaml").is_file():
            return folder
    raise RuntimeError("Could not find the repository root (a folder containing configs/rules.yaml)")


ROOT = _find_repo_root()
sys.path.insert(0, str(ROOT / "srs"))

from Tripwire.monitors.rules import (  # noqa: E402
    DEFAULT_THRESHOLD,
    RepeatedActionConfig,
    check_repeated_actions,
    detect_repeated_actions,
    extract_actions,
    load_config,
    normalize_action,
)


# ---------------------------------------------------------------------------
# Helpers: build records exactly like Schema v1.0
# ---------------------------------------------------------------------------

def step(n, action, parse_ok=True, retry_count=0, **extra):
    """A schema v1.0 step line."""
    record = {
        "type": "step",
        "episode_id": "ep_0001",
        "step": n,
        "timestamp": "2026-11-22T10:14:52.101Z",
        "thought": "",
        "action": action,
        "observation": "You are in the middle of a room.",
        "parse_ok": parse_ok,
        "retry_count": retry_count,
    }
    record.update(extra)
    return record


def summary(outcome="success", num_steps=3):
    """A schema v1.0 episode_summary line."""
    return {
        "type": "episode_summary",
        "schema_version": "1.0",
        "episode_id": "ep_0001",
        "task_id": "pick_heat_then_place_in_recep-Potato-None-SideTable-10",
        "task_input": "heat some potato and put it in sidetable",
        "initial_observation": "You are in the middle of a room.",
        "num_steps": num_steps,
        "outcome": outcome,
        "model": "qwen/qwen3.8-27b",
        "max_steps": 40,
        "start_timestamp": "2026-11-22T10:14:50.000Z",
        "end_timestamp": "2026-11-22T10:15:40.200Z",
    }


def looks(first, count, **kwargs):
    """`count` consecutive 'look' steps numbered from `first`."""
    return [step(first + i, "look", **kwargs) for i in range(count)]


# ---------------------------------------------------------------------------
# Core logic on plain action lists
# ---------------------------------------------------------------------------

class TestNormalActions:
    def test_normal_actions_not_flagged(self):
        actions = ["go to cabinet 1", "open cabinet 1", "take mug 1 from cabinet 1", "go to sinkbasin 1"]
        result = detect_repeated_actions(actions)
        assert result.flagged is False
        assert result.runs == ()
        assert result.evidence == ()
        assert result.first_alert_step is None
        assert result.max_run_length == 1
        assert result.num_steps == 4

    def test_two_in_a_row_not_flagged_by_default(self):
        result = detect_repeated_actions(["look", "look", "go to desk 1"])
        assert result.flagged is False
        assert result.max_run_length == 2

    def test_same_action_far_apart_is_not_consecutive(self):
        result = detect_repeated_actions(["look", "go to desk 1", "look", "go to desk 2", "look"])
        assert result.flagged is False
        assert result.max_run_length == 1

    def test_default_threshold_is_three(self):
        assert DEFAULT_THRESHOLD == 3

    def test_empty_list(self):
        result = detect_repeated_actions([])
        assert (result.flagged, result.num_steps, result.max_run_length) == (False, 0, 0)

    def test_single_action(self):
        result = detect_repeated_actions(["look"])
        assert result.flagged is False and result.max_run_length == 1

    def test_tuple_and_generator_accepted(self):
        assert detect_repeated_actions(("x", "x", "x")).flagged is True
        assert detect_repeated_actions(a for a in ["x", "x", "x"]).flagged is True

    @pytest.mark.parametrize("bad", [None, "look look look", b"look"])
    def test_wrong_input_type_rejected(self, bad):
        with pytest.raises(TypeError):
            detect_repeated_actions(bad)


class TestRunReporting:
    def test_three_in_a_row_reports_action_length_and_steps(self):
        result = detect_repeated_actions(["go to desk 1", "look", "look", "look", "take pen 1 from desk 1"])
        assert result.flagged is True
        assert len(result.runs) == 1
        run = result.runs[0]
        assert run.action == "look"
        assert run.run_length == 3
        assert (run.start_step, run.end_step) == (2, 4)
        assert run.step_indexes == (2, 3, 4)
        assert run.alert_step == 4  # threshold 3 reached at the 3rd occurrence
        assert result.first_alert_step == 4
        assert "'look'" in result.evidence[0] and "[2, 3, 4]" in result.evidence[0]

    def test_long_run_reported_once_alert_at_threshold(self):
        result = detect_repeated_actions(["look"] * 6)
        assert len(result.runs) == 1
        assert result.runs[0].run_length == 6
        assert result.runs[0].alert_step == 3  # not step 6
        assert result.max_run_length == 6

    def test_run_at_start_and_at_end(self):
        result = detect_repeated_actions(["a", "a", "a", "b", "c", "c", "c"])
        assert [r.start_step for r in result.runs] == [1, 5]
        assert result.first_alert_step == 3

    def test_multiple_runs_listed_separately(self):
        result = detect_repeated_actions(["a", "a", "a", "b", "c", "c", "c", "c", "d"])
        assert [(r.action, r.run_length, r.start_step, r.end_step) for r in result.runs] == [
            ("a", 3, 1, 3),
            ("c", 4, 5, 8),
        ]
        assert len(result.evidence) == 2
        assert result.max_run_length == 4


class TestThresholdBoundaries:
    def test_run_equal_to_threshold_is_flagged(self):
        assert detect_repeated_actions(["x"] * 3, threshold=3).flagged is True
        assert detect_repeated_actions(["x"] * 4, threshold=4).flagged is True

    def test_run_one_below_threshold_is_not_flagged(self):
        result = detect_repeated_actions(["x"] * 3, threshold=4)
        assert result.flagged is False
        assert result.max_run_length == 3  # still reported as information

    def test_run_one_above_threshold_is_flagged_once(self):
        result = detect_repeated_actions(["x"] * 4, threshold=3)
        assert result.flagged is True
        assert len(result.runs) == 1 and result.runs[0].run_length == 4

    def test_minimum_threshold_two(self):
        assert detect_repeated_actions(["x", "x"], threshold=2).flagged is True
        assert detect_repeated_actions(["x", "y"], threshold=2).flagged is False

    def test_alert_step_follows_threshold(self):
        result = detect_repeated_actions(["x"] * 5, threshold=4)
        assert result.runs[0].alert_step == 4

    @pytest.mark.parametrize("bad", [1, 0, -3, 2.5, 3.0, "3", None, True, False])
    def test_invalid_threshold_rejected(self, bad):
        with pytest.raises(ValueError):
            detect_repeated_actions(["x", "x", "x"], threshold=bad)


class TestInterruptions:
    def test_different_action_interrupts_run(self):
        actions = ["a", "a", "b", "a", "a"]
        assert detect_repeated_actions(actions, threshold=3).flagged is False
        two = detect_repeated_actions(actions, threshold=2)
        assert [r.step_indexes for r in two.runs] == [(1, 2), (4, 5)]

    def test_alternating_actions_never_form_a_run(self):
        result = detect_repeated_actions(["a", "b"] * 5)
        assert result.flagged is False and result.max_run_length == 1

    @pytest.mark.parametrize("bad", [None, "", "   ", 5, ["look"]])
    def test_unusable_action_interrupts_run(self, bad):
        # three "a" in total, but a bad item in the middle: not consecutive
        assert detect_repeated_actions(["a", "a", bad, "a"], threshold=3).flagged is False

    def test_two_unusable_items_do_not_join_runs(self):
        result = detect_repeated_actions([None, None, "a", "a", "a"])
        assert result.runs[0].step_indexes == (3, 4, 5)

    def test_gap_in_step_numbers_interrupts_run(self):
        # steps 1, 2, 4: step 3 is missing from the log, so these are not 3 consecutive steps
        result = detect_repeated_actions(["look"] * 3, step_indexes=[1, 2, 4])
        assert result.flagged is False
        assert result.max_run_length == 2

    def test_repeated_or_decreasing_step_numbers_interrupt_run(self):
        assert detect_repeated_actions(["x"] * 3, step_indexes=[1, 2, 2]).flagged is False
        assert detect_repeated_actions(["x"] * 3, step_indexes=[3, 2, 1]).flagged is False

    def test_new_run_alerts_only_after_its_own_threshold(self):
        result = detect_repeated_actions(
            ["a", "a", "b", "a", "a", "a"],
            threshold=3
        )

        assert result.flagged is True
        assert len(result.runs) == 1
        assert result.runs[0].action == "a"
        assert result.runs[0].step_indexes == (4, 5, 6)
        assert result.first_alert_step == 6

    def test_detection_resumes_after_unusable_record(self):
        records = [
            step(1, "look"),
            step(2, "look"),
            step(3, None),
            step(4, "look"),
            step(5, "look"),
            step(6, "look"),
        ]

        result = check_repeated_actions(records)

        assert result.flagged is True
        assert result.runs[0].step_indexes == (4, 5, 6)
        assert result.first_alert_step == 6
        assert len(result.skipped) == 1


class TestNormalization:
    def test_normalize_action_function(self):
        assert normalize_action("  Go to  Cabinet 1 ") == "go to cabinet 1"
        assert normalize_action("look\t\n") == "look"
        assert normalize_action("") is None
        assert normalize_action("   ") is None
        assert normalize_action(None) is None
        assert normalize_action(7) is None

    def test_case_and_spaces_ignored(self):
        result = detect_repeated_actions(["Go to  Cabinet 1", " go to cabinet 1 ", "GO TO CABINET 1"])
        assert result.flagged is True
        assert result.runs[0].action == "go to cabinet 1"  # reported in normalized form

    def test_different_arguments_are_different_actions(self):
        result = detect_repeated_actions(["go to cabinet 1", "go to cabinet 2", "go to cabinet 3"])
        assert result.flagged is False

    def test_internal_spaces_are_collapsed_not_removed(self):
        # "take mug 1" must not equal "takemug 1"
        result = detect_repeated_actions(["take mug 1", "take  mug  1", "takemug 1"])
        assert result.flagged is False


class TestStepNumbering:
    def test_default_numbering_is_one_indexed(self):
        assert detect_repeated_actions(["x"] * 3).runs[0].step_indexes == (1, 2, 3)

    def test_custom_step_indexes_used_as_given(self):
        result = detect_repeated_actions(["x", "x", "x"], step_indexes=[10, 11, 12])
        assert (result.runs[0].start_step, result.runs[0].end_step) == (10, 12)

    def test_step_indexes_length_mismatch(self):
        with pytest.raises(ValueError):
            detect_repeated_actions(["x", "x", "x"], step_indexes=[1, 2])

    @pytest.mark.parametrize("bad_step", [0, -1, None, "2", 2.0, True])
    def test_invalid_step_number_makes_item_unusable(self, bad_step):
        result = detect_repeated_actions(["x", "x", "x"], step_indexes=[1, bad_step, 3])
        assert result.flagged is False


# ---------------------------------------------------------------------------
# Schema v1.0 records
# ---------------------------------------------------------------------------

class TestSchemaRecords:
    def test_schema_example_lines_are_not_flagged(self):
        # the three example step lines from the schema document (different actions)
        records = [
            step(1, "go to fridge 1", thought="The potato may be in the fridge."),
            step(2, "open fridge 1", thought="Open it."),
            step(3, "take potato 1 from fridge 1", thought="Take the potato.", retry_count=1),
        ]
        result = check_repeated_actions(records)
        assert result.flagged is False
        assert result.skipped == ()
        assert result.num_steps == 3

    def test_repeated_look_records_flagged_with_logged_step_numbers(self):
        records = [step(1, "go to desk 1")] + looks(2, 3) + [step(5, "take pen 1 from desk 1")]
        result = check_repeated_actions(records)
        assert result.flagged is True
        assert result.runs[0].step_indexes == (2, 3, 4)
        assert result.first_alert_step == 4

    def test_prefix_that_does_not_start_at_step_one(self):
        # the rule uses the logged step numbers; it does not renumber
        result = check_repeated_actions(looks(7, 3))
        assert result.runs[0].step_indexes == (7, 8, 9)

    def test_other_fields_are_ignored(self):
        records = [step(i, "look", thought=f"t{i}", observation=f"obs {i}") for i in (1, 2, 3)]
        assert check_repeated_actions(records).flagged is True

    def test_custom_threshold_via_config(self):
        records = looks(1, 2)
        assert check_repeated_actions(records).flagged is False
        assert check_repeated_actions(records, RepeatedActionConfig(threshold=2)).flagged is True

    def test_empty_records(self):
        result = check_repeated_actions([])
        assert result.flagged is False and result.num_steps == 0

    @pytest.mark.parametrize("bad", [None, "look"])
    def test_wrong_records_type_rejected(self, bad):
        with pytest.raises(TypeError):
            check_repeated_actions(bad)
        with pytest.raises(TypeError):
            extract_actions(bad)

    def test_generator_of_records_accepted(self):
        assert check_repeated_actions(r for r in looks(1, 3)).flagged is True


class TestSummaryRecords:
    def test_summary_is_skipped_silently(self):
        records = looks(1, 2) + [summary()]
        result = check_repeated_actions(records)
        assert result.num_steps == 2
        assert result.skipped == ()
        assert result.flagged is False

    def test_summary_at_end_does_not_hide_a_run(self):
        result = check_repeated_actions(looks(1, 3) + [summary()])
        assert result.flagged is True and result.num_steps == 3

    def test_summary_in_the_middle_does_not_break_a_run(self):
        records = [step(1, "look"), summary(), step(2, "look"), step(3, "look")]
        assert check_repeated_actions(records).flagged is True

    @pytest.mark.parametrize("outcome", ["success", "failure", "max_steps", "error"])
    def test_outcome_never_changes_the_result(self, outcome):
        base = check_repeated_actions(looks(1, 3)).to_dict()
        with_summary = check_repeated_actions(looks(1, 3) + [summary(outcome=outcome)]).to_dict()
        assert with_summary == base

    def test_outcome_field_on_a_step_line_is_ignored(self):
        records = [step(i, "look", outcome="success") for i in (1, 2, 3)]
        assert check_repeated_actions(records).flagged is True

    def test_summary_with_garbage_content_is_still_skipped(self):
        records = looks(1, 3) + [{"type": "episode_summary"}]
        result = check_repeated_actions(records)
        assert result.flagged is True and result.skipped == ()


class TestParseFailures:
    """parse_ok=false means the logger substituted `look`; it is not the agent's action."""

    def test_parse_failure_is_skipped_and_reported(self):
        records = [step(1, "go to desk 1"), step(2, "look", parse_ok=False)]
        result = check_repeated_actions(records)
        assert result.num_steps == 2
        assert len(result.skipped) == 1
        assert result.skipped[0].record_index == 1 and result.skipped[0].step == 2
        assert "parse_ok is false" in result.skipped[0].reason  # distinct from "missing/not a boolean"

    def test_three_substituted_looks_are_not_a_repeated_action(self):
        result = check_repeated_actions(looks(1, 3, parse_ok=False))
        assert result.flagged is False
        assert result.max_run_length == 0
        assert len(result.skipped) == 3

    def test_parse_failure_in_the_middle_interrupts_a_run(self):
        records = [step(1, "look"), step(2, "look", parse_ok=False), step(3, "look")]
        assert check_repeated_actions(records, RepeatedActionConfig(threshold=2)).flagged is False

    def test_parse_failure_does_not_extend_a_real_run(self):
        records = looks(1, 2) + [step(3, "look", parse_ok=False)]
        assert check_repeated_actions(records).flagged is False

    @pytest.mark.parametrize("value", [None, "true", 1, 0])
    def test_missing_or_non_boolean_parse_ok_is_unusable(self, value):
        records = looks(1, 2) + [step(3, "look", parse_ok=value)]
        result = check_repeated_actions(records)
        assert result.flagged is False
        assert "parse_ok" in result.skipped[0].reason

    def test_missing_parse_ok_key(self):
        records = looks(1, 2)
        record = step(3, "look")
        del record["parse_ok"]
        result = check_repeated_actions(records + [record])
        assert result.flagged is False and len(result.skipped) == 1


class TestRetryCount:
    """retry_count counts API retries. It is not an agent action."""

    def test_retries_do_not_add_to_run_length(self):
        records = [step(1, "look", retry_count=0), step(2, "look", retry_count=3), step(3, "look", retry_count=5)]
        result = check_repeated_actions(records)
        assert result.runs[0].run_length == 3  # three agent steps, not 3 + 8

    def test_retries_alone_do_not_make_a_repeat(self):
        result = check_repeated_actions([step(1, "look", retry_count=10)])
        assert result.flagged is False and result.max_run_length == 1

    def test_two_steps_with_many_retries_stay_below_threshold(self):
        records = [step(1, "look", retry_count=4), step(2, "look", retry_count=4)]
        assert check_repeated_actions(records).flagged is False

    def test_retry_count_does_not_affect_the_result(self):
        plain = check_repeated_actions(looks(1, 3, retry_count=0)).to_dict()
        retried = check_repeated_actions(looks(1, 3, retry_count=7)).to_dict()
        assert plain == retried


class TestMissingAndMalformedData:
    def test_missing_action_field_breaks_run_and_is_reported(self):
        gap = step(3, "look")
        del gap["action"]
        records = [step(1, "look"), step(2, "look"), gap, step(4, "look")]
        result = check_repeated_actions(records)
        assert result.flagged is False  # 3 'look' in total, but not consecutive
        assert len(result.skipped) == 1
        assert result.skipped[0].record_index == 2 and "missing" in result.skipped[0].reason

    def test_none_non_string_and_blank_actions_reported(self):
        records = [step(1, None), step(2, "   "), step(3, 42)]
        result = check_repeated_actions(records)
        assert result.flagged is False
        assert [s.record_index for s in result.skipped] == [0, 1, 2]

    def test_non_mapping_record(self):
        records = [step(1, "look"), "not a record", step(3, "look"), step(4, "look")]
        result = check_repeated_actions(records)
        assert result.flagged is False
        assert result.skipped[0].reason == "record is not a mapping"
        assert result.num_steps == 4

    def test_missing_type_is_unusable(self):
        record = step(2, "look")
        del record["type"]
        result = check_repeated_actions([step(1, "look"), record, step(3, "look")])
        assert result.flagged is False
        assert "type" in result.skipped[0].reason

    def test_unknown_type_is_unusable(self):
        result = check_repeated_actions([step(1, "look"), step(2, "look", type="tool_call"), step(3, "look")])
        assert result.flagged is False and len(result.skipped) == 1

    @pytest.mark.parametrize("bad_step", [None, 0, -2, "3", 3.0, True])
    def test_invalid_step_number_makes_record_unusable(self, bad_step):
        records = [step(1, "look"), step(2, "look"), step(bad_step, "look")]
        result = check_repeated_actions(records)
        assert result.flagged is False
        assert "step" in result.skipped[0].reason

    def test_missing_step_key(self):
        record = step(3, "look")
        del record["step"]
        result = check_repeated_actions(looks(1, 2) + [record])
        assert result.flagged is False and result.skipped[0].step is None

    def test_gap_in_logged_step_numbers_is_not_a_run(self):
        records = [step(1, "look"), step(2, "look"), step(4, "look")]
        assert check_repeated_actions(records).flagged is False


class TestNoFutureInformation:
    """The result for a prefix must not depend on anything after the prefix."""

    TRAJECTORY = (
        [step(1, "go to desk 1")] + looks(2, 4) + [step(6, "take pen 1 from desk 1")] + looks(7, 3)
    )  # runs at steps 2-5 (alert 4) and 7-9 (alert 9)

    @pytest.mark.parametrize("k", range(0, 10))
    def test_prefix_flag_state_depends_only_on_steps_up_to_k(self, k):
        prefix = self.TRAJECTORY[:k]
        result = check_repeated_actions(prefix)
        assert result.flagged is (k >= 4)  # first alert at step 4
        if result.flagged:
            assert result.first_alert_step == 4  # same step as with the full trajectory
            assert all(r.end_step <= k for r in result.runs)  # never reports steps beyond the prefix

    def test_full_run_has_both_alerts(self):
        result = check_repeated_actions(self.TRAJECTORY + [summary(outcome="failure")])
        assert [r.alert_step for r in result.runs] == [4, 9]

    def test_alert_step_is_independent_of_how_long_the_run_continues(self):
        short = check_repeated_actions(looks(1, 3))
        long = check_repeated_actions(looks(1, 12))
        assert short.first_alert_step == long.first_alert_step == 3


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class TestOutput:
    def test_result_is_json_serialisable(self):
        result = check_repeated_actions(looks(1, 3) + [step(4, "look", parse_ok=False)])
        data = json.loads(json.dumps(result.to_dict()))
        assert data["flagged"] is True
        assert data["rule_id"] == "repeated_action"
        assert data["taxonomy_id"] == "TX-01"
        assert data["signal_id"] == "SIG-01"
        assert data["first_alert_step"] == 3
        assert data["runs"][0]["step_indexes"] == [1, 2, 3]
        assert data["skipped"][0]["step"] == 4

    def test_summary_text(self):
        assert "not a confirmed failure" in detect_repeated_actions(["x"] * 3).summary
        assert "Not flagged" in detect_repeated_actions(["x"]).summary


# ---------------------------------------------------------------------------
# Configuration (strict validation)
# ---------------------------------------------------------------------------

def write_yaml(tmp_path, text):
    path = tmp_path / "rules.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestConfig:
    def test_real_config_file_matches_code_default(self):
        loaded = load_config(ROOT / "configs" / "rules.yaml")
        assert loaded == RepeatedActionConfig()
        assert loaded.threshold == DEFAULT_THRESHOLD == 3

    def test_default_path_points_at_repo_config(self):
        assert load_config() == load_config(ROOT / "configs" / "rules.yaml")

    def test_custom_threshold_loaded(self, tmp_path):
        assert load_config(write_yaml(tmp_path, "repeated_action:\n  threshold: 5\n")).threshold == 5

    def test_other_rule_sections_are_allowed(self, tmp_path):
        path = write_yaml(tmp_path, "repeated_action:\n  threshold: 4\nsome_future_rule:\n  x: 1\n")
        assert load_config(path).threshold == 4

    def test_dataclass_rejects_bad_threshold(self):
        for bad in (1, 0, 2.5, "3", None, True):
            with pytest.raises(ValueError):
                RepeatedActionConfig(threshold=bad)

    @pytest.mark.parametrize("value", ["1", "0", "-1", "2.5", "'3'", "true", "null", "[3]"])
    def test_invalid_threshold_in_yaml_rejected(self, tmp_path, value):
        with pytest.raises(ValueError):
            load_config(write_yaml(tmp_path, f"repeated_action:\n  threshold: {value}\n"))

    def test_unknown_key_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            load_config(write_yaml(tmp_path, "repeated_action:\n  treshold: 3\n"))

    def test_old_provisional_keys_are_rejected(self, tmp_path):
        text = "repeated_action:\n  threshold: 3\n  normalize: true\n  input:\n    action_field: action\n"
        with pytest.raises(ValueError):
            load_config(write_yaml(tmp_path, text))

    def test_missing_threshold_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            load_config(write_yaml(tmp_path, "repeated_action: {}\n"))

    def test_missing_or_misspelled_section_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            load_config(write_yaml(tmp_path, "repeated_actions:\n  threshold: 3\n"))

    @pytest.mark.parametrize("text", ["", "# only a comment\n", "- 3\n- 4\n", "repeated_action: 3\n", "repeated_action:\n"])
    def test_wrong_shape_rejected(self, tmp_path, text):
        with pytest.raises(ValueError):
            load_config(write_yaml(tmp_path, text))

    def test_missing_config_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")
