import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from make_splits import make_splits  # noqa: E402


def _summaries(success_ids, failure_ids):
    out = {}
    for episode_id in success_ids:
        out[episode_id] = {"episode_id": episode_id, "outcome": "success"}
    for episode_id in failure_ids:
        out[episode_id] = {"episode_id": episode_id, "outcome": "failure"}
    return out


def test_make_splits_is_disjoint_and_covers_every_episode():
    summaries = _summaries(
        success_ids=[f"ep_s{i:02d}" for i in range(10)],
        failure_ids=[f"ep_f{i:02d}" for i in range(20)],
    )
    calibration_ids, test_ids = make_splits(summaries, seed=1)

    assert set(calibration_ids).isdisjoint(test_ids)
    assert set(calibration_ids) | set(test_ids) == set(summaries)


def test_make_splits_is_stratified_by_outcome():
    summaries = _summaries(
        success_ids=[f"ep_s{i:02d}" for i in range(10)],
        failure_ids=[f"ep_f{i:02d}" for i in range(20)],
    )
    calibration_ids, test_ids = make_splits(summaries, seed=1)

    cal_success = sum(1 for i in calibration_ids if summaries[i]["outcome"] == "success")
    cal_failure = sum(1 for i in calibration_ids if summaries[i]["outcome"] == "failure")
    test_success = sum(1 for i in test_ids if summaries[i]["outcome"] == "success")
    test_failure = sum(1 for i in test_ids if summaries[i]["outcome"] == "failure")

    assert cal_success == test_success == 5
    assert cal_failure == test_failure == 10


def test_make_splits_handles_an_odd_count_without_losing_an_episode():
    summaries = _summaries(success_ids=[f"ep_s{i:02d}" for i in range(7)], failure_ids=[])
    calibration_ids, test_ids = make_splits(summaries, seed=1)

    assert len(calibration_ids) + len(test_ids) == 7
    assert len(test_ids) >= len(calibration_ids)  # the odd one out goes to test, never calibration


def test_make_splits_is_deterministic_for_a_fixed_seed():
    summaries = _summaries(
        success_ids=[f"ep_s{i:02d}" for i in range(10)],
        failure_ids=[f"ep_f{i:02d}" for i in range(20)],
    )
    first = make_splits(summaries, seed=42)
    second = make_splits(summaries, seed=42)

    assert first == second
