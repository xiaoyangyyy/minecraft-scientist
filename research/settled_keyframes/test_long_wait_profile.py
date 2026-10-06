from dataclasses import FrozenInstanceError

import pytest
from long_wait_profile import WaitWindow, execute, profile
from pydantic import ValidationError

from minecraft_scientist.exploration.stable_keyframe_map import (
    StableJumpSafeSeedKeyframePolicy,
)

POLICY = StableJumpSafeSeedKeyframePolicy().model_dump(mode="json")


def trajectory(stable_at, *, anchor=48, fail_at=None, failure="execution_failed"):
    """Synthetic public booleans test budgets, not pixel/native success."""
    result = {
        "steps": [],
        "flow_prefixes": [],
        "settles": [],
        "failed_wait_windows": [],
        "recovery_wait_windows": [],
        "terminal": "not_started",
    }

    def ordinal():
        return anchor + len(result["steps"])

    def step(kind, handle):
        assert kind == "wait" and handle is None
        if ordinal() >= 288:
            result["terminal"] = "budget_exhausted"
            return False
        result["steps"].append(ordinal() + 1)
        if fail_at is not None and len(result["steps"]) == fail_at:
            result["terminal"] = failure
            return False
        result["flow_prefixes"].append({"stable": len(result["steps"]) >= stable_at})
        return True

    return result, step, ordinal


def test_only_post_motor_phases_get_new_window():
    assert profile(POLICY, "initial").maximum_wait_ticks == 16
    assert profile(POLICY, "acquisition").maximum_wait_ticks == 24
    assert profile(POLICY, "future").maximum_wait_ticks == 24
    assert StableJumpSafeSeedKeyframePolicy().stability.maximum_wait_ticks == 16


def test_late_public_convergence_uses_first_tail_without_consuming_recovery():
    result, step, ordinal = trajectory(18)
    assert execute(result, "acquisition", step, ordinal, POLICY)
    assert len(result["steps"]) == 20
    assert not result["failed_wait_windows"] and not result["recovery_wait_windows"]
    assert result["settles"][0]["stable_tail"] == [[65, 66], [66, 67], [67, 68]]


def test_early_convergence_is_not_delayed_to_new_maximum():
    result, step, ordinal = trajectory(1)
    assert execute(result, "future", step, ordinal, POLICY)
    assert len(result["steps"]) == 8


def test_full_budget_retains_every_failed_window_and_never_loops_forever():
    result, step, ordinal = trajectory(100)
    assert not execute(result, "acquisition", step, ordinal, POLICY)
    assert len(result["steps"]) == 72
    assert [window["interval"] for window in result["failed_wait_windows"]] == [
        [48, 72],
        [72, 96],
        [96, 120],
    ]
    assert len(result["recovery_wait_windows"]) == 2
    assert all(
        window["status"] == "exhausted" for window in result["recovery_wait_windows"]
    )


def test_streak_spans_window_boundary_without_reset_or_shortcut():
    result, step, ordinal = trajectory(23)
    assert execute(result, "future", step, ordinal, POLICY)
    assert len(result["steps"]) == 25
    assert result["failed_wait_windows"][0]["interval"] == [48, 72]
    assert result["settles"][0]["stable_tail"] == [[70, 71], [71, 72], [72, 73]]


def test_recovery_windows_are_episode_wide_across_different_phase_profiles():
    result, step, ordinal = trajectory(17)
    assert execute(result, "initial", step, ordinal, POLICY)
    first = len(result["steps"])

    def moving(kind, handle):
        status = step(kind, handle)
        if status:
            result["flow_prefixes"][-1]["stable"] = False
        return status

    assert not execute(result, "future", moving, ordinal, POLICY)
    assert len(result["steps"]) == first + 48
    assert len(result["recovery_wait_windows"]) == 2
    assert result["failed_wait_windows"][0]["interval"] == [48, 64]


def test_new_window_cannot_borrow_actions_past_global_cap():
    result, step, ordinal = trajectory(100, anchor=280)
    assert not execute(result, "future", step, ordinal, POLICY)
    assert len(result["steps"]) == 8 and ordinal() == 288
    assert result["terminal"] == "budget_exhausted"


@pytest.mark.parametrize("failure", ["execution_failed", "geometry_failed"])
def test_partial_second_window_keeps_actual_failure(failure):
    result, step, ordinal = trajectory(100, fail_at=26, failure=failure)
    assert not execute(result, "acquisition", step, ordinal, POLICY)
    assert result["recovery_wait_windows"][0]["interval"] == [72, 74]
    assert result["recovery_wait_windows"][0]["status"] == failure


@pytest.mark.parametrize(
    "kwargs",
    [{"maximum_wait_ticks": 25}, {"minimum_wait_ticks": True}, {"stable_prefixes": 2}],
)
def test_profile_is_exact_and_frozen(kwargs):
    with pytest.raises(ValueError, match="exact"):
        WaitWindow(**kwargs)
    with pytest.raises(FrozenInstanceError):
        WaitWindow().maximum_wait_ticks = 25


def test_rejected_geometry_or_unknown_phase_never_dispatches_waits():
    result, step, ordinal = trajectory(1)
    with pytest.raises(ValueError, match="public experiment phase"):
        execute(result, "private-player-rest", step, ordinal, POLICY)
    changed = {
        **POLICY,
        "temporal": {**POLICY["temporal"], "minimum_patch_correlation": 0.5},
    }
    with pytest.raises(ValidationError, match="registered geometry"):
        execute(result, "acquisition", step, ordinal, changed)
    assert result["steps"] == []
