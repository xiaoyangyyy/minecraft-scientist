from types import SimpleNamespace

import pytest

from minecraft_scientist.exploration.public_wait_recovery import settle_with_recovery


def state(stable_at, *, anchor=48, fail_at=None):
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
            result["terminal"] = "geometry_failed"
            return False
        result["flow_prefixes"].append({"stable": len(result["steps"]) >= stable_at})
        return True

    return result, step, ordinal


STABILITY = SimpleNamespace(
    minimum_wait_ticks=8, maximum_wait_ticks=16, stable_prefixes=3
)


def test_stability_uses_first_eligible_prefix_and_no_extra_window():
    result, step, ordinal = state(1)
    assert settle_with_recovery(result, "initial", step, ordinal, STABILITY)
    assert len(result["steps"]) == 8 and not result["failed_wait_windows"]
    assert result["settles"][0]["interval"] == [48, 56]


def test_fall_window_remains_exhausted_after_first_recovered_tail():
    result, step, ordinal = state(17)
    assert settle_with_recovery(result, "acquisition", step, ordinal, STABILITY)
    assert len(result["steps"]) == 19
    assert result["failed_wait_windows"][0]["interval"] == [48, 64]
    assert (
        result["failed_wait_windows"][0]["recorded_terminal"]
        == "acquisition_wait_exhausted"
    )
    assert result["recovery_wait_windows"][0]["interval"] == [64, 67]
    assert result["recovery_wait_windows"][0]["status"] == "stable"
    assert result["settles"][0]["stable_tail"] == [[64, 65], [65, 66], [66, 67]]


def test_stable_streak_is_not_reset_or_shortened_at_window_boundary():
    result, step, ordinal = state(15)
    assert settle_with_recovery(result, "future", step, ordinal, STABILITY)
    assert len(result["steps"]) == 17
    assert result["settles"][0]["stable_tail"] == [[62, 63], [63, 64], [64, 65]]


def test_all_three_failed_windows_remain_and_recovery_is_finite():
    result, step, ordinal = state(100)
    assert not settle_with_recovery(result, "initial", step, ordinal, STABILITY)
    assert len(result["steps"]) == 48
    assert len(result["failed_wait_windows"]) == 3
    assert len(result["recovery_wait_windows"]) == 2
    assert result["terminal"] == "initial_wait_exhausted"
    assert all(
        window["status"] == "exhausted" for window in result["recovery_wait_windows"]
    )


def test_episode_budget_prevents_extra_actual_action():
    result, step, ordinal = state(100, anchor=280)
    assert not settle_with_recovery(result, "future", step, ordinal, STABILITY)
    assert len(result["steps"]) == 8 and ordinal() == 288
    assert result["terminal"] == "budget_exhausted"
    assert not result["recovery_wait_windows"]


def test_native_error_keeps_partial_recovery_interval():
    result, step, ordinal = state(100, fail_at=18)
    assert not settle_with_recovery(result, "acquisition", step, ordinal, STABILITY)
    assert len(result["steps"]) == 18
    assert result["recovery_wait_windows"][0]["interval"] == [64, 66]
    assert result["recovery_wait_windows"][0]["status"] == "geometry_failed"


def test_recovery_budget_is_shared_across_phases():
    result, step, ordinal = state(17)
    assert settle_with_recovery(result, "initial", step, ordinal, STABILITY)
    first_count = len(result["steps"])

    # The next phase remains moving through its own initial window.
    def moving(kind, handle):
        answer = step(kind, handle)
        if answer:
            result["flow_prefixes"][-1]["stable"] = (
                len(result["steps"]) >= first_count + 33
            )
        return answer

    assert not settle_with_recovery(result, "acquisition", moving, ordinal, STABILITY)
    assert len(result["recovery_wait_windows"]) == 2
    assert len(result["steps"]) == first_count + 32


def test_unknown_phase_cannot_execute_controls():
    result, step, ordinal = state(1)
    with pytest.raises(ValueError, match="public experiment phase"):
        settle_with_recovery(result, "private-timing", step, ordinal, STABILITY)
    assert result["steps"] == []
