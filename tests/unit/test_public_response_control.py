import copy

import pytest

from minecraft_scientist.exploration.public_response_control import (
    select_public_response,
)


def records(outcomes):
    return [
        {"trial": trial, "handle": handle, "selected": selected}
        for trial, (handle, selected) in enumerate(outcomes)
    ]


def test_initial_tie_break_and_failure_explore_untried_handles():
    assert select_public_response([])["handle"] == 0
    assert select_public_response(records([(0, False)]))["handle"] == 1
    assert select_public_response(records([(0, False), (1, False)]))["handle"] == 2


@pytest.mark.parametrize("useful", [0, 1, 2, 3])
def test_control_quality_is_learned_not_assigned_by_heading(useful):
    history = records([(handle, handle == useful) for handle in range(4)])
    decision = select_public_response(history)
    assert decision["handle"] == useful
    assert decision["statistics"][useful]["novel_views"] == 1
    assert decision["observed_trials"] == 4


def test_four_trial_run_forces_a_different_control_then_can_return():
    history = records([(0, False), (1, True), (1, False)] + [(2, True)] * 4)
    decision = select_public_response(history)
    assert decision["handle"] == 3 and decision["excluded_handle"] == 2
    history.append({"trial": 7, "handle": 3, "selected": False})
    decision = select_public_response(history)
    assert decision["handle"] == 2 and decision["excluded_handle"] is None


def test_selection_does_not_use_future_outcomes_or_mutate_history():
    history = records([(0, False), (1, True)])
    before = copy.deepcopy(history)
    decision = select_public_response(history)
    assert decision["handle"] == 1 and history == before
    extended = history + [{"trial": 2, "handle": 1, "selected": False}]
    assert select_public_response(extended)["handle"] == 2
    assert select_public_response(history) == decision


@pytest.mark.parametrize(
    "history",
    [
        [{"trial": 0, "handle": True, "selected": False}],
        [{"trial": 0, "handle": 1, "selected": 1}],
        [{"trial": 1, "handle": 1, "selected": True}],
        [{"trial": 0, "handle": 4, "selected": False}],
        [{"handle": 1, "selected": True}],
    ],
)
def test_malformed_or_relabelled_history_is_not_public_evidence(history):
    with pytest.raises(ValueError, match="chronological public"):
        select_public_response(history)


@pytest.mark.parametrize("handle", [0, 1, 2, 3])
def test_recorded_run_cannot_bypass_bounded_control_selection(handle):
    with pytest.raises(ValueError, match="four-trial control bound"):
        select_public_response(records([(handle, True)] * 5))


def test_run_length_is_derived_from_history_not_caller_assertion():
    decision = select_public_response(records([(2, True), (2, True), (3, False)]))
    assert decision["previous_handle"] == 3 and decision["consecutive_trials"] == 1


def test_future_control_uses_only_past_feedback_and_is_distinct():
    history = records(
        [(0, False), (1, True), (1, False)] + [(2, True)] * 4 + [(3, False)]
    )
    assert select_public_response(history)["handle"] == 2
    decision = select_public_response(history, required_distinct_from=2)
    assert decision["handle"] == 1
    assert not decision["statistics"][2]["eligible"]
    assert decision["required_distinct_from"] == 2


@pytest.mark.parametrize("handle", [True, -1, 4, "2"])
def test_future_exclusion_cannot_be_a_private_heading_or_untyped_value(handle):
    with pytest.raises(ValueError, match="opaque handle"):
        select_public_response([], required_distinct_from=handle)


def test_registered_trial_budget_is_not_extended():
    with pytest.raises(ValueError, match="eleven-trial budget"):
        select_public_response(records([(trial % 4, True) for trial in range(12)]))


@pytest.mark.parametrize("record", [None, []])
def test_nonrecord_values_are_not_public_outcomes(record):
    with pytest.raises(ValueError, match="chronological public"):
        select_public_response([record])
