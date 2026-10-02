from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration import action_switch_motion as switch


def _candidate(offset=4):
    return SimpleNamespace(end_offset=offset, motion=SimpleNamespace(supported=True))


def _timeline(handles=(8, 3)):
    return [
        {"start_frame": 0, "end_frame": 4, "kind": "move", "motion_handle": handles[0]},
        {"start_frame": 4, "end_frame": 8, "kind": "wait"},
        {
            "start_frame": 8,
            "end_frame": 16,
            "kind": "move",
            "motion_handle": handles[1],
        },
    ]


@pytest.mark.parametrize("handles", [(8, 3), (0, 91), (19, 1), (12, 2)])
def test_view_planning_uses_public_handle_switch_not_handle_meaning(handles):
    result = switch.plan_action_switch(
        _candidate(), _timeline(handles), anchor_frame=0, frame_count=17
    )
    assert result.available and result.target_frames == (12, 16)
    assert result.current_handle == handles[0] and result.next_handle == handles[1]
    assert "physical diversity unknown" in result.reason


def test_unchanged_handle_waits_and_bounds_are_not_diverse_view_evidence():
    repeated = switch.plan_action_switch(
        _candidate(), _timeline((8, 8)), anchor_frame=0, frame_count=17
    )
    assert not repeated.available and not repeated.target_frames
    too_long = switch.plan_action_switch(
        _candidate(),
        _timeline(),
        anchor_frame=0,
        frame_count=17,
        config=switch.ActionSwitchConfig(temporal={"maximum_frames": 12}),
    )
    assert not too_long.available and too_long.requested_frames == (12, 16)


@pytest.mark.parametrize(
    "change", ["gap", "overlap", "missing_end", "move_without_handle"]
)
def test_incomplete_or_misaligned_public_timeline_rejects(change):
    timeline = _timeline()
    if change == "gap":
        timeline[1]["start_frame"] = 5
    elif change == "overlap":
        timeline[1]["start_frame"] = 3
    elif change == "missing_end":
        timeline.pop()
    else:
        timeline[0].pop("motion_handle")
    with pytest.raises(ValueError):
        switch.plan_action_switch(
            _candidate(), timeline, anchor_frame=0, frame_count=17
        )


def test_no_candidate_is_lazy_and_no_holdout_is_not_success(monkeypatch):
    monkeypatch.setattr(
        switch, "_opencv", lambda: pytest.fail("empty candidate loaded OpenCV")
    )
    plan = switch.plan_action_switch(None, _timeline(), anchor_frame=0, frame_count=17)
    result = switch.inspect_action_switch(None, None, plan, np.eye(3))
    assert result.comparison == "no_candidate" and result.reference_future is None
    assert not result.alternatives
    assert (
        not {"supported", "relative_rotation", "unit_translation"}
        & switch.ActionSwitchEvidence.model_fields.keys()
    )


def test_unavailable_or_fake_success_schema_rejects():
    with pytest.raises(ValidationError):
        switch.ActionSwitchPlan(
            anchor_frame=0, available=True, reason="forged", target_frames=(6, 8)
        )
    with pytest.raises(ValidationError):
        switch.FrozenFutureCheck(status="consistent", reason="forged")
    with pytest.raises(ValidationError):
        switch.ActionSwitchConfig(prefix_pixel_tolerance=float("nan"))


@pytest.mark.parametrize(
    "reference_status,alternative_status,expected",
    [
        ("consistent", "consistent", "ambiguity_survives"),
        ("consistent", "inconsistent", "tested_alternatives_rejected"),
        ("consistent", "unavailable", "incomplete_comparison"),
        ("consistent", None, "incomplete_comparison"),
        ("inconsistent", "consistent", "reference_inconsistent"),
        ("unavailable", "consistent", "unavailable"),
    ],
)
def test_unavailable_alternatives_and_failed_reference_cannot_be_rescued(
    reference_status, alternative_status, expected
):
    result = switch._comparison(
        SimpleNamespace(available=True),
        SimpleNamespace(reference=True, reference_matches_original_checks=True),
        SimpleNamespace(status=reference_status),
        [
            SimpleNamespace(
                prefit_plausible=True,
                future=None
                if alternative_status is None
                else SimpleNamespace(status=alternative_status),
            )
        ],
    )
    assert result == expected
