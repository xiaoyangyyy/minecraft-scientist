import numpy as np
import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration.temporal_tracks import TemporalTracks
from minecraft_scientist.exploration.visibility_acquisition import (
    VisibilityController,
    VisibilityDecision,
    VisibilityPolicy,
)


def tracks(tick, ids=tuple(range(80)), *, shift=(0, 0), count=80):
    seeds = np.array([(65 + i % 10 * 9, 65 + i // 10 * 9) for i in range(count)])
    return TemporalTracks(
        image_shape=(224, 224),
        frame_count=tick + 1,
        seed_count=count,
        surviving_counts=(count,) * tick + (len(ids),),
        track_ids=ids,
        pixels=tuple(
            tuple(map(tuple, seeds[list(ids)] + frame * np.array(shift)))
            for frame in range(tick + 1)
        ),
        minimum_patch_correlations=(1,) * len(ids),
        chain_cycle_errors=(0,) * len(ids),
    )


@pytest.mark.parametrize("handles", [(0, 1), (18, 96), (11, 4)])
def test_handle_names_do_not_affect_visibility_or_bounded_completion(handles):
    controller = VisibilityController(handles)
    for tick in range(1, 11):
        decision = controller.observe(tracks(tick))
        if tick < 6:
            assert (
                decision.instruction == "continue"
                and controller.next_handle == handles[0]
            )
        elif tick == 6:
            assert decision.instruction == "switch" and decision.endpoint_tick == 6
            assert controller.next_handle == handles[1]
        elif tick < 10:
            assert decision.instruction == "continue" and decision.phase == "second"
        else:
            assert decision.instruction == "finish" and decision.holdout_ticks == (
                7,
                10,
            )
            assert controller.next_handle is None
    with pytest.raises(ValueError, match="terminal"):
        controller.observe(tracks(11))


def test_retention_loss_requests_earlier_control_and_never_replenishes_ids():
    controller = VisibilityController((6, 1))
    controller.observe(tracks(1))
    decision = controller.observe(tracks(2, tuple(range(50))))
    assert decision.instruction == "switch" and decision.endpoint_tick == 2
    decision = controller.observe(tracks(3))
    assert (
        decision.retained_ids == tuple(range(50))
        and decision.retention_fraction == 50 / 80
        and decision.phase_initial_track_count == 50
        and decision.phase_retention_fraction == 1
    )
    decision = controller.observe(tracks(4))
    assert decision.instruction == "continue"
    assert controller.observe(tracks(5)).instruction == "continue"
    decision = controller.observe(tracks(6))
    assert decision.instruction == "finish" and decision.holdout_ticks == (3, 6)


def test_second_phase_new_loss_triggers_at_minimum_without_replenishment():
    controller = VisibilityController((6, 1))
    controller.observe(tracks(1))
    switched = controller.observe(tracks(2, tuple(range(50))))
    assert switched.phase_initial_track_count == 80
    early = controller.observe(tracks(3, tuple(range(35))))
    assert early.instruction == "continue"  # Minimum duration remains two ticks.
    decision = controller.observe(tracks(4))  # Raw tracker returns lost IDs.
    assert decision.instruction == "finish" and decision.holdout_ticks == (3, 4)
    assert decision.retained_ids == tuple(range(35))
    assert decision.initial_track_count == 80
    assert decision.retention_fraction == 35 / 80
    assert decision.phase_initial_track_count == 50
    assert decision.phase_retention_fraction == 35 / 50


def test_second_phase_absolute_support_aborts_even_before_loss_trigger():
    controller = VisibilityController((6, 1))
    controller.observe(tracks(1))
    controller.observe(tracks(2, tuple(range(40))))
    decision = controller.observe(tracks(3, tuple(range(31))))
    assert decision.phase_retention_fraction > controller.policy.retention_trigger
    assert decision.instruction == "abort" and not decision.holdout_ticks


@pytest.mark.parametrize(
    "updates",
    [
        {"phase_initial_track_count": 81},
        {"phase_initial_track_count": 30},
        {"phase_initial_track_count": 80, "phase_retention_fraction": 50 / 80},
        {"phase_retention_fraction": 0.9},
    ],
)
def test_phase_retention_schema_rejects_inconsistent_denominators(updates):
    controller = VisibilityController((6, 1))
    controller.observe(tracks(1))
    controller.observe(tracks(2, tuple(range(50))))
    decision = controller.observe(tracks(3, tuple(range(35))))
    forged = decision.model_dump() | updates
    with pytest.raises(ValidationError):
        VisibilityDecision.model_validate(forged)


def test_first_phase_normalizer_and_policy_version_are_explicit():
    policy = VisibilityPolicy()
    assert policy.model_dump()["retention_reference"] == "phase_start_v2"
    with pytest.raises(ValidationError):
        VisibilityPolicy(retention_reference="global_v1")
    decision = VisibilityController((0, 1)).observe(tracks(1, tuple(range(50))))
    assert decision.phase_initial_track_count == decision.initial_track_count == 80
    assert decision.phase_retention_fraction == decision.retention_fraction == 50 / 80
    with pytest.raises(ValidationError):
        VisibilityDecision.model_validate(
            decision.model_dump()
            | {"phase_initial_track_count": 50, "phase_retention_fraction": 1}
        )


def test_first_action_loss_is_not_normalized_out_of_retention():
    controller = VisibilityController((6, 1))
    first = controller.observe(tracks(1, tuple(range(50))))
    assert first.initial_track_count == 80 and first.retention_fraction == 50 / 80
    second = controller.observe(tracks(2, tuple(range(50))))
    assert second.instruction == "switch" and second.endpoint_tick == 2


def test_projected_edges_request_switch_before_actual_tracks_are_lost():
    controller = VisibilityController((0, 2))
    controller.observe(tracks(1, shift=(12, 12)))
    decision = controller.observe(tracks(2, shift=(12, 12)))
    assert len(decision.retained_ids) == 80
    assert decision.projected_edge_fraction >= 0.25
    assert decision.instruction == "switch" and decision.endpoint_tick == 2


@pytest.mark.parametrize("tick", [1, 3])
def test_insufficient_tracks_abort_not_false_success_or_reseed(tick):
    controller = VisibilityController((0, 1))
    for before in range(1, tick):
        controller.observe(tracks(before))
    decision = controller.observe(tracks(tick, tuple(range(20))))
    assert decision.instruction == "abort" and not decision.holdout_ticks
    assert "not a pose contradiction" in decision.reason


def test_invalid_observations_do_not_mutate_history_or_skip_frames():
    controller = VisibilityController((0, 1))
    first = controller.observe(tracks(1))
    for corrupted in (
        tracks(3),
        tracks(2).model_copy(update={"seed_count": 81}),
        tracks(2, shift=(0.1, 0)),
    ):
        with pytest.raises(ValueError):
            controller.observe(corrupted)
        assert controller.decisions == (first,) and controller.next_handle == 0
    assert controller.observe(tracks(2)).instruction == "continue"


def test_distributed_support_is_required():
    controller = VisibilityController((0, 1))
    narrow = tracks(1)
    narrow = narrow.model_copy(update={"pixels": (((80.0, 80.0),) * 80,) * 2})
    assert controller.observe(narrow).instruction == "abort"


@pytest.mark.parametrize("handles", [(1, 1), (-1, 0), (True, 2), (0, 1, 2)])
def test_invalid_controls_reject(handles):
    with pytest.raises(ValueError):
        VisibilityController(handles)


def test_bounds_and_fake_views_reject():
    with pytest.raises(ValidationError):
        VisibilityPolicy(maximum_first_ticks=2, minimum_leg_ticks=3)
    with pytest.raises(ValidationError):
        VisibilityPolicy(
            maximum_first_ticks=12,
            maximum_second_ticks=12,
            temporal={"maximum_frames": 10},
        )
    decision = VisibilityController((0, 1)).observe(tracks(1))
    forged = decision.model_dump()
    forged.update(instruction="finish", next_handle=None, holdout_ticks=(2, 3))
    with pytest.raises(ValidationError):
        VisibilityDecision.model_validate(forged)
    assert (
        not {
            "supported",
            "unit_translation",
            "relative_rotation",
            "navigation_accepted",
        }
        & VisibilityDecision.model_fields.keys()
    )


def test_empty_images_abort_deterministically_without_optional_cv():
    empty = TemporalTracks(
        image_shape=(224, 224),
        frame_count=2,
        seed_count=0,
        surviving_counts=(0, 0),
        track_ids=(),
        pixels=((), ()),
        minimum_patch_correlations=(),
        chain_cycle_errors=(),
    )
    first = VisibilityController((0, 1)).observe(empty)
    assert first == VisibilityController((0, 1)).observe(empty)
    assert first.instruction == "abort" and first.retention_fraction == 0
