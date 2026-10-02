import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")
from test_multiview_motion import CAMERA, _tracks

from minecraft_scientist.exploration.action_switch_motion import (
    ActionSwitchEvidence,
    inspect_action_switch,
    plan_action_switch,
)
from minecraft_scientist.exploration.keyframe_motion import KeyframeAttempt
from minecraft_scientist.exploration.multiview_motion import estimate_multiview_motion
from minecraft_scientist.exploration.pose_direction_search import (
    DirectionSearchConfig,
    _map_validation,
)
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks


@pytest.fixture
def motion():
    prefix = _tracks(frame_count=5)
    rng = np.random.default_rng(123)
    xyz = np.column_stack(
        (rng.uniform(-2, 2, 200), rng.uniform(-2, 2, 200), rng.uniform(4, 9, 200))
    )
    rotation = cv2.Rodrigues(np.array((0.01, 0.04, -0.02)))[0]
    future = []
    for index in range(1, 5):
        shift = np.array((-0.7, 0.05, -0.15)) + np.array((0.05, -0.1, -0.3)) * index / 4
        projected = (xyz @ rotation.T + shift) @ CAMERA.T
        future.append(tuple(map(tuple, projected[:, :2] / projected[:, 2:])))
    extended = TemporalTracks.model_validate(
        prefix.model_dump()
        | {
            "frame_count": 9,
            "pixels": prefix.pixels + tuple(future),
            "surviving_counts": (200,) * 9,
        }
    )
    candidate = KeyframeAttempt(
        end_offset=4,
        tracks=prefix,
        motion=estimate_multiview_motion(prefix, CAMERA, seed=7),
    )
    assert candidate.motion.supported
    plan = plan_action_switch(
        candidate,
        [
            {"start_frame": 0, "end_frame": 4, "kind": "move", "motion_handle": 0},
            {"start_frame": 4, "end_frame": 8, "kind": "move", "motion_handle": 1},
        ],
        anchor_frame=0,
        frame_count=9,
    )
    return candidate, extended, plan


def test_switched_views_validate_frozen_pose_without_reselection(motion):
    candidate, extended, plan = motion
    frozen, camera = candidate.model_dump_json(), CAMERA.copy()
    result = inspect_action_switch(candidate, extended, plan, camera, seed=7)
    assert result.reference_future.status == "consistent"
    assert result.comparison == "no_prefit_witness"
    assert tuple(item.frame_slot for item in result.reference_future.frame_checks) == (
        6,
        8,
    )
    assert (
        result.reference_future.training_track_ids
        == candidate.motion.training_track_ids
    )
    assert (
        result.reference_future.validation_track_ids
        == candidate.motion.validation_track_ids
    )
    assert len(result.alternatives) == 16
    assert candidate.model_dump_json() == frozen
    np.testing.assert_array_equal(camera, CAMERA)
    assert inspect_action_switch(candidate, extended, plan, camera, seed=7) == result


def test_future_withheld_corruption_never_changes_prefit_search(motion):
    candidate, extended, plan = motion
    original = inspect_action_switch(candidate, extended, plan, CAMERA, seed=7)
    pixels = np.asarray(extended.pixels).copy()
    ids = [
        extended.track_ids.index(item) for item in candidate.motion.validation_track_ids
    ]
    pixels[5:, ids, 0] += 12
    corrupted = TemporalTracks.model_validate(
        extended.model_dump() | {"pixels": pixels.tolist()}
    )
    result = inspect_action_switch(candidate, corrupted, plan, CAMERA, seed=7)
    assert result.search == original.search
    assert result.reference_future.status == "inconsistent"
    assert result.comparison == "reference_inconsistent"
    assert all(not item.supported for item in result.reference_future.frame_checks)


def test_lost_validation_and_changed_prefix_are_unavailable_not_contradictions(motion):
    candidate, extended, plan = motion
    training = set(candidate.motion.training_track_ids)
    selected = [
        index for index, item in enumerate(extended.track_ids) if item in training
    ]
    values = extended.model_dump()
    for name in ("track_ids", "minimum_patch_correlations", "chain_cycle_errors"):
        values[name] = [values[name][index] for index in selected]
    values["pixels"] = [
        [frame[index] for index in selected] for frame in values["pixels"]
    ]
    result = inspect_action_switch(
        candidate, TemporalTracks.model_validate(values), plan, CAMERA, seed=7
    )
    assert (
        result.comparison == "unavailable" and not result.reference_future.frame_checks
    )
    assert result.reference_future.status == "unavailable"
    pixels = np.asarray(extended.pixels).copy()
    pixels[1, :, 0] += 0.01
    changed = TemporalTracks.model_validate(
        extended.model_dump() | {"pixels": pixels.tolist()}
    )
    result = inspect_action_switch(candidate, changed, plan, CAMERA, seed=7)
    assert (
        result.reference_future.status == "unavailable"
        and "prefix" in result.reference_future.reason
    )


def test_missing_future_images_do_not_certify_original_or_reject_alternatives(motion):
    candidate, _, plan = motion
    result = inspect_action_switch(candidate, None, plan, CAMERA, seed=7)
    assert result.comparison == "unavailable"
    assert result.reference_future.status == "unavailable"


def test_schema_rejects_reassigned_ids_missing_trials_and_fake_comparison(motion):
    result = inspect_action_switch(*motion[:2], motion[2], CAMERA, seed=7).model_dump()
    for change in (
        {"comparison": "tested_alternatives_rejected"},
        {"alternatives": result["alternatives"][:-1]},
        {
            "reference_future": result["reference_future"]
            | {"validation_track_ids": result["reference_future"]["training_track_ids"]}
        },
    ):
        with pytest.raises(ValidationError):
            ActionSwitchEvidence.model_validate(result | change)


def test_declared_frozen_search_cannot_differ_from_prefix_only_fit(motion):
    candidate, extended, plan = motion
    result = inspect_action_switch(candidate, extended, plan, CAMERA, seed=7)
    forged = result.search.model_copy(update={"reason": "changed after viewing future"})
    with pytest.raises(ValueError, match="frozen search"):
        inspect_action_switch(
            candidate, extended, plan, CAMERA, seed=7, frozen_search=forged
        )


def test_plausible_alternatives_use_identical_future_partitions_and_frozen_fit(motion):
    # Known-pose planar fixture; not a claim that baseline inference selects it.
    candidate, _, plan = motion
    rng = np.random.default_rng(123)
    xyz = np.column_stack(
        (rng.uniform(-2, 2, 200), rng.uniform(-2, 2, 200), np.full(200, 16.0))
    )
    pixels = []
    for slot in range(9):
        alpha = min(1, slot / 4)
        rotation = cv2.Rodrigues(np.array((0.01, 0.04, -0.02)) * alpha)[0]
        shift = np.array((-0.7, 0.05, -0.15)) * alpha
        if slot > 4:
            shift += np.array((0.05, -0.1, -0.3)) * (slot - 4) / 4
        projected = (xyz @ rotation.T + shift) @ CAMERA.T
        pixels.append((projected[:, :2] / projected[:, 2:]).tolist())
    prefix = TemporalTracks.model_validate(
        candidate.tracks.model_dump() | {"pixels": pixels[:5]}
    )
    declared = candidate.model_copy(update={"tracks": prefix})
    reference = _map_validation(
        declared,
        CAMERA,
        np.asarray(candidate.motion.candidate.relative_rotation),
        np.asarray(candidate.motion.candidate.unit_translation),
        DirectionSearchConfig().geometry,
        7,
        cv2,
    )
    assert reference.supported
    declared = declared.model_copy(
        update={
            "motion": candidate.motion.model_copy(
                update={"frame_checks": reference.frame_checks}
            )
        }
    )
    extended = TemporalTracks.model_validate(
        prefix.model_dump()
        | {"pixels": pixels, "frame_count": 9, "surviving_counts": (200,) * 9}
    )
    result = inspect_action_switch(declared, extended, plan, CAMERA, seed=7)
    assert result.search.ambiguity_detected
    eligible = [item for item in result.alternatives if item.prefit_plausible]
    assert eligible and all(item.future is not None for item in eligible)
    assert all(
        item.future.training_track_ids == result.reference_future.training_track_ids
        and item.future.validation_track_ids
        == result.reference_future.validation_track_ids
        for item in eligible
    )
    lost = inspect_action_switch(declared, None, plan, CAMERA, seed=7)
    assert result.search == lost.search
    assert lost.comparison == "unavailable"
    assert all(
        item.future.status == "unavailable"
        for item in lost.alternatives
        if item.prefit_plausible
    )


@pytest.mark.parametrize("seed", [-1, True, 2**31 - 1])
def test_invalid_seed_rejects_before_geometry(motion, seed):
    with pytest.raises(ValueError, match="seed"):
        inspect_action_switch(*motion[:2], motion[2], CAMERA, seed=seed)
