import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")
from test_multiview_motion import CAMERA, _tracks

from minecraft_scientist.exploration.keyframe_motion import KeyframeAttempt
from minecraft_scientist.exploration.multiview_motion import estimate_multiview_motion
from minecraft_scientist.exploration.shadow_motion import (
    ShadowMotionConfig,
    ShadowMotionEvidence,
    validate_shadow_motion,
)
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks


@pytest.fixture
def motion():
    prefix = _tracks(frame_count=5)
    rng = np.random.default_rng(123)
    xyz = np.column_stack(
        (rng.uniform(-2, 2, 200), rng.uniform(-2, 2, 200), rng.uniform(4, 9, 200))
    )
    future = []
    for alpha in (1.25, 1.5):
        rotation = cv2.Rodrigues(np.array((0.01, 0.04, -0.02)) * alpha)[0]
        projected = (
            xyz @ rotation.T + np.array((-0.7, 0.05, -0.15)) * alpha
        ) @ CAMERA.T
        future.append(tuple(map(tuple, projected[:, :2] / projected[:, 2:])))
    extended = TemporalTracks.model_validate(
        prefix.model_dump()
        | {
            "frame_count": 7,
            "pixels": prefix.pixels + tuple(future),
            "surviving_counts": (200,) * 7,
        }
    )
    candidate = KeyframeAttempt(
        end_offset=4, tracks=prefix, motion=estimate_multiview_motion(prefix, CAMERA)
    )
    assert candidate.motion.supported
    return candidate, extended


def test_future_holdout_retains_the_original_pose_and_point_partitions(motion):
    candidate, extended = motion
    original, camera = candidate.model_dump_json(), CAMERA.copy()
    result = validate_shadow_motion(candidate, extended, camera)
    assert result.supported
    assert tuple(item.frame_slot for item in result.frame_checks) == (5, 6)
    assert set(result.training_track_ids) <= set(candidate.motion.training_track_ids)
    assert set(result.validation_track_ids) <= set(
        candidate.motion.validation_track_ids
    )
    assert set(result.training_track_ids).isdisjoint(result.validation_track_ids)
    assert result.relative_rotation == candidate.motion.relative_rotation
    assert result.unit_translation == candidate.motion.unit_translation
    assert candidate.model_dump_json() == original
    np.testing.assert_array_equal(camera, CAMERA)
    assert validate_shadow_motion(candidate, extended, camera) == result


def test_future_validation_cannot_be_fitted_using_withheld_points(motion):
    candidate, extended = motion
    pixels = np.asarray(extended.pixels).copy()
    indices = [
        extended.track_ids.index(identity)
        for identity in candidate.motion.validation_track_ids
    ]
    pixels[5:, indices, 0] += 12
    corrupt = TemporalTracks.model_validate(
        extended.model_dump() | {"pixels": pixels.tolist()}
    )
    result = validate_shadow_motion(candidate, corrupt, CAMERA)
    assert not result.supported and len(result.frame_checks) == 2
    assert all(not item.supported for item in result.frame_checks)
    assert result.unit_translation is None


def test_extended_tracking_cannot_modify_the_past_or_rescue_lost_identities(motion):
    candidate, extended = motion
    pixels = np.asarray(extended.pixels).copy()
    pixels[1, :, 0] += 0.01
    changed = TemporalTracks.model_validate(
        extended.model_dump() | {"pixels": pixels.tolist()}
    )
    result = validate_shadow_motion(candidate, changed, CAMERA)
    assert not result.supported and "frozen prefix" in result.reason
    training = set(candidate.motion.training_track_ids)
    selected = [
        index
        for index, identity in enumerate(extended.track_ids)
        if identity in training
    ]
    values = extended.model_dump()
    for name in ("track_ids", "minimum_patch_correlations", "chain_cycle_errors"):
        values[name] = [values[name][index] for index in selected]
    values["pixels"] = [
        [frame[index] for index in selected] for frame in values["pixels"]
    ]
    rejected = validate_shadow_motion(
        candidate, TemporalTracks.model_validate(values), CAMERA
    )
    assert not rejected.supported and "partitions" in rejected.reason
    assert not rejected.validation_track_ids


def test_missing_future_images_or_candidate_are_not_a_validation_pass(motion):
    candidate, extended = motion
    values = extended.model_dump() | {
        "frame_count": 6,
        "pixels": extended.pixels[:6],
        "surviving_counts": extended.surviving_counts[:6],
    }
    for future in (None, TemporalTracks.model_validate(values)):
        result = validate_shadow_motion(candidate, future, CAMERA)
        assert not result.supported and not result.frame_checks
    assert not validate_shadow_motion(None, None, CAMERA).supported


@pytest.mark.parametrize("seed", [-1, True, 2**31 - 1])
def test_invalid_or_overflowing_seed_rejects(seed):
    with pytest.raises(ValueError, match="seed"):
        validate_shadow_motion(None, None, CAMERA, seed=seed)


@pytest.mark.parametrize(
    "camera", [np.eye(2), np.zeros((3, 3)), np.full((3, 3), np.nan)]
)
def test_invalid_intrinsics_are_not_evidence(motion, camera):
    with pytest.raises(ValueError, match="intrinsics"):
        validate_shadow_motion(*motion, camera)


def test_schema_rejects_reselection_partition_leakage_and_fake_success(motion):
    candidate, extended = motion
    accepted = validate_shadow_motion(candidate, extended, CAMERA).model_dump()
    for change in (
        {"unit_translation": (1, 0, 0)},
        {"supported": False},
        {"frame_checks": accepted["frame_checks"][:1]},
        {"validation_track_ids": accepted["training_track_ids"]},
        {"candidate": None},
    ):
        with pytest.raises(ValidationError):
            ShadowMotionEvidence.model_validate(accepted | change)
    for change in (
        {"future_frames": 1},
        {"prefix_pixel_tolerance": float("nan")},
        {"yaw": 0},
    ):
        with pytest.raises(ValidationError):
            ShadowMotionConfig(**change)
