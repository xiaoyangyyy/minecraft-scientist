import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")

from minecraft_scientist.exploration.camera_intrinsics import CameraResizeSpec
from minecraft_scientist.exploration.multiview_motion import (
    MultiViewMotionConfig,
    MultiViewMotionEvidence,
    estimate_multiview_motion,
    estimate_temporal_endpoint,
)
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks

CAMERA = np.array(((90.0, 0, 112), (0, 90.0, 112), (0, 0, 1)))


def _tracks(frame_count=5, translated=True, camera=CAMERA):
    rng = np.random.default_rng(123)
    xyz = np.column_stack(
        (rng.uniform(-2, 2, 200), rng.uniform(-2, 2, 200), rng.uniform(4, 9, 200))
    )
    pixels = []
    for index in range(frame_count):
        alpha = index / (frame_count - 1)
        rotation = cv2.Rodrigues(np.array((0.01, 0.04, -0.02)) * alpha)[0]
        translation = (
            np.array((-0.7, 0.05, -0.15)) * alpha if translated else np.zeros(3)
        )
        projected = (xyz @ rotation.T + translation) @ camera.T
        pixels.append(tuple(map(tuple, projected[:, :2] / projected[:, 2:])))
    return TemporalTracks(
        image_shape=(224, 224),
        frame_count=frame_count,
        seed_count=len(xyz),
        surviving_counts=(len(xyz),) * frame_count,
        track_ids=tuple(range(len(xyz))),
        pixels=tuple(pixels),
        minimum_patch_correlations=(0.95,) * len(xyz),
        chain_cycle_errors=(0.0,) * len(xyz),
    )


def test_multiview_geometry_uses_distinct_training_and_validation_identities():
    tracks = _tracks()
    result = estimate_multiview_motion(tracks, CAMERA)
    assert result.supported and len(result.frame_checks) == 3
    assert set(result.training_track_ids).isdisjoint(result.validation_track_ids)
    assert len(result.validation_track_ids) >= 8
    assert all(
        check.supported and check.validation_acceptance >= 0.8
        for check in result.frame_checks
    )
    assert result.sensitivity.supported
    direction = np.array((-0.7, 0.05, -0.15))
    direction /= np.linalg.norm(direction)
    assert np.array(result.unit_translation) @ direction > np.cos(np.radians(20))


def test_corrupted_intermediate_points_cannot_pass_endpoint_only_fit():
    tracks = _tracks()
    frames = list(tracks.pixels)
    rng = np.random.default_rng(124)
    frames[2] = tuple(map(tuple, rng.uniform(65, 160, (len(tracks.track_ids), 2))))
    corrupt = TemporalTracks.model_validate(
        tracks.model_dump() | {"pixels": tuple(frames)}
    )
    result = estimate_multiview_motion(corrupt, CAMERA)
    assert result.candidate.supported and not result.supported
    assert result.reason == "intermediate geometry validation failed"
    assert result.unit_translation is None and result.relative_rotation is None


def test_no_intermediate_frame_and_rotation_controls_have_no_motion_pose():
    endpoint_only = estimate_multiview_motion(_tracks(frame_count=2), CAMERA)
    assert not endpoint_only.supported and endpoint_only.reason.startswith(
        "no intermediate"
    )
    rotation = estimate_multiview_motion(_tracks(translated=False), CAMERA)
    assert not rotation.supported and rotation.unit_translation is None


def test_insufficient_common_track_budget_is_a_rejection():
    tracks = _tracks()
    values = tracks.model_dump()
    for key in ("track_ids", "minimum_patch_correlations", "chain_cycle_errors"):
        values[key] = values[key][:24]
    values["pixels"] = tuple(frame[:24] for frame in values["pixels"])
    result = estimate_multiview_motion(TemporalTracks.model_validate(values), CAMERA)
    assert not result.supported and result.unit_translation is None


def test_multiview_prediction_is_deterministic():
    tracks = _tracks()
    assert estimate_multiview_motion(
        tracks, CAMERA, seed=7
    ) == estimate_multiview_motion(tracks, CAMERA, seed=7)


def test_stretched_camera_still_recovers_supported_nonplanar_motion():
    camera = CameraResizeSpec(
        render_size=(640, 360), observation_size=(224, 224)
    ).camera(90)
    original = camera.copy()
    result = estimate_multiview_motion(_tracks(camera=camera), camera)
    assert result.supported and all(check.supported for check in result.frame_checks)
    direction = np.array((-0.7, 0.05, -0.15))
    direction /= np.linalg.norm(direction)
    assert np.array(result.unit_translation) @ direction > np.cos(np.radians(20))
    np.testing.assert_array_equal(camera, original)


def test_refit_rechecks_support_and_cannot_turn_rotation_into_translation():
    matches = _tracks(translated=False).endpoint_correspondences()
    result = estimate_temporal_endpoint(
        matches, CAMERA, config=MultiViewMotionConfig().two_view
    )
    assert not result.supported and result.unit_translation is None


def test_multiview_schema_rejects_invalid_budgets_and_execution_on_rejection():
    for changed in (
        {"minimum_tracks": 24, "minimum_training_inliers": 20},
        {"yaw": 1},
        {"reference_validation_pixels": float("nan")},
    ):
        with pytest.raises(ValidationError):
            MultiViewMotionConfig(**changed)
    good = estimate_multiview_motion(_tracks(), CAMERA).model_dump()
    with pytest.raises(ValidationError):
        MultiViewMotionEvidence.model_validate(
            good | {"validation_track_ids": good["training_track_ids"]}
        )
    with pytest.raises(ValidationError):
        MultiViewMotionEvidence.model_validate(good | {"supported": False})
