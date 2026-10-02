import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")

from minecraft_scientist.exploration.camera_intrinsics import CameraResizeSpec
from minecraft_scientist.exploration.consensus_motion import estimate_consensus_endpoint
from minecraft_scientist.exploration.dense_correspondence import DenseCorrespondences
from minecraft_scientist.exploration.multiview_motion import (
    MultiViewMotionConfig,
    estimate_multiview_motion,
)
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks
from minecraft_scientist.exploration.two_view_motion import estimate_two_view_motion

CAMERA = CameraResizeSpec(render_size=(640, 360), observation_size=(224, 224)).camera(
    90
)
CONFIG = MultiViewMotionConfig()


def _tracks(rotation_only=False, stationary=False, planar=False):
    rng = np.random.default_rng(219)
    xyz = np.column_stack(
        (
            rng.uniform(-2, 2, 250),
            rng.uniform(-1.5, 1.5, 250),
            np.full(250, 7.0) if planar else rng.uniform(4, 10, 250),
        )
    )
    pixels = []
    for index in range(5):
        alpha = index / 4
        rotation = (
            cv2.Rodrigues(np.array((0.01, 0.02, -0.01)) * alpha)[0]
            if not stationary
            else np.eye(3)
        )
        translation = (
            np.zeros(3)
            if rotation_only or stationary
            else np.array((-0.5, 0.03, -0.1)) * alpha
        )
        projected = (xyz @ rotation.T + translation) @ CAMERA.T
        pixels.append(tuple(map(tuple, projected[:, :2] / projected[:, 2:])))
    return TemporalTracks(
        image_shape=(224, 224),
        frame_count=5,
        seed_count=250,
        surviving_counts=(250,) * 5,
        track_ids=tuple(range(250)),
        pixels=tuple(pixels),
        minimum_patch_correlations=(0.95,) * 250,
        chain_cycle_errors=(0.0,) * 250,
    )


def test_conditioned_consensus_recovers_nonplanar_direction_before_pose_gate():
    matches = _tracks().endpoint_correspondences()
    result = estimate_consensus_endpoint(matches, CAMERA, config=CONFIG.two_view)
    assert result.supported
    direction = np.array((-0.5, 0.03, -0.1))
    direction /= np.linalg.norm(direction)
    assert np.array(result.unit_translation) @ direction > 0.99
    assert result == estimate_consensus_endpoint(
        matches, CAMERA, config=CONFIG.two_view
    )


def test_unreliable_minimal_pose_is_not_required_before_consensus_refitting(
    monkeypatch,
):
    matches = _tracks().endpoint_correspondences()
    cv2.setRNGSeed(0)
    initial, _ = cv2.findEssentialMat(
        np.asarray(matches.points0),
        np.asarray(matches.points1),
        CAMERA,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=CONFIG.two_view.reference_ransac_pixels,
    )
    original = cv2.recoverPose

    def unreliable_minimal(matrix, first, second, camera, **kwargs):
        if np.array_equal(matrix, initial):
            return (
                0,
                np.eye(3),
                np.array(((1.0,), (0,), (0,))),
                np.zeros((len(first), 1), dtype=np.uint8),
            )
        return original(matrix, first, second, camera, **kwargs)

    monkeypatch.setattr(cv2, "recoverPose", unreliable_minimal)
    baseline = estimate_two_view_motion(matches, CAMERA, config=CONFIG.two_view)
    improved = estimate_consensus_endpoint(matches, CAMERA, config=CONFIG.two_view)
    assert not baseline.supported and improved.supported


@pytest.mark.parametrize(
    "change", [{"rotation_only": True}, {"stationary": True}, {"planar": True}]
)
def test_rotation_stationary_and_degenerate_planar_pairs_are_not_rescued(change):
    result = estimate_consensus_endpoint(
        _tracks(**change).endpoint_correspondences(), CAMERA, config=CONFIG.two_view
    )
    assert not result.supported and result.unit_translation is None


def test_consensus_is_used_in_every_sensitivity_trial_and_keeps_intermediate_holdouts():
    result = estimate_multiview_motion(
        _tracks(), CAMERA, config=CONFIG, endpoint_estimator=estimate_consensus_endpoint
    )
    assert result.supported and result.sensitivity.supported
    assert len(result.sensitivity.trials) == 16
    assert set(result.training_track_ids).isdisjoint(result.validation_track_ids)
    assert all(check.supported for check in result.frame_checks)


def test_unrelated_points_cannot_pass_consensus_and_inputs_are_not_mutated():
    original = _tracks().endpoint_correspondences()
    values = original.model_dump()
    values["points1"] = tuple(
        map(tuple, np.random.default_rng(19).uniform(50, 170, (250, 2)))
    )
    matches = DenseCorrespondences.model_validate(values)
    camera = CAMERA.copy()
    result = estimate_consensus_endpoint(matches, camera, config=CONFIG.two_view)
    assert not result.supported and result.relative_rotation is None
    np.testing.assert_array_equal(camera, CAMERA)
    for invalid in (-1, True, 2**31):
        with pytest.raises(ValueError, match="seed"):
            estimate_consensus_endpoint(
                original, CAMERA, config=CONFIG.two_view, seed=invalid
            )
