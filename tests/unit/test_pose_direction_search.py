import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")
from test_multiview_motion import CAMERA, _tracks

from minecraft_scientist.exploration.keyframe_motion import KeyframeAttempt
from minecraft_scientist.exploration.multiview_motion import estimate_multiview_motion
from minecraft_scientist.exploration.pose_direction_search import (
    DirectionSearchConfig,
    DirectionSearchEvidence,
    _fit_rotation,
    _map_validation,
    _signed_sampson,
    inspect_direction_search,
)
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks


@pytest.fixture
def candidate():
    tracks = _tracks()
    motion = estimate_multiview_motion(tracks, CAMERA, seed=7)
    assert motion.supported
    return KeyframeAttempt(end_offset=4, tracks=tracks, motion=motion)


def test_frozen_reference_map_replays_exact_original_pnp(candidate):
    before = candidate.model_dump_json()
    result = inspect_direction_search(candidate, CAMERA, seed=7)
    assert result.reference.supported and result.reference_matches_original_checks
    assert len(result.trials) == 16 and not result.ambiguity_detected
    assert result.training_track_ids == candidate.motion.training_track_ids
    assert result.validation_track_ids == candidate.motion.validation_track_ids
    assert candidate.model_dump_json() == before
    assert result == inspect_direction_search(candidate, CAMERA, seed=7)
    for trial in result.trials:
        assert trial.training_objective_final <= trial.training_objective_initial
        assert trial.direction_separation_degrees == pytest.approx(
            trial.requested_direction_offset_degrees, abs=1e-8
        )
        assert trial.rotation_adjustment_degrees <= 3 + 1e-8


def test_corrupt_reference_is_uninformative_and_cannot_start_search(
    candidate, monkeypatch
):
    import minecraft_scientist.exploration.pose_direction_search as search

    pixels = np.asarray(candidate.tracks.pixels).copy()
    validation = [
        candidate.tracks.track_ids.index(item)
        for item in candidate.motion.validation_track_ids
    ]
    pixels[1:-1, validation, 0] += 12
    corrupt = candidate.model_copy(
        update={
            "tracks": TemporalTracks.model_validate(
                candidate.tracks.model_dump() | {"pixels": pixels.tolist()}
            )
        }
    )
    monkeypatch.setattr(
        search,
        "_fit_rotation",
        lambda *args: pytest.fail("failed reference started optimization"),
    )
    result = inspect_direction_search(corrupt, CAMERA, seed=7)
    assert not result.reference.supported
    assert not result.reference_matches_original_checks
    assert not result.trials and not result.ambiguity_detected


def test_training_rotation_fit_reduces_residual_without_changing_fixed_center(
    candidate,
):
    first, second = [
        np.column_stack((np.asarray(frame), np.ones(len(frame))))
        for frame in (candidate.tracks.pixels[0], candidate.tracks.pixels[-1])
    ]
    rotation = cv2.Rodrigues(np.array((0.01, 0.04, -0.02)))[0]
    shift = np.array((-0.7, 0.05, -0.15))
    center = -rotation.T @ (shift / np.linalg.norm(shift))
    perturbed = cv2.Rodrigues(np.array((0.005, -0.004, 0.003)))[0] @ rotation
    changed, initial, final, steps, adjustment = _fit_rotation(
        first, second, CAMERA, perturbed, center, DirectionSearchConfig(), 0.5, cv2
    )
    assert steps > 0 and final < initial * 1e-6 and adjustment < 3
    assert (
        np.max(np.abs(_signed_sampson(first, second, CAMERA, changed, center))) < 1e-4
    )
    np.testing.assert_allclose(-changed.T @ (-changed @ center), center, atol=1e-12)


def test_known_planar_pose_has_direction_constrained_witness(candidate):
    # Known-pose sanity fixture, not a claim about what the baseline would learn.
    rng = np.random.default_rng(123)
    xyz = np.column_stack(
        (rng.uniform(-2, 2, 200), rng.uniform(-2, 2, 200), np.full(200, 16.0))
    )
    pixels = []
    for alpha in np.linspace(0, 1, 5):
        rotation = cv2.Rodrigues(np.array((0.01, 0.04, -0.02)) * alpha)[0]
        projected = (
            xyz @ rotation.T + np.array((-0.7, 0.05, -0.15)) * alpha
        ) @ CAMERA.T
        pixels.append((projected[:, :2] / projected[:, 2:]).tolist())
    tracks = TemporalTracks.model_validate(
        candidate.tracks.model_dump() | {"pixels": pixels}
    )
    declared = candidate.model_copy(update={"tracks": tracks})
    original_rotation = np.asarray(candidate.motion.candidate.relative_rotation)
    original_translation = np.asarray(candidate.motion.candidate.unit_translation)
    reference = _map_validation(
        declared,
        CAMERA,
        original_rotation,
        original_translation,
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
    result = inspect_direction_search(declared, CAMERA, seed=7)
    assert result.reference_matches_original_checks and result.ambiguity_detected
    assert all(
        trial.validation.supported and trial.direction_separation_degrees > 20
        for trial in result.trials
        if trial.plausible
    )


def test_schema_cannot_drop_grid_or_forge_reference_and_witness(candidate):
    result = inspect_direction_search(candidate, CAMERA, seed=7).model_dump()
    for change in (
        {"trials": result["trials"][:-1]},
        {"ambiguity_detected": True},
        {"reference_matches_original_checks": False},
        {"validation_track_ids": result["training_track_ids"]},
    ):
        with pytest.raises(ValidationError):
            DirectionSearchEvidence.model_validate(result | change)


@pytest.mark.parametrize("seed", [-1, True, 2**31 - 1])
def test_search_seed_bounds(seed):
    with pytest.raises(ValueError, match="seed"):
        inspect_direction_search(None, CAMERA, seed=seed)


@pytest.mark.parametrize(
    "camera", [np.eye(2), np.zeros((3, 3)), np.full((3, 3), np.nan)]
)
def test_search_intrinsic_bounds(camera):
    with pytest.raises(ValueError, match="intrinsics"):
        inspect_direction_search(None, camera)
