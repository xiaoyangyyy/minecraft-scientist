import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")
from test_multiview_motion import CAMERA, _tracks

from minecraft_scientist.exploration.keyframe_motion import KeyframeAttempt
from minecraft_scientist.exploration.multiview_motion import estimate_multiview_motion
from minecraft_scientist.exploration.pose_competition import (
    PoseCompetitionConfig,
    PoseCompetitionEvidence,
    _translation,
    inspect_pose_competition,
)
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks


@pytest.fixture
def candidate():
    tracks = _tracks()
    result = KeyframeAttempt(
        end_offset=4, tracks=tracks, motion=estimate_multiview_motion(tracks, CAMERA)
    )
    assert result.motion.supported
    return result


def test_fixed_rotation_translation_fit_uses_epipolar_geometry(candidate):
    pixels = np.asarray(candidate.tracks.pixels)
    endpoints = [
        np.column_stack((frame, np.ones(len(frame))))
        for frame in (pixels[0], pixels[-1])
    ]
    rotation = cv2.Rodrigues(np.array((0.01, 0.04, -0.02)))[0]
    result = _translation(*endpoints, CAMERA, rotation, 0.5)
    expected = np.array((-0.7, 0.05, -0.15))
    expected /= np.linalg.norm(expected)
    assert abs(result @ expected) == pytest.approx(1.0, abs=1e-9)


def test_competition_retains_complete_grid_and_original_identities(candidate):
    frozen, camera = candidate.model_dump_json(), CAMERA.copy()
    result = inspect_pose_competition(candidate, camera)
    assert len(result.trials) == 19
    assert result.training_track_ids == candidate.motion.training_track_ids
    assert result.validation_track_ids == candidate.motion.validation_track_ids
    assert set(result.training_track_ids).isdisjoint(result.validation_track_ids)
    assert result.trials[0].plausible
    assert not result.ambiguity_detected
    assert "uniqueness unproved" in result.reason
    assert (
        not {"supported", "unit_translation", "relative_rotation"}
        & PoseCompetitionEvidence.model_fields.keys()
    )
    assert candidate.model_dump_json() == frozen
    np.testing.assert_array_equal(camera, CAMERA)
    assert inspect_pose_competition(candidate, camera) == result


def test_withheld_intermediate_pixels_cannot_fit_an_alternative(candidate):
    pixels = np.asarray(candidate.tracks.pixels).copy()
    slots = [
        candidate.tracks.track_ids.index(identity)
        for identity in candidate.motion.validation_track_ids
    ]
    pixels[1:-1, slots, 0] += 12
    corrupt = candidate.model_copy(
        update={
            "tracks": TemporalTracks.model_validate(
                candidate.tracks.model_dump() | {"pixels": pixels.tolist()}
            )
        }
    )
    result = inspect_pose_competition(corrupt, CAMERA)
    assert not any(trial.plausible for trial in result.trials)
    assert result.trials[0].endpoint_partitions_valid
    assert all(not item.supported for item in result.trials[0].frame_checks)


def test_distant_planar_scene_supplies_a_real_competing_map_witness(candidate):
    # Known-pose synthetic fixture, not a claim that the baseline selects this map.
    # Both the original known pose and alternative explain the withheld pixels.
    rng = np.random.default_rng(123)
    xyz = np.column_stack(
        (rng.uniform(-2, 2, 200), rng.uniform(-2, 2, 200), np.full(200, 16.0))
    )
    frames = []
    for alpha in np.linspace(0, 1, 5):
        rotation = cv2.Rodrigues(np.array((0.01, 0.04, -0.02)) * alpha)[0]
        projected = (
            xyz @ rotation.T + np.array((-0.7, 0.05, -0.15)) * alpha
        ) @ CAMERA.T
        frames.append((projected[:, :2] / projected[:, 2:]).tolist())
    tracks = TemporalTracks.model_validate(
        candidate.tracks.model_dump() | {"pixels": frames}
    )
    declared = candidate.model_copy(update={"tracks": tracks})
    result = inspect_pose_competition(declared, CAMERA)
    assert result.ambiguity_detected and result.trials[0].plausible
    witnesses = [
        trial
        for trial in result.trials
        if trial.plausible and trial.direction_separation_degrees >= 20
    ]
    assert witnesses and all(trial.endpoint_partitions_valid for trial in witnesses)
    assert all(
        all(check.supported for check in trial.frame_checks) for trial in witnesses
    )


@pytest.mark.parametrize("seed", [-1, True, 2**31 - 1])
def test_invalid_or_overflowing_seed_rejects(seed):
    with pytest.raises(ValueError, match="seed"):
        inspect_pose_competition(None, CAMERA, seed=seed)


@pytest.mark.parametrize(
    "camera", [np.eye(2), np.zeros((3, 3)), np.full((3, 3), np.nan)]
)
def test_invalid_camera_rejects_without_a_candidate(camera):
    with pytest.raises(ValueError, match="intrinsics"):
        inspect_pose_competition(None, camera)


def test_schema_rejects_missing_trials_and_forged_ambiguity(candidate):
    result = inspect_pose_competition(candidate, CAMERA).model_dump()
    for change in (
        {"ambiguity_detected": True},
        {"trials": result["trials"][:-1]},
        {"validation_track_ids": result["training_track_ids"]},
    ):
        with pytest.raises(ValidationError):
            PoseCompetitionEvidence.model_validate(result | change)
    for offsets in ((1, 0.5), (0,), (1, 1), (4,), (np.nan,)):
        with pytest.raises(ValidationError):
            PoseCompetitionConfig(rotation_offsets_degrees=offsets)


def test_no_candidate_is_not_evidence_of_uniqueness():
    result = inspect_pose_competition(None, CAMERA)
    assert not result.ambiguity_detected and not result.trials
    assert result.reason == "no supported frozen local candidate"
