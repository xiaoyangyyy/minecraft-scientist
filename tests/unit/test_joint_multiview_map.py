import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")

from minecraft_scientist.exploration import joint_multiview_map as joint
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks

CAMERA = np.array(((90.0, 0, 112), (0, 120.0, 112), (0, 0, 1)))


def fixture_tracks(count=150, frames=10, noise=0):
    rng = np.random.default_rng(411)
    xyz = np.column_stack(
        (
            rng.uniform(-2, 2, count),
            rng.uniform(-1.6, 1.6, count),
            rng.uniform(4, 9, count),
        )
    )
    pixels, rotations, translations = [], [], []
    for slot in range(frames):
        center = np.array(
            (0.23 * min(slot, 3), 0.02 * np.sin(slot), 0.20 * max(slot - 3, 0))
        )
        rotation = cv2.Rodrigues(np.array((0.001 * slot, 0.005 * slot, 0.001 * slot)))[
            0
        ]
        translation = -rotation @ center
        observed, _ = joint._project(xyz, rotation, translation, CAMERA)
        pixels.append(
            tuple(map(tuple, observed + rng.normal(0, noise, observed.shape)))
        )
        rotations.append(rotation)
        translations.append(translation)
    tracks = TemporalTracks(
        image_shape=(224, 224),
        frame_count=frames,
        seed_count=count,
        surviving_counts=(count,) * frames,
        track_ids=tuple(range(count)),
        pixels=tuple(pixels),
        minimum_patch_correlations=(0.95,) * count,
        chain_cycle_errors=(0.0,) * count,
    )
    return tracks, np.array(rotations), np.array(translations)


def prefix(tracks, frames):
    return TemporalTracks.model_validate(
        tracks.model_dump()
        | {
            "frame_count": frames,
            "pixels": tracks.pixels[:frames],
            "surviving_counts": tracks.surviving_counts[:frames],
        }
    )


def test_joint_training_fit_and_new_control_validation_recover_bent_trajectory():
    extended, expected_r, expected_t = fixture_tracks()
    past = prefix(extended, 7)
    fitted = joint.fit_joint_map(past, CAMERA, bootstrap_slot=3, seed=13)
    assert fitted.supported and not fitted.navigation_accepted
    assert set(fitted.training_ids).isdisjoint(fitted.validation_ids)
    assert len(fitted.poses) == 7 and [
        item.frame_slot for item in fitted.frame_checks
    ] == list(range(7))
    scale = np.linalg.norm(expected_t[3])
    for pose in fitted.poses:
        np.testing.assert_allclose(
            pose.rotation, expected_r[pose.frame_slot], atol=0.002
        )
        np.testing.assert_allclose(
            pose.translation, expected_t[pose.frame_slot] / scale, atol=0.02
        )
    future = joint.validate_joint_future(fitted, past, extended, target_slots=(8, 9))
    assert future["status"] == "consistent" and not future["navigation_accepted"]
    assert [check["frame_slot"] for check in future["frame_checks"]] == [8, 9]
    for pose in future["poses"]:
        np.testing.assert_allclose(
            pose["translation"], expected_t[pose["frame_slot"]] / scale, atol=0.02
        )


def test_analytic_training_jacobians_match_se3_and_point_finite_differences():
    points = np.array(((0.5, -0.4, 5), (1, 0.2, 6)))
    rotation = cv2.Rodrigues(np.array((0.02, 0.05, -0.01)))[0]
    translation = np.array((-0.5, 0.01, -0.1))
    observed, _ = joint._project(points, rotation, translation, CAMERA)
    pose, point, residual = joint._linearize(
        points, rotation, translation, observed, CAMERA, 1
    )
    assert not residual.any()
    epsilon = 1e-6
    for axis in range(6):
        step = np.eye(6)[axis] * epsilon
        change = cv2.Rodrigues(step[:3])[0]
        moved, _ = joint._project(
            points, change @ rotation, change @ translation + step[3:], CAMERA
        )
        np.testing.assert_allclose(
            (moved - observed) / epsilon, pose[:, :, axis], atol=1e-4
        )
    for axis in range(3):
        moved, _ = joint._project(
            points + np.eye(3)[axis] * epsilon, rotation, translation, CAMERA
        )
        np.testing.assert_allclose(
            (moved - observed) / epsilon, point[:, :, axis], atol=1e-4
        )


def test_joint_optimization_reduces_noisy_training_cost_monotonically():
    tracks, _, _ = fixture_tracks(frames=7, noise=0.15)
    fitted = joint.fit_joint_map(tracks, CAMERA, bootstrap_slot=3, seed=13)
    assert fitted.supported and len(fitted.training_cost_history) > 1
    assert fitted.training_cost_history[-1] < fitted.training_cost_history[0]
    assert np.all(np.diff(fitted.training_cost_history) < 0)
    assert np.linalg.norm(fitted.poses[3].translation) == pytest.approx(1)
    np.testing.assert_array_equal(fitted.poses[0].translation, (0, 0, 0))
    np.testing.assert_array_equal(fitted.poses[0].rotation, np.eye(3))


def test_withheld_points_never_change_optimized_training_poses():
    tracks, _, _ = fixture_tracks(frames=7)
    good = joint.fit_joint_map(tracks, CAMERA, bootstrap_slot=3, seed=13)
    pixels = np.array(tracks.pixels)
    # Corrupt only reserved validation rays in one past view. The map must fail,
    # but neither bootstrap nor optimized training points/poses may adapt.
    pixels[5, list(good.validation_ids)] += (12, -9)
    corrupted = TemporalTracks.model_validate(
        tracks.model_dump()
        | {"pixels": tuple(tuple(map(tuple, frame)) for frame in pixels)}
    )
    bad = joint.fit_joint_map(corrupted, CAMERA, bootstrap_slot=3, seed=13)
    assert not bad.supported
    assert bad.poses == good.poses and bad.training_points == good.training_points
    assert bad.training_cost_history == good.training_cost_history
    assert bad.reserved_training_ids == good.reserved_training_ids
    assert bad.validation_ids == good.validation_ids


def test_future_validation_does_not_call_bundle_or_refit_past(monkeypatch):
    extended, _, _ = fixture_tracks()
    past = prefix(extended, 7)
    fitted = joint.fit_joint_map(past, CAMERA, bootstrap_slot=3, seed=13)
    before = fitted.model_dump_json()

    def forbidden(*args, **kwargs):
        raise AssertionError("future pixels cannot rerun past map fitting")

    monkeypatch.setattr(joint, "_bundle", forbidden)
    assert (
        joint.validate_joint_future(fitted, past, extended, target_slots=(8, 9))[
            "status"
        ]
        == "consistent"
    )
    assert fitted.model_dump_json() == before


def test_corrupt_future_withheld_rays_are_rejected_without_training_pose_adaptation():
    extended, _, _ = fixture_tracks()
    past = prefix(extended, 7)
    fitted = joint.fit_joint_map(past, CAMERA, bootstrap_slot=3, seed=13)
    good = joint.validate_joint_future(fitted, past, extended, target_slots=(8, 9))
    pixels = np.array(extended.pixels)
    pixels[8:, list(fitted.validation_ids)] += (12, -9)
    corrupted = TemporalTracks.model_validate(
        extended.model_dump()
        | {"pixels": tuple(tuple(map(tuple, frame)) for frame in pixels)}
    )
    bad = joint.validate_joint_future(fitted, past, corrupted, target_slots=(8, 9))
    assert bad["status"] == "inconsistent" and bad["poses"] == good["poses"]


def test_missing_original_landmark_cannot_be_replaced_by_other_survivors():
    extended, _, _ = fixture_tracks()
    past = prefix(extended, 7)
    fitted = joint.fit_joint_map(past, CAMERA, bootstrap_slot=3, seed=13)
    from minecraft_scientist.exploration.second_leg_shadow import retain_tracks

    reduced = retain_tracks(
        extended, set(extended.track_ids) - set(fitted.validation_ids[7:])
    )
    assert len(reduced.track_ids) > 100
    assert (
        joint.validate_joint_future(fitted, past, reduced, target_slots=(8, 9))[
            "status"
        ]
        == "unavailable"
    )


def test_surviving_original_partitions_are_filtered_without_replenishment_or_refit():
    from minecraft_scientist.exploration.second_leg_shadow import retain_tracks

    extended, _, _ = fixture_tracks()
    past = prefix(extended, 7)
    fitted = joint.fit_joint_map(past, CAMERA, bootstrap_slot=3, seed=13)
    removed = {fitted.training_ids[0], fitted.validation_ids[0]}
    reduced = retain_tracks(extended, set(extended.track_ids) - removed)
    before = fitted.model_dump_json()
    result = joint.validate_joint_future(fitted, past, reduced, target_slots=(8, 9))
    assert result["status"] == "consistent"
    assert set(result["training_ids"]) == set(fitted.training_ids) - removed
    assert set(result["validation_ids"]) == set(fitted.validation_ids) - removed
    assert fitted.model_dump_json() == before


@pytest.mark.parametrize("targets", [(1,), (6,), (9, 8), (8, 8), (True,), (10,), ()])
def test_invalid_future_targets_reject(targets):
    extended, _, _ = fixture_tracks()
    past = prefix(extended, 7)
    fitted = joint.fit_joint_map(past, CAMERA, bootstrap_slot=3, seed=13)
    with pytest.raises(ValueError, match="targets"):
        joint.validate_joint_future(fitted, past, extended, target_slots=targets)


def test_future_prefix_cannot_revise_frozen_pixels():
    extended, _, _ = fixture_tracks()
    past = prefix(extended, 7)
    fitted = joint.fit_joint_map(past, CAMERA, bootstrap_slot=3, seed=13)
    pixels = np.array(extended.pixels)
    pixels[2, 0, 0] += 0.01
    forged = TemporalTracks.model_validate(
        extended.model_dump()
        | {"pixels": tuple(tuple(map(tuple, frame)) for frame in pixels)}
    )
    with pytest.raises(ValueError, match="revised"):
        joint.validate_joint_future(fitted, past, forged, target_slots=(8, 9))


def test_joint_fit_is_deterministic_and_leaves_inputs_unchanged():
    tracks, _, _ = fixture_tracks(frames=7, noise=0.15)
    original, original_camera = tracks.model_dump_json(), CAMERA.copy()
    first = joint.fit_joint_map(tracks, CAMERA, bootstrap_slot=3, seed=13)
    assert first == joint.fit_joint_map(tracks, CAMERA, bootstrap_slot=3, seed=13)
    assert tracks.model_dump_json() == original
    np.testing.assert_array_equal(CAMERA, original_camera)


def test_storage_order_cannot_change_id_partitions_or_geometry():
    tracks, _, _ = fixture_tracks(frames=7, noise=0.15)
    permutation = np.random.default_rng(431).permutation(len(tracks.track_ids))
    values = tracks.model_dump()
    for name in ("track_ids", "minimum_patch_correlations", "chain_cycle_errors"):
        values[name] = tuple(values[name][index] for index in permutation)
    values["pixels"] = tuple(
        tuple(frame[index] for index in permutation) for frame in tracks.pixels
    )
    reordered = TemporalTracks.model_validate(values)
    original = joint.fit_joint_map(tracks, CAMERA, bootstrap_slot=3, seed=13)
    changed = joint.fit_joint_map(reordered, CAMERA, bootstrap_slot=3, seed=13)
    assert original.reserved_training_ids == changed.reserved_training_ids
    assert original.validation_ids == changed.validation_ids
    assert (
        original.poses == changed.poses
        and original.training_points == changed.training_points
    )


@pytest.mark.parametrize(
    "bootstrap,seed", [(True, 1), (7, 1), (1, 1), (3, True), (3, -1)]
)
def test_invalid_bootstrap_and_seed_reject_before_opencv(monkeypatch, bootstrap, seed):
    tracks, _, _ = fixture_tracks(frames=7)

    def forbidden():
        raise AssertionError("invalid policy must not enter geometric fitting")

    monkeypatch.setattr(joint, "_opencv", forbidden)
    with pytest.raises(ValueError):
        joint.fit_joint_map(tracks, CAMERA, bootstrap_slot=bootstrap, seed=seed)


@pytest.mark.parametrize(
    "changed",
    [
        {"track_budget": 31},
        {"maximum_frames": 17},
        {"iterations": True},
        {"initial_damping": float("nan")},
        {"yaw": 1},
    ],
)
def test_joint_config_rejects_invalid_values(changed):
    with pytest.raises(ValidationError):
        joint.JointMapConfig(**changed)


def test_rotation_and_insufficient_tracks_do_not_yield_a_map():
    tracks, _, _ = fixture_tracks(count=24, frames=7)
    assert not joint.fit_joint_map(tracks, CAMERA, bootstrap_slot=3).supported
    tracks, _, _ = fixture_tracks(frames=7)
    stationary = TemporalTracks.model_validate(
        tracks.model_dump() | {"pixels": (tracks.pixels[0],) * tracks.frame_count}
    )
    rejected = joint.fit_joint_map(stationary, CAMERA, bootstrap_slot=3)
    assert not rejected.supported and not rejected.poses
