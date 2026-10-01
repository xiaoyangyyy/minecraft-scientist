import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")

from minecraft_scientist.exploration import visual_odometry as odometry
from minecraft_scientist.exploration.visual_geometry import (
    VisualGeometryConfig,
    _features,
    _frame,
    _matched_indices,
    estimate_visual_goal_bearing,
)
from minecraft_scientist.exploration.visual_odometry import (
    RGBVisualOdometry,
    VisualOdometryConfig,
    VisualPoseEvidence,
    triangulate_landmarks,
)

CAMERA = np.array(((90.0, 0, 112), (0, 90.0, 112), (0, 0, 1)))


def _project(points, center):
    values = (points - center) @ CAMERA.T
    return values[:, :2] / values[:, 2:]


def test_preprocessing_cannot_read_hud_outside_world_crop():
    rng = np.random.default_rng(7)
    original = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    altered = rng.integers(0, 256, original.shape, dtype=np.uint8)
    altered[44:170, 40:183] = original[44:170, 40:183]
    config = VisualGeometryConfig()
    first, second = _features(original, config, cv2), _features(altered, config, cv2)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(
        odometry._tracking_gray(original, cv2), odometry._tracking_gray(altered, cv2)
    )


def test_match_identity_preserves_different_orientations_at_same_pixel():
    pixels = np.array(((80, 80), (80, 80), (100, 100)), dtype=float)
    descriptors = np.random.default_rng(1).normal(size=(3, 128)).astype(np.float32)
    permutation = np.array((1, 2, 0))
    left, right = _matched_indices(
        (pixels, descriptors),
        (pixels[permutation], descriptors[permutation]),
        VisualGeometryConfig(),
        cv2,
    )
    np.testing.assert_array_equal(left, (0, 1, 2))
    np.testing.assert_array_equal(right, (2, 0, 1))


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((3, 3)),
        np.eye(2),
        np.array(((-1, 0, 112), (0, 90, 112), (0, 0, 1))),
        np.array(((90, 0, 112), (0, 90, 112), (0, 1, 1))),
        np.full((3, 3), np.nan),
    ],
)
def test_invalid_camera_is_rejected(matrix):
    with pytest.raises(ValueError):
        RGBVisualOdometry(matrix)


def test_geometry_inputs_reject_nonfinite_pixels_and_privileged_configuration():
    with pytest.raises(ValueError):
        _frame(np.full((64, 64, 3), np.nan))
    with pytest.raises(ValueError):
        _frame(np.zeros((10, 10, 3)))
    with pytest.raises(ValidationError):
        VisualOdometryConfig(xpos=3)
    with pytest.raises(ValueError):
        estimate_visual_goal_bearing([np.zeros((64, 64, 3))], [])


def test_triangulation_recovers_supported_points_and_rejects_low_parallax():
    points = np.array(((-1, -1, 4), (1, 1, 6), (0, 0, 10000)), dtype=float)
    center = np.array((0.5, 0, 0))
    recovered, valid = triangulate_landmarks(
        _project(points, np.zeros(3)),
        _project(points, center),
        np.eye(3),
        np.zeros(3),
        np.eye(3),
        -center,
        CAMERA,
        config=VisualOdometryConfig(),
        cv2=cv2,
    )
    np.testing.assert_allclose(recovered[:2], points[:2], atol=1e-8)
    np.testing.assert_array_equal(valid, (True, True, False))


def test_triangulation_rejects_behind_camera_and_inconsistent_projection():
    points = np.array(((-1, -1, -4), (1, 1, 6)), dtype=float)
    center = np.array((0.5, 0, 0))
    second = _project(points, center)
    second[1, 1] += 25
    _, valid = triangulate_landmarks(
        _project(points, np.zeros(3)),
        second,
        np.eye(3),
        np.zeros(3),
        np.eye(3),
        -center,
        CAMERA,
        config=VisualOdometryConfig(),
        cv2=cv2,
    )
    assert not valid.any()


def test_blank_frames_never_fabricate_supported_camera_positions():
    tracker = RGBVisualOdometry(CAMERA)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    for moving in (False, False, True, True, False, True):
        pose = tracker.process(frame, motion_allowed=moving)
        assert not pose.tracked and not pose.initialized
        assert pose.camera_center is None and pose.map_points == 0
    with pytest.raises(ValueError, match="dimensions changed"):
        tracker.process(np.zeros((64, 64, 3)), motion_allowed=True)


def test_anchor_accumulates_baseline_and_lost_frames_do_not_mutate_map(monkeypatch):
    rng = np.random.default_rng(91)
    points = np.column_stack(
        (
            rng.uniform(-1.8, 1.8, 100),
            rng.uniform(-1.5, 1.5, 100),
            rng.uniform(4, 8, 100),
        )
    )
    descriptors = rng.normal(size=(100, 128)).astype(np.float32)
    state = {"index": 0, "lost": False}
    centers = [np.array((index * 0.1, 0, 0)) for index in range(8)]
    projections = [_project(points, center) for center in centers]
    monkeypatch.setattr(
        odometry, "_features", lambda *_: (projections[state["index"]], descriptors)
    )
    tracker = RGBVisualOdometry(CAMERA)

    def flow(_first, _second, pixels):
        if state["lost"]:
            return pixels.copy(), np.zeros(len(pixels), dtype=bool)
        # Pixel identities are carried forward, not recovered by nearest spatial
        # lookup in production. This fixture supplies perfect measured tracks.
        previous = projections[state["index"] - 1]
        identities = np.argmin(
            np.linalg.norm(pixels[:, None] - previous[None], axis=2), axis=1
        )
        return projections[state["index"]][identities], np.ones(len(pixels), dtype=bool)

    monkeypatch.setattr(tracker, "_flow", flow)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    results = []
    for index in range(6):
        state["index"] = index
        results.append(tracker.process(frame, motion_allowed=index > 0))
    assert all(not result.tracked for result in results[:4])
    assert results[4].status == "initialized"
    assert results[4].origin_frame_index == 0
    assert results[5].status == "tracked"
    np.testing.assert_allclose(results[5].camera_center, (1.25, 0, 0), atol=1e-4)
    saved_points, saved_rotation = tracker.points.copy(), tracker.rotation.copy()
    saved_translation = tracker.translation.copy()
    state.update(index=6, lost=True)
    monkeypatch.setattr(
        tracker, "_map_matches", lambda _: (np.empty((0, 3)), np.empty((0, 2)))
    )
    failed = tracker.process(frame, motion_allowed=True)
    assert not failed.tracked and failed.camera_center is None
    np.testing.assert_array_equal(tracker.points, saved_points)
    np.testing.assert_array_equal(tracker.rotation, saved_rotation)
    np.testing.assert_array_equal(tracker.translation, saved_translation)


def test_optical_flow_recovers_pixel_translation_without_hud_influence():
    rng = np.random.default_rng(42)
    first = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    second = cv2.warpAffine(
        first, np.array(((1, 0, 2), (0, 1, 1)), dtype=float), (224, 224)
    )
    tracker = RGBVisualOdometry(CAMERA)
    pixels = np.array(((80, 80), (110, 110), (140, 140)), dtype=float)
    target, valid = tracker._flow(
        odometry._tracking_gray(first, cv2),
        odometry._tracking_gray(second, cv2),
        pixels,
    )
    assert valid.all()
    np.testing.assert_allclose(target - pixels, np.tile((2, 1), (3, 1)), atol=0.25)


def test_homography_seed_is_valid_for_both_forward_and_reverse_flow():
    rng = np.random.default_rng(42)
    first = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    homography = np.array(((1, 0, 8), (0, 1, 4), (0, 0, 1)), dtype=float)
    second = cv2.warpPerspective(first, homography, (224, 224))
    tracker = RGBVisualOdometry(CAMERA)
    tracker.flow_homography = homography
    pixels = np.array(((80, 80), (110, 110), (140, 140)), dtype=float)
    target, valid = tracker._flow(
        odometry._tracking_gray(first, cv2),
        odometry._tracking_gray(second, cv2),
        pixels,
    )
    assert valid.all()
    np.testing.assert_allclose(target - pixels, np.tile((8, 4), (3, 1)), atol=0.25)


def test_long_tracking_gap_cannot_recover_pose_from_repeated_descriptors(monkeypatch):
    tracker = RGBVisualOdometry(CAMERA)
    tracker.initialized = True
    tracker.previous_features = (np.empty((0, 2)), None)
    tracker.last_tracked_frame = 0
    tracker.frame_index = tracker.config.maximum_tracking_gap_frames
    monkeypatch.setattr(
        tracker,
        "_map_matches",
        lambda _: pytest.fail("unvalidated descriptor relocalization"),
    )
    result = tracker.process(
        np.zeros((224, 224, 3), dtype=np.uint8), motion_allowed=True
    )
    assert result.status == "lost_temporal_continuity"
    assert not result.tracked and result.camera_center is None


def test_identical_motion_frames_do_not_prove_translation():
    frame = np.random.default_rng(11).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    tracker = RGBVisualOdometry(CAMERA)
    tracker.process(frame, motion_allowed=False)
    for _ in range(8):
        result = tracker.process(frame, motion_allowed=True)
        assert not result.initialized and not result.tracked
        assert result.camera_center is None


@pytest.mark.parametrize(
    "invalid",
    [
        {"tracked": False, "camera_center": (0, 0, 0)},
        {"tracked": True, "initialized": False, "camera_center": (0, 0, 0)},
        {"tracked": True, "camera_center": None},
        {"tracked": True, "camera_center": (float("nan"), 0, 0)},
        {"origin_frame_index": 2},
        {"inliers": -1},
    ],
)
def test_pose_schema_rejects_fabricated_or_invalid_positions(invalid):
    values = {
        "frame_index": 1,
        "status": "tracked",
        "tracked": True,
        "initialized": True,
        "camera_center": (0, 0, 0),
        "origin_frame_index": 0,
    }
    values.update(invalid)
    with pytest.raises(ValidationError):
        VisualPoseEvidence.model_validate(values)


def test_configuration_cannot_require_more_landmarks_than_its_budget():
    with pytest.raises(ValidationError, match="landmark budget"):
        VisualOdometryConfig(minimum_tracking_inliers=40, landmark_budget=32)
    with pytest.raises(ValidationError, match="feature budget"):
        VisualOdometryConfig(
            minimum_map_points=40, features=VisualGeometryConfig(feature_budget=32)
        )
