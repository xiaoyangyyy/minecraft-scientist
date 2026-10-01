import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")

from minecraft_scientist.exploration.dense_correspondence import DenseCorrespondences
from minecraft_scientist.exploration.two_view_motion import estimate_two_view_motion

CAMERA = np.array(((90.0, 0, 112), (0, 90.0, 112), (0, 0, 1)))


def _project(points, rotation=None, translation=None):
    camera_points = points @ (np.eye(3) if rotation is None else rotation).T
    camera_points += np.zeros(3) if translation is None else translation
    image = camera_points @ CAMERA.T
    return image[:, :2] / image[:, 2:]


def _evidence(first, second):
    return DenseCorrespondences(
        points0=tuple(map(tuple, first)),
        points1=tuple(map(tuple, second)),
        confidence=(0.9,) * len(first),
        image_shape0=(224, 224),
        image_shape1=(224, 224),
        raw_match_count=len(first),
    )


def _points():
    rng = np.random.default_rng(70)
    return np.column_stack(
        (rng.uniform(-2, 2, 150), rng.uniform(-2, 2, 150), rng.uniform(4, 9, 150))
    )


def test_supported_nonplanar_motion_recovers_translation_and_rotation():
    points = _points()
    rotation = cv2.Rodrigues(np.array((0.01, 0.07, 0.02)))[0]
    translation = np.array((-0.4, 0.05, -0.1))
    result = estimate_two_view_motion(
        _evidence(_project(points), _project(points, rotation, translation)), CAMERA
    )
    assert result.supported and result.triangulated_inliers >= 15
    np.testing.assert_allclose(result.relative_rotation, rotation, atol=1e-4)
    np.testing.assert_allclose(
        result.unit_translation, translation / np.linalg.norm(translation), atol=1e-4
    )


@pytest.mark.parametrize("angle", [0.0, 0.12, -0.15])
def test_pure_rotation_and_identical_frames_cannot_supply_translation(angle):
    points = _points()
    rotation = cv2.Rodrigues(np.array((0, angle, 0)))[0]
    result = estimate_two_view_motion(
        _evidence(_project(points), _project(points, rotation)), CAMERA
    )
    assert not result.supported
    assert result.reason == "pure rotation or stationary ambiguity"
    assert result.unit_translation is None


def test_geometry_rejects_insufficient_or_localized_correspondences():
    points = _points()
    first, second = (
        _project(points),
        _project(points, translation=np.array((-0.4, 0, 0))),
    )
    assert not estimate_two_view_motion(
        _evidence(first[:10], second[:10]), CAMERA
    ).supported
    first = np.tile((80, 80), (30, 1)).astype(float)
    second = first + (1, 1)
    result = estimate_two_view_motion(_evidence(first, second), CAMERA)
    assert not result.supported and result.reason == "insufficient spatial coverage"


def test_geometry_rejects_invalid_intrinsics():
    points = _project(_points())
    for invalid in (
        np.zeros((3, 3)),
        np.eye(2),
        np.array(((90, 1, 112), (0, 90, 112), (0, 0, 1))),
    ):
        with pytest.raises(ValueError):
            estimate_two_view_motion(_evidence(points, points), invalid)
