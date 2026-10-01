import numpy as np
import pytest

from minecraft_scientist.evaluation.visual_pose import fit_similarity


def test_similarity_recovers_gauge_on_calibration_and_predicts_separate_points():
    rng = np.random.default_rng(4)
    calibration = rng.normal(size=(10, 3))
    theta = 0.7
    rotation = np.array(
        (
            (np.cos(theta), 0, np.sin(theta)),
            (0, 1, 0),
            (-np.sin(theta), 0, np.cos(theta)),
        )
    )
    translation = np.array((12, -4, 7))
    target = 3 * calibration @ rotation.T + translation
    scale, recovered, offset = fit_similarity(calibration, target)
    assert scale == pytest.approx(3)
    np.testing.assert_allclose(recovered, rotation, atol=1e-12)
    np.testing.assert_allclose(offset, translation, atol=1e-12)
    separate = rng.normal(size=(7, 3))
    np.testing.assert_allclose(
        scale * separate @ recovered.T + offset,
        3 * separate @ rotation.T + translation,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "points",
    [np.zeros((5, 3)), np.column_stack((np.arange(5), np.zeros(5), np.zeros(5)))],
)
def test_unobservable_gauge_is_rejected(points):
    with pytest.raises(ValueError, match="unobservable"):
        fit_similarity(points, points)


def test_similarity_cannot_fit_reflections_as_camera_rotations():
    points = np.random.default_rng(0).normal(size=(20, 3))
    reflected = points * (-1, 1, 1)
    scale, rotation, translation = fit_similarity(points, reflected)
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert np.linalg.norm(scale * points @ rotation.T + translation - reflected) > 1


@pytest.mark.parametrize(
    "source,target",
    [
        (np.zeros((2, 3)), np.zeros((2, 3))),
        (np.zeros((5, 2)), np.zeros((5, 2))),
        (np.zeros((5, 3)), np.zeros((6, 3))),
        (np.full((5, 3), np.nan), np.zeros((5, 3))),
    ],
)
def test_invalid_gauge_evidence_is_rejected(source, target):
    with pytest.raises(ValueError):
        fit_similarity(source, target)
