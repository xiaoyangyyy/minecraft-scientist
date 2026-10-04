import copy

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")
cv2 = pytest.importorskip("cv2")

from ceres_fit_probe import ceres_bundle, isolated_optimizer, shared

from minecraft_scientist.exploration.joint_multiview_map import JointMapConfig, _project


@pytest.fixture
def geometry():
    rng = np.random.default_rng(781)
    points = rng.uniform((-2, -1.5, 4), (2, 1.5, 9), (60, 3))
    camera = np.array(((180, 0, 224), (0, 320, 224), (0, 0, 1)), dtype=float)
    rotations = np.repeat(np.eye(3)[None], 5, axis=0)
    translations = -np.array(
        [(0.4 * slot, 0.02 * np.sin(slot), 0.1 * slot) for slot in range(5)]
    )
    scale = np.linalg.norm(translations[2])
    points, translations = points / scale, translations / scale
    pixels = np.array(
        [
            _project(points, rotation, shift, camera)[0]
            for rotation, shift in zip(rotations, translations, strict=True)
        ]
    )
    perturbed = points + rng.normal(0, 0.005, points.shape)
    return perturbed, rotations, translations, pixels, camera


def test_ceres_training_fit_preserves_inputs_origin_gauge_and_pixel_accuracy(geometry):
    original = copy.deepcopy(geometry)
    result = ceres_bundle(*geometry, 2, JointMapConfig(), cv2, pixel_scale=2)
    assert result is not None
    points, rotations, translations, history = result
    assert history[-1] <= history[0]
    assert np.allclose(rotations[0], np.eye(3), atol=1e-7)
    assert np.allclose(translations[0], 0, atol=1e-7)
    assert np.isclose(np.linalg.norm(translations[2]), 1, atol=1e-7)
    for slot in range(5):
        assert (
            np.max(
                np.linalg.norm(
                    _project(points, rotations[slot], translations[slot], geometry[-1])[
                        0
                    ]
                    - geometry[3][slot],
                    axis=1,
                )
            )
            < 0.01
        )
    assert all(
        np.array_equal(first, second)
        for first, second in zip(geometry, original, strict=True)
    )


@pytest.mark.parametrize("bad", ["sensor", "origin"])
def test_probe_rejects_unregistered_sensor_or_origin_before_solver(geometry, bad):
    values = list(geometry)
    if bad == "sensor":
        values[-1][0, 2] = 112
    else:
        values[2][0, 0] = 0.1
    with pytest.raises(ValueError, match="registered 448-pixel"):
        ceres_bundle(*values, 2, JointMapConfig(), cv2)


def test_nonfinite_or_behind_camera_initialization_is_not_a_map(geometry):
    values = list(geometry)
    values[0][:, 2] = -1
    assert ceres_bundle(*values, 2, JointMapConfig(), cv2) is None


def test_optimizer_alias_restores_on_exception():
    original = shared._bundle
    with pytest.raises(RuntimeError, match="synthetic"), isolated_optimizer():
        assert shared._bundle is ceres_bundle
        raise RuntimeError("synthetic failure")
    assert shared._bundle is original
