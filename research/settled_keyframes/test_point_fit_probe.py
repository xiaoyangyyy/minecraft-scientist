import copy

import numpy as np
import pytest
from point_fit_probe import refine_past_points

from minecraft_scientist.exploration.joint_multiview_map import JointMapConfig, _project


@pytest.fixture
def fixed_rays():
    rng = np.random.default_rng(804)
    points = rng.uniform((-2, -1.5, 4), (2, 1.5, 9), (48, 3))
    camera = np.array(((180, 0, 224), (0, 320, 224), (0, 0, 1)), dtype=float)
    rotations = np.repeat(np.eye(3)[None], 12, axis=0)
    translations = -np.array(
        [(0.09 * slot, 0.02 * slot, 0.03 * slot) for slot in range(12)]
    )
    pixels = np.array(
        [
            _project(points, rotation, shift, camera)[0]
            for rotation, shift in zip(rotations, translations, strict=True)
        ]
    )
    return pixels, rotations, translations, camera, points


def test_exact_past_rays_recover_points_without_changing_inputs(fixed_rays):
    inputs = fixed_rays[:4]
    original = copy.deepcopy(inputs)
    result = refine_past_points(*inputs, config=JointMapConfig(), scale=2)
    assert result is not None
    points, diagnostic = result
    assert np.allclose(points, fixed_rays[4], atol=1e-7)
    assert diagnostic["poses_refined"] is False
    assert diagnostic["fitting_frames"] == 12
    assert diagnostic["every_point_cost_nonincreasing"]
    assert all(np.array_equal(a, b) for a, b in zip(inputs, original, strict=True))
    # A genuinely unseen synthetic camera is not an input to fitting.
    shift = np.array((-1.4, 0.08, -0.2))
    assert np.allclose(
        _project(points, np.eye(3), shift, inputs[3])[0],
        _project(fixed_rays[4], np.eye(3), shift, inputs[3])[0],
        atol=1e-6,
    )


def test_noisy_rays_reduce_robust_cost_deterministically(fixed_rays):
    inputs = list(fixed_rays[:4])
    rng = np.random.default_rng(805)
    inputs[0] = inputs[0] + rng.normal(0, 0.5, inputs[0].shape)
    inputs[0][3, :8] += (3.5, -2.5)
    first = refine_past_points(*inputs, config=JointMapConfig(), scale=2)
    second = refine_past_points(*inputs, config=JointMapConfig(), scale=2)
    assert first is not None and second is not None
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert first[1]["final_cost"] < first[1]["initial_cost"]
    assert np.all(np.diff(first[1]["cost_history"]) <= 1e-10)


@pytest.mark.parametrize("bad", ["nonfinite", "rotation", "shape", "scale"])
def test_invalid_fixed_geometry_is_rejected(fixed_rays, bad):
    inputs = list(fixed_rays[:4])
    scale = 2
    if bad == "nonfinite":
        inputs[0][0, 0, 0] = np.nan
    elif bad == "rotation":
        inputs[1][0, 0, 0] = 2
    elif bad == "shape":
        inputs[2] = inputs[2][:-1]
    else:
        scale = 0
    with pytest.raises(ValueError, match="finite fixed-pose"):
        refine_past_points(*inputs, config=JointMapConfig(), scale=scale)


def test_zero_baseline_does_not_create_a_map(fixed_rays):
    inputs = list(fixed_rays[:4])
    inputs[0] = np.repeat(inputs[0][:1], 12, axis=0)
    inputs[2] = np.zeros_like(inputs[2])
    assert refine_past_points(*inputs, config=JointMapConfig(), scale=2) is None
