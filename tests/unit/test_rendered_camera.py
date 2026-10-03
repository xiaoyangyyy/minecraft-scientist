import copy

import numpy as np
import pytest
from pydantic import ValidationError

from minecraft_scientist.evaluation.relative_motion import (
    camera_rotation,
    relative_motion_truth,
    score_motion_truth,
    score_relative_motion,
)
from minecraft_scientist.evaluation.rendered_camera import (
    WorldRenderSnapshot,
    decompose_rendered_camera,
    rendered_motion_truth,
    score_rendered_motion,
)


def snapshot(sequence=1, base=(100.0, 70.0, -40.0), yaw=30, pitch=5, effects=None):
    projection = np.array(
        ((0.8, 0, 0, 0), (0, 1.4, 0, 0), (0, 0, -1.001, -0.1), (0, 0, -1, 0))
    )
    view = np.eye(4)
    view[:3, :3] = np.diag((1, -1, -1)) @ camera_rotation(yaw, pitch)
    if effects is not None:
        projection = projection @ effects
    return {
        "sequence": sequence,
        "partial_ticks": 0.0,
        "model_view": view.tolist(),
        "projection": projection.tolist(),
        "base_position": list(base),
        "shader_pack_loaded": False,
        "shadow_pass": False,
    }


def test_pure_matrix_decomposition_recovers_opencv_axes_and_intrinsics():
    record = snapshot()
    camera = decompose_rendered_camera(record, observation_size=(448, 448))
    np.testing.assert_allclose(camera.rotation, camera_rotation(30, 5), atol=1e-12)
    np.testing.assert_allclose(camera.center, record["base_position"], atol=1e-12)
    np.testing.assert_allclose(
        camera.camera, ((179.2, 0, 224), (0, 313.6, 224), (0, 0, 1)), atol=1e-12
    )
    assert camera.reconstruction_error < 1e-12 and not camera.navigation_accepted


def test_projection_effects_are_retained_in_pose_not_discarded_or_fit_to_pixels():
    angle = 0.04
    effects = np.array(
        (
            (np.cos(angle), -np.sin(angle), 0, 0.02),
            (np.sin(angle), np.cos(angle), 0, -0.03),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    record = snapshot(effects=effects)
    camera = decompose_rendered_camera(record, observation_size=(448, 448))
    gl_rotation = np.array(record["model_view"])[:3, :3]
    expected_rotation = np.diag((1, -1, -1)) @ effects[:3, :3] @ gl_rotation
    expected_offset = -gl_rotation.T @ effects[:3, :3].T @ effects[:3, 3]
    np.testing.assert_allclose(camera.rotation, expected_rotation, atol=1e-12)
    np.testing.assert_allclose(camera.camera_center_offset, expected_offset, atol=1e-12)
    np.testing.assert_allclose(
        camera.center, np.array(record["base_position"]) + expected_offset, atol=1e-12
    )
    # Recover direct clip-projected pixels, including effects, from unrelated world points.
    world = np.array(((101, 70, -30), (99, 69, -28), (101, 71, -25)), dtype=float)
    local = world - record["base_position"]
    clip = (
        np.column_stack((local, np.ones(3)))
        @ (np.array(record["projection"]) @ np.array(record["model_view"])).T
    )
    direct = np.column_stack(
        (224 * (clip[:, 0] / clip[:, 3] + 1), 224 * (1 - clip[:, 1] / clip[:, 3]))
    )
    predicted = (
        (world - camera.center)
        @ np.array(camera.rotation).T
        @ np.array(camera.camera).T
    )
    np.testing.assert_allclose(
        predicted[:, :2] / predicted[:, 2, None], direct, atol=1e-10
    )


def test_observation_resize_changes_only_intrinsics():
    record = snapshot()
    large = decompose_rendered_camera(record, observation_size=(448, 448))
    small = decompose_rendered_camera(record, observation_size=(224, 224))
    np.testing.assert_allclose(large.rotation, small.rotation, atol=1e-12)
    np.testing.assert_allclose(large.center, small.center, atol=1e-12)
    np.testing.assert_allclose(
        np.array(large.camera)[:2], np.array(small.camera)[:2] * 2, atol=1e-12
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("sequence", True),
        ("sequence", 0),
        ("partial_ticks", float("nan")),
        ("shader_pack_loaded", True),
        ("shader_pack_loaded", 0),
        ("shadow_pass", True),
        ("shadow_pass", 0),
    ],
)
def test_invalid_or_unsupported_render_snapshot_rejected(key, value):
    with pytest.raises((ValueError, ValidationError)):
        decompose_rendered_camera(
            snapshot() | {key: value}, observation_size=(448, 448)
        )


@pytest.mark.parametrize(
    "case",
    [
        "view-scale",
        "view-reflection",
        "view-nonaffine",
        "singular-projection",
        "ill-conditioned",
        "negative-handedness",
        "nan",
        "string",
        "bool",
    ],
)
def test_malformed_or_unobservable_matrices_fail_closed(case):
    record = snapshot()
    if case == "view-scale":
        record["model_view"][0][0] *= 2
    elif case == "view-reflection":
        record["model_view"][0] = [-value for value in record["model_view"][0]]
    elif case == "view-nonaffine":
        record["model_view"][3][0] = 0.1
    elif case == "singular-projection":
        record["projection"][0] = [0, 0, 0, 0]
    elif case == "ill-conditioned":
        record["projection"][0][0] = 1e-14
    elif case == "negative-handedness":
        record["projection"][0] = [-value for value in record["projection"][0]]
    else:
        record["projection"][0][0] = {
            "nan": float("nan"),
            "string": "0.8",
            "bool": True,
        }[case]
    with pytest.raises((ValueError, ValidationError)):
        decompose_rendered_camera(record, observation_size=(448, 448))


@pytest.mark.parametrize(
    "size", [(448,), (448.0, 448), (True, 448), (10, 448), (9000, 448)]
)
def test_invalid_size_rejected(size):
    with pytest.raises(ValueError):
        decompose_rendered_camera(snapshot(), observation_size=size)


def test_rendered_and_body_motion_use_identical_scoring_when_camera_models_agree():
    body0 = {"xpos": 100.0, "ypos": 70.0, "zpos": -40.0, "yaw": 30.0, "pitch": 5.0}
    body1 = body0 | {"xpos": 100.3, "zpos": -39.6, "yaw": 33.0}
    first = decompose_rendered_camera(snapshot(), observation_size=(448, 448))
    second = decompose_rendered_camera(
        snapshot(sequence=2, base=(100.3, 70.0, -39.6), yaw=33),
        observation_size=(448, 448),
    )
    truth = relative_motion_truth(body0, body1)
    prediction = {
        "supported": True,
        "relative_rotation": truth["relative_rotation"],
        "unit_translation": truth["unit_translation"],
    }
    camera_score = score_rendered_motion(prediction, first, second)
    body_score = score_relative_motion(prediction, body0, body1)
    assert camera_score["correct"] and body_score["correct"]
    assert camera_score["translation_distance_3d"] == pytest.approx(
        body_score["translation_distance_3d"]
    )
    assert camera_score["direction_error_degrees"] == pytest.approx(
        body_score["direction_error_degrees"], abs=1e-5
    )
    assert camera_score["rotation_error_degrees"] == pytest.approx(
        body_score["rotation_error_degrees"], abs=1e-5
    )
    with pytest.raises(ValueError, match="chronological"):
        rendered_motion_truth(second, first)


@pytest.mark.parametrize(
    "case", ["distance", "boolean-distance", "rotation", "unit", "stationary-unit"]
)
def test_shared_scoring_rejects_bad_truth(case):
    truth = {
        "distance": 1.0,
        "relative_rotation": np.eye(3),
        "unit_translation": np.array((1.0, 0, 0)),
    }
    if case == "distance":
        truth["distance"] = float("nan")
    elif case == "boolean-distance":
        truth["distance"] = True
    elif case == "rotation":
        truth["relative_rotation"] = np.ones((3, 3))
    elif case == "unit":
        truth["unit_translation"] *= 2
    else:
        truth["distance"] = 0
    with pytest.raises(ValueError, match="truth"):
        score_motion_truth({"supported": False}, truth)


def test_matrix_input_is_immutable_and_replay_exact():
    record = snapshot()
    original = copy.deepcopy(record)
    first = decompose_rendered_camera(record, observation_size=(448, 448))
    assert record == original
    assert first == decompose_rendered_camera(
        WorldRenderSnapshot.model_validate(record), observation_size=(448, 448)
    )
