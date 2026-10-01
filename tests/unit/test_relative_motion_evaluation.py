import numpy as np
import pytest

from minecraft_scientist.evaluation.relative_motion import (
    camera_rotation,
    relative_motion_truth,
    score_relative_motion,
)

ORIGIN = {"xpos": 0.0, "ypos": 0.0, "zpos": 0.0, "yaw": 0.0, "pitch": 0.0}


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (90, 0), (270, -20), (23, 40)])
def test_minecraft_camera_basis_is_proper(yaw, pitch):
    rotation = camera_rotation(yaw, pitch)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)


def test_translation_is_in_query_camera_frame_not_world_axes():
    forward = relative_motion_truth(ORIGIN, ORIGIN | {"zpos": 1})
    np.testing.assert_allclose(forward["unit_translation"], (0, 0, -1))
    upward = relative_motion_truth(ORIGIN, ORIGIN | {"ypos": 1})
    np.testing.assert_allclose(upward["unit_translation"], (0, 1, 0))
    westward = relative_motion_truth(
        ORIGIN | {"yaw": 90}, ORIGIN | {"yaw": 90, "xpos": -1}
    )
    np.testing.assert_allclose(westward["unit_translation"], (0, 0, -1), atol=1e-12)


def test_correct_supported_motion_and_wrong_sign_have_separate_scores():
    target = ORIGIN | {"zpos": 1}
    prediction = {
        "supported": True,
        "relative_rotation": np.eye(3).tolist(),
        "unit_translation": (0, 0, -1),
    }
    correct = score_relative_motion(prediction, ORIGIN, target)
    assert correct["correct"] and correct["direction_error_degrees"] == pytest.approx(0)
    wrong = score_relative_motion(
        prediction | {"unit_translation": (0, 0, 1)}, ORIGIN, target
    )
    assert not wrong["correct"] and wrong["direction_error_degrees"] == pytest.approx(
        180
    )


def test_unsupported_motion_is_not_zero_error_or_success():
    result = score_relative_motion({"supported": False}, ORIGIN, ORIGIN | {"zpos": 1})
    assert not result["correct"] and result["status"] == "unsupported_motion"
    assert (
        result["direction_error_degrees"] is None
        and result["rotation_error_degrees"] is None
    )


def test_rotation_only_control_cannot_report_supported_translation():
    target = ORIGIN | {"yaw": 30}
    rotation = relative_motion_truth(ORIGIN, target)["relative_rotation"]
    good = score_relative_motion({"supported": False}, ORIGIN, target)
    bad = score_relative_motion(
        {
            "supported": True,
            "relative_rotation": rotation,
            "unit_translation": (0, 0, 1),
        },
        ORIGIN,
        target,
    )
    assert (
        good["correct"] and not bad["correct"] and bad["status"] == "false_translation"
    )


def test_evaluator_rejects_invalid_pose_and_nonunit_prediction():
    with pytest.raises(ValueError):
        relative_motion_truth(ORIGIN, ORIGIN | {"xpos": float("nan")})
    with pytest.raises(ValueError):
        camera_rotation(0, 100)
    with pytest.raises(ValueError):
        score_relative_motion(
            {
                "supported": True,
                "relative_rotation": np.eye(3),
                "unit_translation": (0, 0, 4),
            },
            ORIGIN,
            ORIGIN | {"zpos": 1},
        )


@pytest.mark.parametrize(
    "name", ["minimum_translation", "maximum_direction_error", "maximum_rotation_error"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_threshold_cannot_turn_real_motion_into_a_correct_control(
    name, value
):
    with pytest.raises(ValueError, match="thresholds"):
        score_relative_motion(
            {"supported": False}, ORIGIN, ORIGIN | {"zpos": 1}, **{name: value}
        )
