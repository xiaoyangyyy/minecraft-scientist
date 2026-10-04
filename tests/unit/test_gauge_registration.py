import numpy as np
import pytest

from minecraft_scientist.evaluation.gauge_registration import score_gauge_truth
from minecraft_scientist.exploration.map_gauge_bridge import GaugeSimilarity


@pytest.fixture
def example():
    angle = 0.3
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))
    center = np.array((0.2, 0.1, 0.4))
    model = GaugeSimilarity(
        scale=2, rotation=rotation.tolist(), translation=tuple(center)
    )
    shift = -rotation.T @ center * 2
    distance = float(np.linalg.norm(shift))
    truth = {
        "relative_rotation": rotation.T,
        "unit_translation": shift / distance,
        "distance": distance,
    }
    kwargs = {
        "parent_baseline": 2.0,
        "child_baseline": 4.0,
        "parent_gauge_baseline": 1.0,
        "child_gauge_baseline": 1.0,
    }
    return model, truth, kwargs


def test_evaluator_calibration_does_not_become_agent_metric_scale(example):
    model, truth, kwargs = example
    result = score_gauge_truth(model, truth, **kwargs)
    assert result["parameters_correct"]
    assert result["scale_relative_error"] == pytest.approx(0)
    assert result["center_error_evaluator_calibrated_blocks"] == pytest.approx(0)
    assert result["rotation_error_degrees"] == pytest.approx(0, abs=1e-6)
    assert result["diagnostic_only"] and not result["agent_metric_scale_known"]


def test_nonunit_map_bootstrap_baselines_are_accounted_for(example):
    model, truth, kwargs = example
    kwargs.update(parent_gauge_baseline=2.0, child_gauge_baseline=2.0)
    model = model.model_copy(
        update={"translation": tuple(np.asarray(model.translation) * 2)}
    )
    assert score_gauge_truth(model, truth, **kwargs)["parameters_correct"]


@pytest.mark.parametrize(
    "key",
    [
        "parent_baseline",
        "child_baseline",
        "parent_gauge_baseline",
        "child_gauge_baseline",
    ],
)
def test_unobservable_truth_never_returns_a_correct_score(example, key):
    model, truth, kwargs = example
    kwargs[key] = 0.0
    result = score_gauge_truth(model, truth, **kwargs)
    assert result["parameters_correct"] is None
    assert result["status"] == "unobservable_evaluator_baseline"


@pytest.mark.parametrize(
    "key,value",
    [
        ("parent_baseline", True),
        ("child_baseline", float("nan")),
        ("parent_gauge_baseline", -1),
    ],
)
def test_invalid_baselines_reject(example, key, value):
    model, truth, kwargs = example
    kwargs[key] = value
    with pytest.raises(ValueError):
        score_gauge_truth(model, truth, **kwargs)


@pytest.mark.parametrize(
    "case",
    [
        "scale",
        "center",
        "rotation",
        "truth_rotation",
        "truth_direction",
        "truth_distance",
    ],
)
def test_wrong_similarity_is_retained_without_private_runner_up_selection(
    example, case
):
    model, truth, kwargs = example
    if case.startswith("truth_"):
        truth[
            case.removeprefix("truth_")
            if case == "truth_distance"
            else {
                "truth_rotation": "relative_rotation",
                "truth_direction": "unit_translation",
            }[case]
        ] = (
            -1.0
            if case == "truth_distance"
            else np.zeros((3, 3))
            if case == "truth_rotation"
            else np.ones(3)
        )
        with pytest.raises(ValueError):
            score_gauge_truth(model, truth, **kwargs)
        return
    if case == "scale":
        model = model.model_copy(update={"scale": 3.0})
    elif case == "center":
        model = model.model_copy(update={"translation": (1.0, 1.0, 1.0)})
    else:
        model = model.model_copy(update={"rotation": tuple(map(tuple, np.eye(3)))})
    result = score_gauge_truth(model, truth, **kwargs)
    assert result["parameters_correct"] is False
