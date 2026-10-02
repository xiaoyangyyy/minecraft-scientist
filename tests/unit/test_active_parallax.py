import numpy as np
import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration.active_parallax import (
    DirectionSample,
    ParallaxPolicy,
    direction_hypotheses,
    propose_control_pairs,
)


def _samples():
    return tuple(
        DirectionSample(
            handle=handle,
            first=10 * handle + slot,
            second=10 * handle + slot + 1,
            direction=direction,
            sensitivity_radius_degrees=2,
        )
        for handle, direction in ((0, (1, 0, 0)), (3, (0, 0, 1)))
        for slot in (0, 2)
    )


def test_active_pairs_use_directional_diversity_and_keep_unknown_exploration():
    result = propose_control_pairs(_samples())
    assert result["handle_pairs"] == [[0, 3], [3, 0], [1, 2], [2, 1]]
    assert result["ranked_pairs"][0]["robustly_noncollinear"]
    assert result["ranked_pairs"][0]["conditional_turn_bounds"] == [86, 94]
    assert all(first != second for first, second in result["handle_pairs"])
    assert propose_control_pairs(_samples()) == result


def test_unknown_or_single_observation_does_not_invent_a_direction():
    for samples in ((), _samples()[::2]):
        result = propose_control_pairs(samples)
        assert all(
            item["expected_turn_degrees"] is None for item in result["ranked_pairs"]
        )
        assert len(result["handle_pairs"]) == 4


def test_repeated_image_intervals_cannot_inflate_support():
    samples = _samples()
    with pytest.raises(ValueError, match="duplicate"):
        direction_hypotheses(samples + samples[:1])


def test_contradictory_direction_samples_remain_unknown():
    samples = tuple(
        DirectionSample(
            handle=0,
            first=index,
            second=index + 1,
            direction=direction,
            sensitivity_radius_degrees=2,
        )
        for index, direction in ((0, (1, 0, 0)), (2, (-1, 0, 0)))
    )
    assert direction_hypotheses(samples) == ()
    assert all(
        item["expected_turn_degrees"] is None
        for item in propose_control_pairs(samples)["ranked_pairs"]
    )


def test_spread_and_sensitivity_are_not_called_true_confidence():
    samples = tuple(
        item.model_copy(update={"sensitivity_radius_degrees": 60})
        for item in _samples()
    )
    result = propose_control_pairs(samples)
    assert not any(item["robustly_noncollinear"] for item in result["ranked_pairs"])
    assert len(result["handle_pairs"]) == 4  # Still an experiment, not accepted motion.


@pytest.mark.parametrize(
    "change",
    [
        {"pair_budget": 2, "exploitation_budget": 2},
        {"minimum_samples": 1},
        {"pair_budget": True},
        {"minimum_turn_degrees": np.nan},
    ],
)
def test_invalid_or_nonexploratory_policy_rejects(change):
    with pytest.raises(ValidationError):
        ParallaxPolicy(**change)


def test_nonunit_or_empty_interval_direction_rejects():
    for direction, first, second in (((2, 0, 0), 0, 1), ((1, 0, 0), 1, 1)):
        with pytest.raises(ValidationError):
            DirectionSample(
                handle=0,
                first=first,
                second=second,
                direction=direction,
                sensitivity_radius_degrees=2,
            )
