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
    assert result["handle_pairs"] == [[0, 3], [3, 0], [0, 1], [2, 3]]
    assert result["ranked_pairs"][0]["robustly_noncollinear"]
    assert result["ranked_pairs"][0]["conditional_turn_bounds"] == [86, 94]
    assert all(first != second for first, second in result["handle_pairs"])
    assert propose_control_pairs(_samples()) == result


def test_unknown_directions_remain_unknown_and_exploration_covers_all_handles():
    result = propose_control_pairs(())
    assert all(item["expected_turn_degrees"] is None for item in result["ranked_pairs"])
    assert result["handle_pairs"][:2] == [[0, 1], [2, 3]]
    assert len(result["handle_pairs"]) == 4


def test_single_observation_is_a_proposal_not_an_exploitation_certificate():
    result = propose_control_pairs(_samples()[::2])
    assert result["handle_pairs"][:2] == [[0, 3], [3, 0]]
    assert all(
        item["role"] != "conditional_exploitation" for item in result["selections"]
    )
    assert not any(item["robustly_noncollinear"] for item in result["ranked_pairs"])


def test_near_opposite_known_pair_is_not_exploited_for_positive_sine_score():
    samples = tuple(
        item.model_copy(
            update={
                "direction": (-float(np.cos(np.pi / 10)), 0, float(np.sin(np.pi / 10)))
            }
        )
        if item.handle == 3
        else item
        for item in _samples()
    )
    result = propose_control_pairs(samples)
    opposite = next(
        item for item in result["ranked_pairs"] if item["handles"] == [0, 3]
    )
    assert opposite["expected_turn_degrees"] == pytest.approx(162)
    assert opposite["information_proxy"] == 0
    assert all(
        item["role"] == "anchored_unknown_exploration" for item in result["selections"]
    )
    assert all(
        any(handle in (0, 3) for handle in pair) for pair in result["handle_pairs"]
    )
    assert {handle for pair in result["handle_pairs"] for handle in pair} == {
        0,
        1,
        2,
        3,
    }


def test_missing_sample_does_not_remove_potentially_useful_new_direction():
    samples = tuple(
        item.model_copy(update={"direction": (-1.0, 0.0, 0.0)})
        if item.handle == 3
        else item
        for item in _samples()
    ) + (
        DirectionSample(
            handle=2,
            first=30,
            second=31,
            direction=(0, 0, 1),
            sensitivity_radius_degrees=2,
        ),
    )
    result = propose_control_pairs(samples)
    assert result["handle_pairs"][:2] == [[0, 2], [2, 0]]
    assert all(
        1 in pair and any(handle in (0, 3) for handle in pair)
        for pair in result["handle_pairs"][2:]
    )
    assert [item["role"] for item in result["selections"]] == [
        "provisional_direction_resampling",
        "provisional_direction_resampling",
        "anchored_unknown_exploration",
        "anchored_unknown_exploration",
    ]


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
