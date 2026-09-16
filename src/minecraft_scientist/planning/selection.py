"""Shared information-gain, cost, and control-risk experiment utility."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar


@dataclass(frozen=True, slots=True)
class ExperimentUtilityWeights:
    information_gain: float = 1.0
    cost: float = 0.05
    control_failure: float = 0.50


DEFAULT_UTILITY_WEIGHTS = ExperimentUtilityWeights()
Pair = TypeVar("Pair")


def experiment_utility(
    features: Sequence[float],
    weights: ExperimentUtilityWeights = DEFAULT_UTILITY_WEIGHTS,
) -> float:
    entropy = float(features[1])
    normalized_cost = float(features[4]) if len(features) > 4 else 0.0
    failure_risk = float(features[5]) if len(features) > 5 else 0.0
    return (
        weights.information_gain * entropy
        - weights.cost * normalized_cost
        - weights.control_failure * failure_risk
    )


def select_best_experiment(
    pairs: Sequence[Pair],
    features: Sequence[Sequence[float]],
    weights: ExperimentUtilityWeights = DEFAULT_UTILITY_WEIGHTS,
) -> Pair | None:
    options = [
        (
            experiment_utility(row, weights),
            float(row[1]),
            -float(row[4]) if len(row) > 4 else 0.0,
            repr(pair),
            pair,
        )
        for pair, row in zip(pairs, features, strict=True)
        if float(row[2]) == 0.0
    ]
    if not options:
        return None
    _, entropy, _, _, pair = max(options, key=lambda item: item[:-1])
    return None if entropy == 0.0 else pair
