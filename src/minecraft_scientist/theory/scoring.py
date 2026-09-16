"""Shared validation and MDL-style scoring."""

from __future__ import annotations

import math

from minecraft_scientist.protocols import TheoryIR

from .ir import TheoryInterpreter


def intervention_error(predictions: list[bool], observations: list[bool]) -> float:
    if len(predictions) != len(observations):
        raise ValueError("predictions and observations must have equal length")
    if not observations:
        return 1.0
    return sum(
        pred != obs for pred, obs in zip(predictions, observations, strict=True)
    ) / len(observations)


def theory_score(
    ir: TheoryIR,
    predictions: list[bool],
    observations: list[bool],
    *,
    complexity_weight: float = 0.001,
    intervention_weight: float = 2.0,
) -> float:
    error = intervention_error(predictions, observations)
    probability = min(1.0 - 1e-9, max(1e-9, 1.0 - error))
    negative_log_likelihood = -len(observations) * math.log(probability)
    return (
        negative_log_likelihood
        + complexity_weight * TheoryInterpreter.description_length(ir)
        + intervention_weight * error
    )


def numeric_theory_score(
    ir: TheoryIR,
    predictions: list[float],
    observations: list[float],
    *,
    complexity_weight: float = 0.001,
    intervention_weight: float = 2.0,
) -> float:
    if len(predictions) != len(observations):
        raise ValueError("predictions and observations must have equal length")
    if not observations:
        return float("inf")
    rmse = math.sqrt(
        sum(
            (prediction - observation) ** 2
            for prediction, observation in zip(predictions, observations, strict=True)
        )
        / len(observations)
    )
    negative_log_likelihood = len(observations) * math.log1p(rmse)
    return (
        negative_log_likelihood
        + complexity_weight * TheoryInterpreter.description_length(ir)
        + intervention_weight * rmse
    )
