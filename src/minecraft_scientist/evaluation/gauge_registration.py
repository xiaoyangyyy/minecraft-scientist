"""Evaluator-only relative gauge diagnostics; never feed them to public control."""

from __future__ import annotations

import math

import numpy as np

from minecraft_scientist.exploration.map_gauge_bridge import (
    GaugeBridgePolicy,
    GaugeSimilarity,
)


def score_gauge_truth(
    similarity,
    displacement,
    *,
    parent_baseline,
    child_baseline,
    parent_gauge_baseline,
    child_gauge_baseline,
    policy=None,
):
    """Assess an already frozen candidate, without choosing/promoting a map pair.

    Metric baselines are privileged evaluation calibration, not agent scale.
    Parent/child gauge baselines account for their explicit bootstrap normalizers.
    """
    model = GaugeSimilarity.model_validate(similarity)
    settings = GaugeBridgePolicy.model_validate(
        policy.model_dump() if isinstance(policy, GaugeBridgePolicy) else policy or {}
    )
    values = (
        parent_baseline,
        child_baseline,
        parent_gauge_baseline,
        child_gauge_baseline,
    )
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in values
    ):
        raise ValueError("gauge truth baselines must be finite nonnegative scalars")
    relative = np.asarray(displacement["relative_rotation"], dtype=float)
    direction = displacement["unit_translation"]
    distance = displacement["distance"]
    if (
        type(distance) not in (int, float)
        or not math.isfinite(distance)
        or distance < 0
        or relative.shape != (3, 3)
        or not np.isfinite(relative).all()
        or not np.allclose(relative @ relative.T, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(relative), 1, atol=1e-6)
        or (distance <= 1e-9 and direction is not None)
        or (
            distance > 1e-9
            and (
                np.asarray(direction).shape != (3,)
                or not np.isfinite(direction).all()
                or not np.isclose(np.linalg.norm(direction), 1, atol=1e-6)
            )
        )
    ):
        raise ValueError("gauge truth must be normalized finite chronological motion")
    result = {
        "status": "unobservable_evaluator_baseline",
        "parameters_correct": None,
        "scale_relative_error": None,
        "rotation_error_degrees": None,
        "center_error_parent_gauge_units": None,
        "center_error_evaluator_calibrated_blocks": None,
        "parent_evaluator_baseline_blocks": parent_baseline,
        "child_evaluator_baseline_blocks": child_baseline,
        "diagnostic_only": True,
        "agent_metric_scale_known": False,
    }
    # Tiny ground-truth baselines cannot calibrate relative monocular scale.
    if (
        min(parent_baseline, child_baseline) < 0.15
        or min(parent_gauge_baseline, child_gauge_baseline) <= 1e-9
    ):
        return result
    units = parent_baseline / parent_gauge_baseline
    expected_scale = (child_baseline / child_gauge_baseline) / units
    expected_center = (
        (-relative.T @ np.asarray(direction) * distance / units)
        if distance > 1e-9
        else np.zeros(3)
    )
    scale_error = abs(model.scale / expected_scale - 1)
    center_error = float(
        np.linalg.norm(np.asarray(model.translation) - expected_center)
    )
    rotation_error = float(
        np.degrees(
            np.arccos(
                np.clip(
                    (np.trace(np.asarray(model.rotation).T @ relative.T) - 1) / 2, -1, 1
                )
            )
        )
    )
    correct = (
        scale_error <= settings.maximum_relative_point_error
        and center_error <= settings.maximum_center_error
        and rotation_error <= settings.maximum_rotation_error_degrees
    )
    result.update(
        status="parameters_within_registered_tolerances"
        if correct
        else "parameters_outside_registered_tolerances",
        parameters_correct=bool(correct),
        scale_relative_error=float(scale_error),
        rotation_error_degrees=rotation_error,
        center_error_parent_gauge_units=center_error,
        center_error_evaluator_calibrated_blocks=center_error * units,
    )
    return result
