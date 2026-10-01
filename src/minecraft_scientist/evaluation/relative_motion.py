"""Evaluator-only camera convention and relative-motion scoring.

Minecraft position/yaw/pitch telemetry is deliberately not imported by perception
or planning. Player pose approximates rendered camera pose (view bobbing is a
known limitation), so these diagnostics do not certify full navigation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def camera_rotation(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    if (
        not math.isfinite(yaw_degrees)
        or not math.isfinite(pitch_degrees)
        or not -90 <= pitch_degrees <= 90
    ):
        raise ValueError("camera angles must be finite and pitch in [-90, 90]")
    yaw, pitch = math.radians(yaw_degrees), math.radians(pitch_degrees)
    # Minecraft yaw 0 faces +Z, yaw +90 faces -X; positive pitch looks down.
    # OpenCV camera axes are right, down, forward.
    return np.array(
        (
            (-math.cos(yaw), 0, -math.sin(yaw)),
            (
                math.sin(yaw) * math.sin(pitch),
                -math.cos(pitch),
                -math.cos(yaw) * math.sin(pitch),
            ),
            (
                -math.sin(yaw) * math.cos(pitch),
                -math.sin(pitch),
                math.cos(yaw) * math.cos(pitch),
            ),
        )
    )


def relative_motion_truth(
    first: Mapping[str, float], second: Mapping[str, float]
) -> dict:
    first_position = np.array(
        [first[key] for key in ("xpos", "ypos", "zpos")], dtype=float
    )
    second_position = np.array(
        [second[key] for key in ("xpos", "ypos", "zpos")], dtype=float
    )
    if not np.isfinite(first_position).all() or not np.isfinite(second_position).all():
        raise ValueError("evaluator positions must be finite")
    first_rotation = camera_rotation(first["yaw"], first["pitch"])
    second_rotation = camera_rotation(second["yaw"], second["pitch"])
    translation = second_rotation @ (first_position - second_position)
    distance = float(np.linalg.norm(translation))
    return {
        "distance": distance,
        "relative_rotation": second_rotation @ first_rotation.T,
        "unit_translation": translation / distance if distance > 1e-9 else None,
    }


def score_relative_motion(
    prediction: Mapping,
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    minimum_translation: float = 0.15,
    maximum_direction_error: float = 20,
    maximum_rotation_error: float = 5,
) -> dict:
    if (
        not all(
            math.isfinite(value)
            for value in (
                minimum_translation,
                maximum_direction_error,
                maximum_rotation_error,
            )
        )
        or minimum_translation <= 0
        or not 0 < maximum_direction_error < 180
        or not 0 < maximum_rotation_error < 180
    ):
        raise ValueError("relative-motion evaluation thresholds are invalid")
    truth = relative_motion_truth(first, second)
    translated = truth["distance"] >= minimum_translation
    supported = bool(prediction["supported"])
    result = {
        "translation_distance_3d": truth["distance"],
        "translated": translated,
        "supported": supported,
        "direction_error_degrees": None,
        "rotation_error_degrees": None,
    }
    if not supported:
        result.update(
            correct=not translated,
            status="unsupported_motion" if translated else "control_rejected",
        )
        return result
    rotation = np.asarray(prediction["relative_rotation"], dtype=float)
    translation = np.asarray(prediction["unit_translation"], dtype=float)
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
        or not np.isclose(np.linalg.norm(translation), 1, atol=1e-5)
    ):
        raise ValueError(
            "supported motion must contain a proper rotation and unit translation"
        )
    rotation_error = math.degrees(
        math.acos(
            float(
                np.clip(
                    (np.trace(rotation @ truth["relative_rotation"].T) - 1) / 2, -1, 1
                )
            )
        )
    )
    result["rotation_error_degrees"] = rotation_error
    if not translated:
        result.update(correct=False, status="false_translation")
        return result
    direction_error = math.degrees(
        math.acos(float(np.clip(translation @ truth["unit_translation"], -1, 1)))
    )
    correct = (
        direction_error <= maximum_direction_error
        and rotation_error <= maximum_rotation_error
    )
    result.update(
        direction_error_degrees=direction_error,
        correct=correct,
        status="correct_supported_motion" if correct else "wrong_supported_motion",
    )
    return result
