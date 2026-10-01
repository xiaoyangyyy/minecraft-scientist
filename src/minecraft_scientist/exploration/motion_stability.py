"""Public-pixel sensitivity checks for ambiguous two-view motion estimates.

These checks test repeatability, not truth or calibrated confidence. A consistent
false correspondence field can still pass. No candidate is navigation evidence
until independently validated on disjoint trajectories.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .dense_correspondence import DenseCorrespondences
from .two_view_motion import (
    TwoViewMotionConfig,
    TwoViewMotionEvidence,
    estimate_two_view_motion,
)


class MotionStabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trials: int = Field(default=16, ge=4, le=64)
    retained_fraction: FiniteFloat = Field(default=0.8, gt=0.5, lt=1)
    reference_pixel_noise: FiniteFloat = Field(default=0.25, ge=0, le=2)
    relative_focal_noise: FiniteFloat = Field(default=0.01, ge=0, le=0.1)
    minimum_supported_fraction: FiniteFloat = Field(default=0.8, gt=0, le=1)
    maximum_direction_deviation: FiniteFloat = Field(default=20, gt=0, lt=180)
    maximum_rotation_deviation: FiniteFloat = Field(default=3, gt=0, lt=180)


class MotionSensitivityTrial(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    supported: bool
    reason: str
    direction_deviation_degrees: FiniteFloat | None = Field(default=None, ge=0, le=180)
    rotation_deviation_degrees: FiniteFloat | None = Field(default=None, ge=0, le=180)

    @model_validator(mode="after")
    def _complete(self) -> MotionSensitivityTrial:
        present = (
            self.direction_deviation_degrees is not None,
            self.rotation_deviation_degrees is not None,
        )
        if any(value != self.supported for value in present):
            raise ValueError("trial deviations must be present exactly when supported")
        return self


class StableMotionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    supported: bool
    reason: str
    candidate: TwoViewMotionEvidence
    trials: tuple[MotionSensitivityTrial, ...] = ()
    supported_fraction: FiniteFloat = Field(default=0, ge=0, le=1)
    direction_deviation_95_degrees: FiniteFloat | None = Field(
        default=None, ge=0, le=180
    )
    rotation_deviation_95_degrees: FiniteFloat | None = Field(
        default=None, ge=0, le=180
    )
    relative_rotation: (
        tuple[tuple[FiniteFloat, FiniteFloat, FiniteFloat], ...] | None
    ) = None
    unit_translation: tuple[FiniteFloat, FiniteFloat, FiniteFloat] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> StableMotionEvidence:
        if self.supported:
            if (
                not self.candidate.supported
                or not self.trials
                or self.relative_rotation != self.candidate.relative_rotation
                or self.unit_translation != self.candidate.unit_translation
                or self.unit_translation is None
                or self.relative_rotation is None
            ):
                raise ValueError(
                    "supported stability evidence requires its candidate pose"
                )
            _pose(self.candidate)
        elif self.relative_rotation is not None or self.unit_translation is not None:
            raise ValueError(
                "rejected stability evidence cannot expose an executable pose"
            )
        expected = (
            sum(trial.supported for trial in self.trials) / len(self.trials)
            if self.trials
            else 0
        )
        if not math.isclose(self.supported_fraction, expected, abs_tol=1e-12):
            raise ValueError("supported fraction disagrees with recorded trials")
        return self


def _pose(evidence: TwoViewMotionEvidence) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(evidence.relative_rotation, dtype=float)
    translation = np.asarray(evidence.unit_translation, dtype=float)
    if (
        not evidence.supported
        or rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
        or not np.isclose(np.linalg.norm(translation), 1, atol=1e-5)
    ):
        raise ValueError(
            "motion candidate must contain a proper rotation and unit translation"
        )
    return rotation, translation


def assess_motion_stability(
    correspondences: DenseCorrespondences,
    camera: np.ndarray,
    *,
    config: MotionStabilityConfig | None = None,
    motion_config: TwoViewMotionConfig | None = None,
    seed: int = 0,
) -> StableMotionEvidence:
    """Re-fit deterministic point subsets under bounded pixel/intrinsic noise.

    The original pose is retained only when the fits remain locally stable. The
    returned fraction is a sensitivity statistic, not probability of correctness.
    Subsampling never reads action kinds or privileged motion labels.
    """
    config = config or MotionStabilityConfig()
    motion_config = motion_config or TwoViewMotionConfig()
    camera = np.asarray(camera, dtype=float).copy()
    candidate = estimate_two_view_motion(
        correspondences, camera, config=motion_config, seed=seed
    )
    if not candidate.supported:
        return StableMotionEvidence(
            supported=False, reason="base motion unsupported", candidate=candidate
        )
    rotation, translation = _pose(candidate)
    rng = np.random.default_rng(seed)
    points0, points1 = (
        np.asarray(correspondences.points0),
        np.asarray(correspondences.points1),
    )
    sample_size = max(
        motion_config.minimum_matches,
        math.ceil(len(points0) * config.retained_fraction),
    )
    scale = correspondences.image_shape0[1] / 224
    trials = []
    for index in range(config.trials):
        indices = np.sort(rng.choice(len(points0), size=sample_size, replace=False))
        first, second = [
            points[indices]
            + rng.uniform(
                -config.reference_pixel_noise * scale,
                config.reference_pixel_noise * scale,
                (sample_size, 2),
            )
            for points in (points0, points1)
        ]
        valid = np.ones(sample_size, dtype=bool)
        for points, (height, width) in (
            (first, correspondences.image_shape0),
            (second, correspondences.image_shape1),
        ):
            valid &= (
                (points >= 0).all(axis=1)
                & (points[:, 0] < width)
                & (points[:, 1] < height)
            )
        focal_factor = 1 + rng.uniform(
            -config.relative_focal_noise, config.relative_focal_noise
        )
        perturbed_camera = camera.copy()
        perturbed_camera[0, 0] *= focal_factor
        perturbed_camera[1, 1] *= focal_factor
        perturbed = DenseCorrespondences(
            points0=tuple(map(tuple, first[valid])),
            points1=tuple(map(tuple, second[valid])),
            confidence=tuple(np.asarray(correspondences.confidence)[indices][valid]),
            image_shape0=correspondences.image_shape0,
            image_shape1=correspondences.image_shape1,
            raw_match_count=int(valid.sum()),
        )
        fit = estimate_two_view_motion(
            perturbed, perturbed_camera, config=motion_config, seed=seed + index + 1
        )
        if not fit.supported:
            trials.append(MotionSensitivityTrial(supported=False, reason=fit.reason))
            continue
        trial_rotation, trial_translation = _pose(fit)
        direction = math.degrees(
            math.acos(float(np.clip(translation @ trial_translation, -1, 1)))
        )
        turn = math.degrees(
            math.acos(
                float(np.clip((np.trace(rotation @ trial_rotation.T) - 1) / 2, -1, 1))
            )
        )
        trials.append(
            MotionSensitivityTrial(
                supported=True,
                reason=fit.reason,
                direction_deviation_degrees=direction,
                rotation_deviation_degrees=turn,
            )
        )
    positives = [trial for trial in trials if trial.supported]
    fraction = len(positives) / config.trials
    direction95 = (
        float(
            np.quantile(
                [trial.direction_deviation_degrees for trial in positives], 0.95
            )
        )
        if positives
        else None
    )
    rotation95 = (
        float(
            np.quantile([trial.rotation_deviation_degrees for trial in positives], 0.95)
        )
        if positives
        else None
    )
    supported = (
        fraction >= config.minimum_supported_fraction
        and direction95 is not None
        and direction95 <= config.maximum_direction_deviation
        and rotation95 is not None
        and rotation95 <= config.maximum_rotation_deviation
    )
    return StableMotionEvidence(
        supported=supported,
        reason="locally stable motion; accuracy unvalidated"
        if supported
        else "motion sensitive to correspondence or camera perturbations",
        candidate=candidate,
        trials=tuple(trials),
        supported_fraction=fraction,
        direction_deviation_95_degrees=direction95,
        rotation_deviation_95_degrees=rotation95,
        relative_rotation=candidate.relative_rotation if supported else None,
        unit_translation=candidate.unit_translation if supported else None,
    )
