"""Intermediate-frame validation of an RGB-only endpoint motion candidate.

The two-view map is fitted without intermediate observations. Intermediate PnP
uses a fixed training-point subset and scores distinct withheld points. This is
internal geometric validation, not independent ground-truth navigation evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .dense_correspondence import DenseCorrespondences
from .motion_stability import (
    MotionSensitivityTrial,
    MotionStabilityConfig,
    StableMotionEvidence,
    _pose,
)
from .temporal_tracks import TemporalTracks
from .two_view_motion import (
    TwoViewMotionConfig,
    TwoViewMotionEvidence,
    estimate_two_view_motion,
)
from .visual_geometry import _cells, _opencv
from .visual_odometry import VisualOdometryConfig, triangulate_landmarks


class MultiViewMotionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_tracks: int = Field(default=32, ge=24)
    minimum_training_inliers: int = Field(default=12, ge=8)
    minimum_validation_tracks: int = Field(default=8, ge=6)
    validation_fraction: FiniteFloat = Field(default=0.25, gt=0, lt=0.5)
    minimum_validation_acceptance: FiniteFloat = Field(default=0.8, gt=0, le=1)
    minimum_validation_cells: int = Field(default=2, ge=2, le=16)
    reference_pnp_pixels: FiniteFloat = Field(default=1, gt=0)
    reference_validation_pixels: FiniteFloat = Field(default=1.5, gt=0)
    two_view: TwoViewMotionConfig = TwoViewMotionConfig(
        reference_ransac_pixels=0.5, reference_reprojection_pixels=1
    )
    stability: MotionStabilityConfig = MotionStabilityConfig()

    @model_validator(mode="after")
    def _budget(self) -> MultiViewMotionConfig:
        if (
            self.minimum_tracks
            < self.minimum_training_inliers + self.minimum_validation_tracks
        ):
            raise ValueError("track budget cannot support independent PnP validation")
        return self


class IntermediateFrameCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_slot: int = Field(ge=1)
    supported: bool
    reason: str
    training_inliers: int = Field(default=0, ge=0)
    validation_tracks: int = Field(default=0, ge=0)
    validation_acceptance: FiniteFloat = Field(default=0, ge=0, le=1)
    validation_error_95_pixels: FiniteFloat | None = Field(default=None, ge=0)
    validation_cells: int = Field(default=0, ge=0)


class MultiViewMotionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    supported: bool
    reason: str
    candidate: TwoViewMotionEvidence
    triangulated_tracks: int = Field(default=0, ge=0)
    training_track_ids: tuple[int, ...] = ()
    validation_track_ids: tuple[int, ...] = ()
    frame_checks: tuple[IntermediateFrameCheck, ...] = ()
    sensitivity: StableMotionEvidence | None = None
    relative_rotation: (
        tuple[tuple[FiniteFloat, FiniteFloat, FiniteFloat], ...] | None
    ) = None
    unit_translation: tuple[FiniteFloat, FiniteFloat, FiniteFloat] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> MultiViewMotionEvidence:
        training, validation = (
            set(self.training_track_ids),
            set(self.validation_track_ids),
        )
        if (
            len(training) != len(self.training_track_ids)
            or len(validation) != len(self.validation_track_ids)
            or training & validation
        ):
            raise ValueError(
                "intermediate fitting and validation track identities must be disjoint"
            )
        if self.supported:
            if (
                not self.candidate.supported
                or not self.frame_checks
                or not all(item.supported for item in self.frame_checks)
                or self.sensitivity is None
                or not self.sensitivity.supported
                or self.relative_rotation != self.candidate.relative_rotation
                or self.unit_translation != self.candidate.unit_translation
            ):
                raise ValueError(
                    "supported multi-view motion requires all geometric validations"
                )
            _pose(self.candidate)
        elif self.relative_rotation is not None or self.unit_translation is not None:
            raise ValueError(
                "rejected multi-view evidence cannot expose an executable pose"
            )
        return self


def estimate_temporal_endpoint(
    matches: DenseCorrespondences,
    camera: np.ndarray,
    *,
    config: TwoViewMotionConfig,
    seed: int = 0,
) -> TwoViewMotionEvidence:
    """Refit a supported minimal-set model using its epipolar inlier field.

    This does not rescue rejected controls or low-parallax pairs. Three calibrated
    eight-point fits reduce minimal-sample variance; cheirality, reprojection and
    coverage are checked again. Internal consistency still does not establish truth.
    """
    candidate = estimate_two_view_motion(matches, camera, config=config, seed=seed)
    if not candidate.supported:
        return candidate
    camera = np.asarray(camera, dtype=float)
    first, second = np.asarray(matches.points0), np.asarray(matches.points1)
    homogeneous0 = np.column_stack((first, np.ones(len(first))))
    homogeneous1 = np.column_stack((second, np.ones(len(second))))
    inverse = np.linalg.inv(camera)
    normalized0, normalized1 = homogeneous0 @ inverse.T, homogeneous1 @ inverse.T
    x0, y0 = normalized0[:, :2].T
    x1, y1 = normalized1[:, :2].T
    design = np.column_stack(
        (x1 * x0, x1 * y0, x1, y1 * x0, y1 * y0, y1, x0, y0, np.ones(len(first)))
    )
    rotation, translation = _pose(candidate)
    tx, ty, tz = translation
    essential = np.array(((0, -tz, ty), (tz, 0, -tx), (-ty, tx, 0))) @ rotation
    threshold = config.reference_ransac_pixels * matches.image_shape0[1] / 224

    def inliers(matrix):
        fundamental = inverse.T @ matrix @ inverse
        line1, line0 = homogeneous0 @ fundamental.T, homogeneous1 @ fundamental
        numerator = np.sum(homogeneous1 * line1, axis=1) ** 2
        denominator = np.sum(line1[:, :2] ** 2 + line0[:, :2] ** 2, axis=1)
        errors = np.sqrt(numerator / np.maximum(denominator, 1e-15))
        return np.isfinite(errors) & (errors <= threshold)

    selected = inliers(essential)
    for _ in range(3):
        if int(selected.sum()) < config.minimum_matches:
            return TwoViewMotionEvidence(
                supported=False,
                reason="refined endpoint lacks epipolar support",
                matches=len(first),
            )
        _, _, basis = np.linalg.svd(design[selected], full_matrices=False)
        fitted = basis[-1].reshape(3, 3)
        left, singular, right = np.linalg.svd(fitted)
        average = (singular[0] + singular[1]) / 2
        essential = left @ np.diag((average, average, 0)) @ right
        selected = inliers(essential)
    cv2 = _opencv()
    count, rotation, translation, positive = cv2.recoverPose(
        essential, first, second, camera, mask=selected.astype(np.uint8).reshape(-1, 1)
    )
    points, valid = triangulate_landmarks(
        first,
        second,
        np.eye(3),
        np.zeros(3),
        rotation,
        translation.ravel(),
        camera,
        config=VisualOdometryConfig(
            minimum_parallax_degrees=config.minimum_parallax_degrees,
            reprojection_pixels=config.reference_reprojection_pixels
            * matches.image_shape0[1]
            / 224,
        ),
        cv2=cv2,
    )
    valid &= positive.ravel() > 0
    cells = min(
        _cells(first[valid], *matches.image_shape0),
        _cells(second[valid], *matches.image_shape1),
    )
    values = candidate.model_dump()
    values.update(
        positive_inliers=int(count),
        triangulated_inliers=int(valid.sum()),
        spatial_cells=cells,
    )
    if (
        count < config.minimum_positive_inliers
        or int(valid.sum()) < config.minimum_positive_inliers
        or cells < config.minimum_spatial_cells
    ):
        values.update(
            supported=False,
            reason="refined endpoint lacks positive-depth spatial support",
            relative_rotation=None,
            unit_translation=None,
            median_parallax_degrees=None,
        )
    else:
        center = -rotation.T @ translation.ravel()
        first_rays = points[valid] / np.linalg.norm(
            points[valid], axis=1, keepdims=True
        )
        second_rays = points[valid] - center
        second_rays /= np.linalg.norm(second_rays, axis=1, keepdims=True)
        angles = np.degrees(
            np.arccos(np.clip(np.sum(first_rays * second_rays, axis=1), -1, 1))
        )
        values.update(
            reason="refitted endpoint motion; accuracy unvalidated",
            relative_rotation=tuple(map(tuple, rotation)),
            unit_translation=tuple(translation.ravel()),
            median_parallax_degrees=float(np.median(angles)),
        )
    return TwoViewMotionEvidence.model_validate(values)


def _endpoint_sensitivity(
    matches: DenseCorrespondences,
    camera: np.ndarray,
    *,
    config: MotionStabilityConfig,
    motion_config: TwoViewMotionConfig,
    seed: int,
    endpoint_estimator: Callable | None = None,
) -> StableMotionEvidence:
    estimator = endpoint_estimator or estimate_temporal_endpoint
    candidate = estimator(matches, camera, config=motion_config, seed=seed)
    if not candidate.supported:
        return StableMotionEvidence(
            supported=False, reason="refitted endpoint unsupported", candidate=candidate
        )
    rotation, translation = _pose(candidate)
    rng = np.random.default_rng(seed)
    first, second = np.asarray(matches.points0), np.asarray(matches.points1)
    size = max(
        motion_config.minimum_matches, math.ceil(len(first) * config.retained_fraction)
    )
    scale = matches.image_shape0[1] / 224
    trials = []
    for index in range(config.trials):
        selected = np.sort(rng.choice(len(first), size=size, replace=False))
        points = [
            array[selected]
            + rng.uniform(
                -config.reference_pixel_noise * scale,
                config.reference_pixel_noise * scale,
                (size, 2),
            )
            for array in (first, second)
        ]
        valid = np.ones(size, dtype=bool)
        for array, (height, width) in zip(
            points, (matches.image_shape0, matches.image_shape1), strict=True
        ):
            valid &= (
                (array >= 0).all(axis=1)
                & (array[:, 0] < width)
                & (array[:, 1] < height)
            )
        perturbation = 1 + rng.uniform(
            -config.relative_focal_noise, config.relative_focal_noise
        )
        perturbed_camera = camera.copy()
        perturbed_camera[0, 0] *= perturbation
        perturbed_camera[1, 1] *= perturbation
        perturbed = DenseCorrespondences(
            points0=tuple(map(tuple, points[0][valid])),
            points1=tuple(map(tuple, points[1][valid])),
            confidence=tuple(np.asarray(matches.confidence)[selected][valid]),
            image_shape0=matches.image_shape0,
            image_shape1=matches.image_shape1,
            raw_match_count=int(valid.sum()),
        )
        fit = estimator(
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
        reason="refitted endpoint locally stable; accuracy unvalidated"
        if supported
        else "refitted endpoint sensitive to pixel or camera perturbations",
        candidate=candidate,
        trials=tuple(trials),
        supported_fraction=fraction,
        direction_deviation_95_degrees=direction95,
        rotation_deviation_95_degrees=rotation95,
        relative_rotation=candidate.relative_rotation if supported else None,
        unit_translation=candidate.unit_translation if supported else None,
    )


def estimate_multiview_motion(
    tracks: TemporalTracks,
    camera: np.ndarray,
    *,
    config: MultiViewMotionConfig | None = None,
    seed: int = 0,
    endpoint_estimator: Callable | None = None,
) -> MultiViewMotionEvidence:
    config = config or MultiViewMotionConfig()
    camera = np.asarray(camera, dtype=float)
    matches = tracks.endpoint_correspondences()
    estimator = endpoint_estimator or estimate_temporal_endpoint
    candidate = estimator(matches, camera, config=config.two_view, seed=seed)
    base = {"candidate": candidate}
    if not candidate.supported:
        return MultiViewMotionEvidence(
            supported=False, reason="endpoint geometry unsupported", **base
        )
    if tracks.frame_count < 3:
        return MultiViewMotionEvidence(
            supported=False,
            reason="no intermediate image for independent geometry validation",
            **base,
        )
    cv2 = _opencv()
    rotation, translation = _pose(candidate)
    pixels = np.asarray(tracks.pixels, dtype=float)
    points, valid = triangulate_landmarks(
        pixels[0],
        pixels[-1],
        np.eye(3),
        np.zeros(3),
        rotation,
        translation,
        camera,
        config=VisualOdometryConfig(
            minimum_parallax_degrees=config.two_view.minimum_parallax_degrees,
            reprojection_pixels=config.two_view.reference_reprojection_pixels
            * tracks.image_shape[1]
            / 224,
        ),
        cv2=cv2,
    )
    indices = np.flatnonzero(valid)
    base["triangulated_tracks"] = len(indices)
    if len(indices) < config.minimum_tracks:
        return MultiViewMotionEvidence(
            supported=False,
            reason="insufficient common tracks for withheld-point validation",
            **base,
        )
    permutation = np.random.default_rng(seed).permutation(indices)
    validation_count = max(
        config.minimum_validation_tracks,
        math.ceil(len(indices) * config.validation_fraction),
    )
    validation_indices, training_indices = (
        np.sort(permutation[:validation_count]),
        np.sort(permutation[validation_count:]),
    )
    if len(training_indices) < config.minimum_training_inliers:
        return MultiViewMotionEvidence(
            supported=False,
            reason="insufficient fitting points after validation split",
            **base,
        )
    base["training_track_ids"] = tuple(
        tracks.track_ids[index] for index in training_indices
    )
    base["validation_track_ids"] = tuple(
        tracks.track_ids[index] for index in validation_indices
    )
    scale = tracks.image_shape[1] / 224
    checks = []
    for frame_slot in range(1, tracks.frame_count - 1):
        cv2.setRNGSeed(seed + frame_slot)
        success, vector, offset, inliers = cv2.solvePnPRansac(
            points[training_indices],
            pixels[frame_slot, training_indices],
            camera,
            None,
            iterationsCount=200,
            reprojectionError=config.reference_pnp_pixels * scale,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        count = 0 if inliers is None else len(inliers)
        if not success or count < config.minimum_training_inliers:
            checks.append(
                IntermediateFrameCheck(
                    frame_slot=frame_slot,
                    supported=False,
                    reason="intermediate PnP has insufficient fitting support",
                    training_inliers=count,
                    validation_tracks=validation_count,
                )
            )
            continue
        selected = training_indices[inliers.ravel()]
        vector, offset = cv2.solvePnPRefineLM(
            points[selected], pixels[frame_slot, selected], camera, None, vector, offset
        )
        if not np.isfinite(vector).all() or not np.isfinite(offset).all():
            checks.append(
                IntermediateFrameCheck(
                    frame_slot=frame_slot,
                    supported=False,
                    reason="intermediate PnP is nonfinite",
                    training_inliers=count,
                    validation_tracks=validation_count,
                )
            )
            continue
        predicted_camera = (
            points[validation_indices] @ cv2.Rodrigues(vector)[0].T + offset.ravel()
        )
        depth = predicted_camera[:, 2]
        projected = predicted_camera @ camera.T
        projected = projected[:, :2] / np.maximum(projected[:, 2:], 1e-9)
        errors = np.linalg.norm(
            projected - pixels[frame_slot, validation_indices], axis=1
        )
        positive = np.isfinite(errors) & (depth > 0)
        passing = positive & (errors <= config.reference_validation_pixels * scale)
        fraction = float(passing.mean())
        error95 = float(np.quantile(errors[positive], 0.95)) if positive.any() else None
        cells = _cells(
            pixels[frame_slot, validation_indices[passing]], *tracks.image_shape
        )
        supported = bool(
            positive.all()
            and fraction >= config.minimum_validation_acceptance
            and error95 is not None
            and error95 <= config.reference_validation_pixels * scale
            and cells >= config.minimum_validation_cells
        )
        checks.append(
            IntermediateFrameCheck(
                frame_slot=frame_slot,
                supported=supported,
                reason="withheld-point geometry consistent"
                if supported
                else "withheld-point geometry inconsistent",
                training_inliers=count,
                validation_tracks=validation_count,
                validation_acceptance=fraction,
                validation_error_95_pixels=error95,
                validation_cells=cells,
            )
        )
    base["frame_checks"] = tuple(checks)
    if not all(item.supported for item in checks):
        return MultiViewMotionEvidence(
            supported=False, reason="intermediate geometry validation failed", **base
        )
    sensitivity = _endpoint_sensitivity(
        matches,
        camera,
        config=config.stability,
        motion_config=config.two_view,
        seed=seed,
        endpoint_estimator=endpoint_estimator,
    )
    base["sensitivity"] = sensitivity
    if not sensitivity.supported:
        return MultiViewMotionEvidence(
            supported=False,
            reason="validated geometry remains locally unstable",
            **base,
        )
    return MultiViewMotionEvidence(
        supported=True,
        reason="multi-view consistent motion; external accuracy unvalidated",
        relative_rotation=candidate.relative_rotation,
        unit_translation=candidate.unit_translation,
        **base,
    )
