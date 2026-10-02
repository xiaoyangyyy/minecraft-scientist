"""Training-only direction-constrained map search with a frozen-map sanity replay.

This is an incomplete observability diagnostic, not a pose estimator. Alternative
rotations minimize endpoint Sampson residuals at fixed camera-center directions.
All original identities and withheld intermediate pixels must then remain valid.
No absence of a witness certifies uniqueness or promotes navigation knowledge.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .keyframe_motion import KeyframeAttempt
from .motion_stability import _pose
from .multiview_motion import IntermediateFrameCheck, MultiViewMotionConfig
from .pose_competition import _epipolar_errors
from .visual_geometry import _cells, _opencv
from .visual_odometry import VisualOdometryConfig, triangulate_landmarks


class DirectionSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_offsets_degrees: tuple[FiniteFloat, ...] = (25, 45)
    direction_separation_degrees: FiniteFloat = Field(default=20, gt=0, lt=90)
    maximum_rotation_adjustment_degrees: FiniteFloat = Field(default=3, gt=0, le=3)
    iterations: int = Field(default=16, ge=1, le=32, strict=True)
    geometry: MultiViewMotionConfig = MultiViewMotionConfig()

    @model_validator(mode="after")
    def _bounded(self):
        if (
            not self.direction_offsets_degrees
            or len(self.direction_offsets_degrees) > 3
            or tuple(sorted(set(self.direction_offsets_degrees)))
            != self.direction_offsets_degrees
            or not all(
                self.direction_separation_degrees < value <= 60
                for value in self.direction_offsets_degrees
            )
        ):
            raise ValueError("direction grid must be unique, sorted and witness-sized")
        return self

    def grid(self):
        # Eight fixed directions in a deterministic tangent basis for each cone.
        return tuple(
            (angle, azimuth)
            for angle in self.direction_offsets_degrees
            for azimuth in range(0, 360, 45)
        )


class MapValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    supported: bool
    reason: str
    training_tracks: int = Field(ge=0, strict=True)
    validation_tracks: int = Field(ge=0, strict=True)
    endpoint_partitions_valid: bool = False
    frame_checks: tuple[IntermediateFrameCheck, ...] = ()

    @model_validator(mode="after")
    def _consistent(self):
        expected = bool(
            self.endpoint_partitions_valid
            and self.frame_checks
            and all(item.supported for item in self.frame_checks)
        )
        if expected != self.supported:
            raise ValueError("map support requires all endpoints and withheld frames")
        return self


class DirectionalMapTrial(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_direction_offset_degrees: FiniteFloat = Field(gt=0, le=60)
    tangent_azimuth_degrees: int = Field(ge=0, lt=360, strict=True)
    direction_separation_degrees: FiniteFloat = Field(ge=0, le=180)
    rotation_adjustment_degrees: FiniteFloat = Field(ge=0, le=180)
    training_epipolar_error_95_pixels: FiniteFloat = Field(ge=0)
    training_objective_initial: FiniteFloat = Field(ge=0)
    training_objective_final: FiniteFloat = Field(ge=0)
    accepted_steps: int = Field(ge=0, le=32, strict=True)
    plausible: bool
    reason: str
    validation: MapValidation | None = None

    @model_validator(mode="after")
    def _complete(self):
        if self.plausible and (
            self.validation is None or not self.validation.supported
        ):
            raise ValueError("a witness needs original withheld-map validation")
        if self.training_objective_final > self.training_objective_initial + 1e-9:
            raise ValueError("training search cannot accept a worsening step")
        return self


class DirectionSearchEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: DirectionSearchConfig
    ambiguity_detected: bool
    reason: str
    training_track_ids: tuple[int, ...] = ()
    validation_track_ids: tuple[int, ...] = ()
    reference: MapValidation | None = None
    reference_matches_original_checks: bool = False
    trials: tuple[DirectionalMapTrial, ...] = ()

    @model_validator(mode="after")
    def _consistent(self):
        train, valid = self.training_track_ids, self.validation_track_ids
        if (
            len(set(train)) != len(train)
            or len(set(valid)) != len(valid)
            or set(train) & set(valid)
        ):
            raise ValueError("direction search preserves disjoint original identities")
        reference_valid = bool(
            self.reference is not None
            and self.reference.supported
            and self.reference_matches_original_checks
        )
        if self.reference_matches_original_checks and (
            self.reference is None or not self.reference.supported
        ):
            raise ValueError("reference replay cannot match absent/failed evidence")
        expected = reference_valid and any(
            item.plausible
            and item.direction_separation_degrees
            >= self.config.direction_separation_degrees
            for item in self.trials
        )
        if expected != self.ambiguity_detected:
            raise ValueError("ambiguity requires a replayed reference and a witness")
        if self.reference is not None and (
            not train
            or not valid
            or self.reference.training_tracks != len(train)
            or self.reference.validation_tracks != len(valid)
        ):
            raise ValueError("reference partition counts must preserve original IDs")
        if reference_valid:
            if (
                tuple(
                    (
                        item.requested_direction_offset_degrees,
                        item.tangent_azimuth_degrees,
                    )
                    for item in self.trials
                )
                != self.config.grid()
            ):
                raise ValueError("search must retain the complete fixed direction grid")
        elif self.trials:
            raise ValueError("invalid reference cannot support a direction search")
        for item in self.trials:
            if item.plausible and (
                item.rotation_adjustment_degrees
                > self.config.maximum_rotation_adjustment_degrees + 1e-8
                or abs(
                    item.direction_separation_degrees
                    - item.requested_direction_offset_degrees
                )
                > 1e-6
            ):
                raise ValueError(
                    "witness must satisfy its fixed direction/rotation bounds"
                )
        return self


def _signed_sampson(first, second, camera, rotation, center):
    translation = -rotation @ center
    x, y, z = translation
    skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    inverse = np.linalg.inv(camera)
    fundamental = inverse.T @ skew @ rotation @ inverse
    line1, line0 = first @ fundamental.T, second @ fundamental
    denominator = np.sum(line1[:, :2] ** 2 + line0[:, :2] ** 2, axis=1)
    return np.sum(second * line1, axis=1) / np.sqrt(np.maximum(denominator, 1e-15))


def _huber(residuals, threshold):
    absolute = np.abs(residuals)
    return float(
        np.mean(
            np.where(
                absolute <= threshold,
                residuals**2 / 2,
                threshold * (absolute - threshold / 2),
            )
        )
    )


def _fit_rotation(first, second, camera, original, center, config, threshold, cv2):
    """Bounded finite-difference LM; only training endpoint rays enter this fit."""
    parameters, damping, accepted = np.zeros(3), 1e-3, 0
    bound = np.radians(config.maximum_rotation_adjustment_degrees)

    def residual(vector):
        return _signed_sampson(
            first, second, camera, cv2.Rodrigues(vector)[0] @ original, center
        )

    errors = residual(parameters)
    initial = objective = _huber(errors, threshold)
    step_size = 1e-6
    for _ in range(config.iterations):
        jacobian = np.column_stack(
            [
                (
                    residual(parameters + np.eye(3)[axis] * step_size)
                    - residual(parameters - np.eye(3)[axis] * step_size)
                )
                / (2 * step_size)
                for axis in range(3)
            ]
        )
        weights = np.minimum(1.0, threshold / np.maximum(np.abs(errors), 1e-12))
        normal = jacobian.T @ (weights[:, None] * jacobian)
        gradient = jacobian.T @ (weights * errors)
        update = np.linalg.solve(
            normal + damping * np.diag(np.maximum(np.diag(normal), 1)), -gradient
        )
        proposed = parameters + update
        length = np.linalg.norm(proposed)
        if length > bound:
            proposed *= bound / length
        changed = residual(proposed)
        score = _huber(changed, threshold)
        if score < objective:
            parameters, errors, objective = proposed, changed, score
            accepted += 1
            damping = max(1e-9, damping / 3)
        else:
            damping *= 10
        if np.linalg.norm(update) < 1e-9:
            break
    rotation = cv2.Rodrigues(parameters)[0] @ original
    return (
        rotation,
        initial,
        objective,
        accepted,
        float(np.degrees(np.linalg.norm(parameters))),
    )


def _map_validation(candidate, camera, rotation, translation, settings, seed, cv2):
    tracks = candidate.tracks
    indices = {identity: index for index, identity in enumerate(tracks.track_ids)}
    training = np.array([indices[item] for item in candidate.motion.training_track_ids])
    validation = np.array(
        [indices[item] for item in candidate.motion.validation_track_ids]
    )
    # Original ordering is essential to reproduce PnP's deterministic sample draws.
    pixels = np.asarray(tracks.pixels, dtype=float)
    scale = tracks.image_shape[1] / 224
    points, valid = triangulate_landmarks(
        pixels[0],
        pixels[-1],
        np.eye(3),
        np.zeros(3),
        rotation,
        translation,
        camera,
        config=VisualOdometryConfig(
            minimum_parallax_degrees=settings.two_view.minimum_parallax_degrees,
            reprojection_pixels=settings.two_view.reference_reprojection_pixels * scale,
        ),
        cv2=cv2,
    )
    base = {"training_tracks": len(training), "validation_tracks": len(validation)}
    if not valid[np.concatenate((training, validation))].all():
        return MapValidation(
            supported=False, reason="map loses original endpoint identities", **base
        )
    base["endpoint_partitions_valid"] = True
    checks = []
    for slot in range(1, tracks.frame_count - 1):
        cv2.setRNGSeed(seed + slot)
        success, vector, shift, inliers = cv2.solvePnPRansac(
            points[training],
            pixels[slot, training],
            camera,
            None,
            iterationsCount=200,
            reprojectionError=settings.reference_pnp_pixels * scale,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        count = 0 if inliers is None else len(inliers)
        item = {
            "frame_slot": slot,
            "training_inliers": count,
            "validation_tracks": len(validation),
        }
        if not success or count < settings.minimum_training_inliers:
            checks.append(
                IntermediateFrameCheck(
                    supported=False, reason="map PnP lacks fitting support", **item
                )
            )
            continue
        selected = training[inliers.ravel()]
        vector, shift = cv2.solvePnPRefineLM(
            points[selected], pixels[slot, selected], camera, None, vector, shift
        )
        if not np.isfinite(vector).all() or not np.isfinite(shift).all():
            checks.append(
                IntermediateFrameCheck(
                    supported=False, reason="map PnP is nonfinite", **item
                )
            )
            continue
        transformed = points[validation] @ cv2.Rodrigues(vector)[0].T + shift.ravel()
        positive = transformed[:, 2] > 0
        projected = transformed @ camera.T
        errors = np.linalg.norm(
            projected[:, :2] / np.maximum(projected[:, 2:], 1e-9)
            - pixels[slot, validation],
            axis=1,
        )
        positive &= np.isfinite(errors)
        passing = positive & (errors <= settings.reference_validation_pixels * scale)
        fraction = float(passing.mean())
        error95 = float(np.quantile(errors[positive], 0.95)) if positive.any() else None
        cells = _cells(pixels[slot, validation[passing]], *tracks.image_shape)
        supported = bool(
            positive.all()
            and fraction >= settings.minimum_validation_acceptance
            and error95 is not None
            and error95 <= settings.reference_validation_pixels * scale
            and cells >= settings.minimum_validation_cells
        )
        checks.append(
            IntermediateFrameCheck(
                supported=supported,
                reason="map withheld pixels consistent"
                if supported
                else "map withheld pixels inconsistent",
                validation_acceptance=fraction,
                validation_error_95_pixels=error95,
                validation_cells=cells,
                **item,
            )
        )
    base["frame_checks"] = tuple(checks)
    supported = bool(checks and all(item.supported for item in checks))
    return MapValidation(
        supported=supported,
        reason="map reproduced" if supported else "map withheld validation failed",
        **base,
    )


def _matches_reference(reference, original):
    if not reference.supported or len(reference.frame_checks) != len(
        original.frame_checks
    ):
        return False
    for replayed, recorded in zip(
        reference.frame_checks, original.frame_checks, strict=True
    ):
        for name, value in replayed.model_dump().items():
            if name == "reason":
                continue
            expected = getattr(recorded, name)
            if isinstance(value, float):
                if expected is None or not math.isclose(
                    value, expected, rel_tol=1e-8, abs_tol=1e-8
                ):
                    return False
            elif value != expected:
                return False
    return True


def inspect_direction_search(
    candidate: KeyframeAttempt | None, camera, *, seed=0, config=None
):
    config = config or DirectionSearchConfig()
    if type(seed) is not int or not 0 <= seed < 2**31 - 1024:
        raise ValueError("direction search seed must reserve trial/frame seeds")
    camera = np.asarray(camera, dtype=float)
    if (
        camera.shape != (3, 3)
        or not np.isfinite(camera).all()
        or camera[0, 0] <= 0
        or camera[1, 1] <= 0
        or not np.allclose(camera[2], (0, 0, 1))
        or not np.isclose(camera[0, 1], 0)
        or not np.isclose(camera[1, 0], 0)
    ):
        raise ValueError(
            "direction search requires finite zero-skew pinhole intrinsics"
        )
    if candidate is None or not candidate.motion.supported:
        return DirectionSearchEvidence(
            config=config,
            ambiguity_detected=False,
            reason="no supported frozen local candidate",
        )
    training, validation = (
        candidate.motion.training_track_ids,
        candidate.motion.validation_track_ids,
    )
    if (
        len(training) < config.geometry.minimum_training_inliers
        or len(validation) < config.geometry.minimum_validation_tracks
        or not set(training + validation) <= set(candidate.tracks.track_ids)
        or candidate.tracks.frame_count != candidate.end_offset + 1
    ):
        raise ValueError("direction search requires complete original partitions")
    cv2 = _opencv()
    rotation, translation = _pose(candidate.motion.candidate)
    reference = _map_validation(
        candidate, camera, rotation, translation, config.geometry, seed, cv2
    )
    matches = _matches_reference(reference, candidate.motion)
    base = {
        "config": config,
        "training_track_ids": training,
        "validation_track_ids": validation,
        "reference": reference,
        "reference_matches_original_checks": matches,
    }
    if not matches:
        return DirectionSearchEvidence(
            ambiguity_detected=False,
            reason="original map replay failed; search uninformative",
            **base,
        )
    center = -rotation.T @ translation
    tangent = np.eye(3)[int(np.argmin(np.abs(center)))]
    tangent -= center * (tangent @ center)
    tangent /= np.linalg.norm(tangent)
    perpendicular = np.cross(center, tangent)
    slots = [candidate.tracks.track_ids.index(identity) for identity in training]
    pixels = np.asarray(candidate.tracks.pixels, dtype=float)
    first, second = [
        np.column_stack((frame[slots], np.ones(len(slots))))
        for frame in (pixels[0], pixels[-1])
    ]
    threshold = (
        config.geometry.two_view.reference_ransac_pixels
        * candidate.tracks.image_shape[1]
        / 224
    )
    trials = []
    for index, (angle, azimuth) in enumerate(config.grid()):
        theta, phi = np.radians((angle, azimuth))
        direction = math.cos(theta) * center + math.sin(theta) * (
            math.cos(phi) * tangent + math.sin(phi) * perpendicular
        )
        changed, initial, final, steps, adjustment = _fit_rotation(
            first, second, camera, rotation, direction, config, threshold, cv2
        )
        shift = -changed @ direction
        error95 = float(
            np.quantile(_epipolar_errors(first, second, camera, changed, shift), 0.95)
        )
        validated = None
        if error95 <= threshold:
            validated = _map_validation(
                candidate,
                camera,
                changed,
                shift,
                config.geometry,
                seed + (index + 1) * 32,
                cv2,
            )
        plausible = bool(validated is not None and validated.supported)
        trials.append(
            DirectionalMapTrial(
                requested_direction_offset_degrees=angle,
                tangent_azimuth_degrees=azimuth,
                direction_separation_degrees=math.degrees(
                    math.acos(float(np.clip(center @ direction, -1, 1)))
                ),
                rotation_adjustment_degrees=adjustment,
                training_epipolar_error_95_pixels=error95,
                training_objective_initial=initial,
                training_objective_final=final,
                accepted_steps=steps,
                plausible=plausible,
                reason="different map explains original withheld pixels"
                if plausible
                else "direction-constrained map lacks original support",
                validation=validated,
            )
        )
    ambiguous = any(item.plausible for item in trials)
    return DirectionSearchEvidence(
        ambiguity_detected=ambiguous,
        reason="different motion explains the same withheld pixels"
        if ambiguous
        else "no witness in incomplete fitted family; uniqueness unproved",
        trials=tuple(trials),
        **base,
    )
