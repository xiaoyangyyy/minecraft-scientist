"""Search a bounded public-pixel family for competing maps, never replace a pose.

A witness disproves uniqueness inside the tested tolerance. No witness does not
prove uniqueness: the rotation grid and linear translation fit are incomplete.
Endpoint maps use original identities; intermediate PnP never fits withheld rays.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .keyframe_motion import KeyframeAttempt
from .motion_stability import _pose
from .multiview_motion import IntermediateFrameCheck, MultiViewMotionConfig
from .visual_geometry import _cells, _opencv
from .visual_odometry import VisualOdometryConfig, triangulate_landmarks


class PoseCompetitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rotation_offsets_degrees: tuple[FiniteFloat, ...] = (0.25, 0.5, 1.0)
    direction_separation_degrees: FiniteFloat = Field(default=20, gt=0, lt=180)
    geometry: MultiViewMotionConfig = MultiViewMotionConfig()

    @model_validator(mode="after")
    def _bounded(self):
        if (
            not self.rotation_offsets_degrees
            or len(self.rotation_offsets_degrees) > 4
            or tuple(sorted(set(self.rotation_offsets_degrees)))
            != self.rotation_offsets_degrees
            or not all(0 < value <= 3 for value in self.rotation_offsets_degrees)
        ):
            raise ValueError(
                "competition needs a unique increasing bounded rotation grid"
            )
        return self

    def grid(self):
        return ((0.0, 0.0, 0.0),) + tuple(
            tuple(sign * value if column == axis else 0.0 for column in range(3))
            for value in self.rotation_offsets_degrees
            for axis in range(3)
            for sign in (-1, 1)
        )


class PoseWitnessTrial(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rotation_offset_degrees: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    plausible: bool
    reason: str
    direction_separation_degrees: FiniteFloat | None = Field(default=None, ge=0, le=180)
    training_tracks: int = Field(default=0, ge=0, strict=True)
    validation_tracks: int = Field(default=0, ge=0, strict=True)
    training_epipolar_error_95_pixels: FiniteFloat | None = Field(default=None, ge=0)
    endpoint_partitions_valid: bool = False
    frame_checks: tuple[IntermediateFrameCheck, ...] = ()

    @model_validator(mode="after")
    def _plausible(self):
        if self.plausible and (
            self.direction_separation_degrees is None
            or not self.endpoint_partitions_valid
            or not self.frame_checks
            or not all(item.supported for item in self.frame_checks)
        ):
            raise ValueError(
                "plausible alternatives require endpoint and withheld-frame evidence"
            )
        return self


class PoseCompetitionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: PoseCompetitionConfig
    ambiguity_detected: bool
    reason: str
    training_track_ids: tuple[int, ...] = ()
    validation_track_ids: tuple[int, ...] = ()
    trials: tuple[PoseWitnessTrial, ...] = ()

    @model_validator(mode="after")
    def _consistent(self):
        train, valid = self.training_track_ids, self.validation_track_ids
        if (
            len(set(train)) != len(train)
            or len(set(valid)) != len(valid)
            or set(train) & set(valid)
        ):
            raise ValueError("competition preserves distinct original point identities")
        expected = any(
            trial.plausible
            and trial.direction_separation_degrees
            >= self.config.direction_separation_degrees
            for trial in self.trials
        )
        if self.ambiguity_detected != expected:
            raise ValueError(
                "ambiguity flag must match recorded competing explanations"
            )
        if self.trials and (
            tuple(trial.rotation_offset_degrees for trial in self.trials)
            != self.config.grid()
            or not train
            or not valid
        ):
            raise ValueError(
                "competition must retain the complete fixed grid and point partitions"
            )
        return self


def _epipolar_errors(first, second, camera, rotation, translation):
    x, y, z = translation
    skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    inverse = np.linalg.inv(camera)
    fundamental = inverse.T @ skew @ rotation @ inverse
    line1, line0 = first @ fundamental.T, second @ fundamental
    numerator = np.sum(second * line1, axis=1) ** 2
    denominator = np.sum(line1[:, :2] ** 2 + line0[:, :2] ** 2, axis=1)
    return np.sqrt(numerator / np.maximum(denominator, 1e-15))


def _translation(first, second, camera, rotation, threshold):
    """Fixed-rotation robust homogeneous fit using only original training rays."""
    inverse = np.linalg.inv(camera)
    design = np.cross(second @ inverse.T, (first @ inverse.T) @ rotation.T)
    weights = np.ones(len(first))
    for _ in range(4):
        _, singular, basis = np.linalg.svd(
            design * weights[:, None], full_matrices=False
        )
        if len(singular) != 3 or singular[1] < 1e-12:
            return None
        translation = basis[-1]
        errors = _epipolar_errors(first, second, camera, rotation, translation)
        weights = np.sqrt(np.minimum(1.0, threshold / np.maximum(errors, 1e-12)))
    return translation


def _trial(candidate, camera, config, offset, seed, cv2):
    settings, tracks = config.geometry, candidate.tracks
    indices = {identity: index for index, identity in enumerate(tracks.track_ids)}
    training = np.array([indices[item] for item in candidate.motion.training_track_ids])
    validation = np.array(
        [indices[item] for item in candidate.motion.validation_track_ids]
    )
    original_rotation, original_translation = _pose(candidate.motion.candidate)
    rotation = cv2.Rodrigues(np.radians(offset))[0] @ original_rotation
    pixels = np.asarray(tracks.pixels, dtype=float)
    first = np.column_stack((pixels[0], np.ones(len(indices))))
    second = np.column_stack((pixels[-1], np.ones(len(indices))))
    scale = tracks.image_shape[1] / 224
    threshold = settings.two_view.reference_ransac_pixels * scale
    translation = _translation(
        first[training], second[training], camera, rotation, threshold
    )
    base = {
        "rotation_offset_degrees": offset,
        "training_tracks": len(training),
        "validation_tracks": len(validation),
    }

    def reject(reason):
        return PoseWitnessTrial(plausible=False, reason=reason, **base)

    if translation is None:
        return reject("training rays cannot identify a fixed-rotation translation")
    error95 = float(
        np.quantile(
            _epipolar_errors(
                first[training], second[training], camera, rotation, translation
            ),
            0.95,
        )
    )
    base["training_epipolar_error_95_pixels"] = error95
    if error95 > threshold:
        return reject("alternative lacks original training epipolar support")
    map_config = VisualOdometryConfig(
        minimum_parallax_degrees=settings.two_view.minimum_parallax_degrees,
        reprojection_pixels=settings.two_view.reference_reprojection_pixels * scale,
    )
    ids = np.concatenate((training, validation))
    choices = []
    # Resolve sign by training cheirality only; withheld rays never choose a sign.
    for sign in (-1, 1):
        points, valid = triangulate_landmarks(
            pixels[0, ids],
            pixels[-1, ids],
            np.eye(3),
            np.zeros(3),
            rotation,
            sign * translation,
            camera,
            config=map_config,
            cv2=cv2,
        )
        choices.append((int(valid[: len(training)].sum()), sign, points, valid))
    _, sign, points, valid = max(choices, key=lambda item: (item[0], item[1]))
    translation = sign * translation
    original_center = -original_rotation.T @ original_translation
    center = -rotation.T @ translation
    base["direction_separation_degrees"] = math.degrees(
        math.acos(float(np.clip(original_center @ center, -1, 1)))
    )
    if not valid.all():
        return reject(
            "alternative loses original positive-depth/parallax/reprojection identities"
        )
    base["endpoint_partitions_valid"] = True
    train = np.arange(len(training))
    withheld = np.arange(len(training), len(ids))
    checks = []
    for slot in range(1, tracks.frame_count - 1):
        cv2.setRNGSeed(seed + slot)
        success, vector, shift, inliers = cv2.solvePnPRansac(
            points[train],
            pixels[slot, training],
            camera,
            None,
            iterationsCount=200,
            reprojectionError=settings.reference_pnp_pixels * scale,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        count = 0 if inliers is None else len(inliers)
        if not success or count < settings.minimum_training_inliers:
            checks.append(
                IntermediateFrameCheck(
                    frame_slot=slot,
                    supported=False,
                    reason="alternative PnP lacks fitting support",
                    training_inliers=count,
                    validation_tracks=len(validation),
                )
            )
            continue
        selected = train[inliers.ravel()]
        vector, shift = cv2.solvePnPRefineLM(
            points[selected],
            pixels[slot, training[inliers.ravel()]],
            camera,
            None,
            vector,
            shift,
        )
        if not np.isfinite(vector).all() or not np.isfinite(shift).all():
            checks.append(
                IntermediateFrameCheck(
                    frame_slot=slot,
                    supported=False,
                    reason="alternative PnP is nonfinite",
                    training_inliers=count,
                    validation_tracks=len(validation),
                )
            )
            continue
        projected = points[withheld] @ cv2.Rodrigues(vector)[0].T + shift.ravel()
        positive = projected[:, 2] > 0
        projected = projected @ camera.T
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
                frame_slot=slot,
                supported=supported,
                reason="alternative withheld-frame pixels consistent"
                if supported
                else "alternative withheld-frame pixels inconsistent",
                training_inliers=count,
                validation_tracks=len(validation),
                validation_acceptance=fraction,
                validation_error_95_pixels=error95,
                validation_cells=cells,
            )
        )
    base["frame_checks"] = tuple(checks)
    plausible = bool(checks and all(item.supported for item in checks))
    return PoseWitnessTrial(
        plausible=plausible,
        reason="competing map fits original withheld frames; accuracy unvalidated"
        if plausible
        else "competing map fails original withheld frames",
        **base,
    )


def inspect_pose_competition(
    candidate: KeyframeAttempt | None, camera, *, seed=0, config=None
):
    config = config or PoseCompetitionConfig()
    if type(seed) is not int or not 0 <= seed < 2**31 - 1024:
        raise ValueError("competition seed must reserve all trial/frame seeds")
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
        raise ValueError("competition requires finite zero-skew pinhole intrinsics")
    if candidate is None or not candidate.motion.supported:
        return PoseCompetitionEvidence(
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
        raise ValueError(
            "competition requires original complete fitting/withheld partitions"
        )
    cv2 = _opencv()
    trials = tuple(
        _trial(candidate, camera, config, offset, seed + index * 32, cv2)
        for index, offset in enumerate(config.grid())
    )
    ambiguous = any(
        trial.plausible
        and trial.direction_separation_degrees >= config.direction_separation_degrees
        for trial in trials
    )
    return PoseCompetitionEvidence(
        config=config,
        ambiguity_detected=ambiguous,
        reason="different motion explains the same withheld pixels"
        if ambiguous
        else "no witness in incomplete fixed family; uniqueness unproved",
        training_track_ids=training,
        validation_track_ids=validation,
        trials=trials,
    )
