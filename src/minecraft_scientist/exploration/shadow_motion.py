"""Veto a frozen local pose using two subsequent images, without reselection.

The map uses only the original endpoints. Future PnP trains on the original
training identities and scores the original disjoint validation identities.
Temporal holdout consistency is not ground-truth accuracy or metric navigation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .keyframe_motion import KeyframeAttempt
from .motion_stability import _pose
from .multiview_motion import IntermediateFrameCheck, MultiViewMotionConfig
from .temporal_tracks import TemporalTrackConfig, TemporalTracks
from .visual_geometry import _cells, _opencv
from .visual_odometry import VisualOdometryConfig, triangulate_landmarks


class ShadowMotionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    future_frames: Literal[2] = 2
    temporal: TemporalTrackConfig = TemporalTrackConfig()
    geometry: MultiViewMotionConfig = MultiViewMotionConfig()
    prefix_pixel_tolerance: FiniteFloat = Field(default=1e-6, gt=0, le=1e-3)


class ShadowMotionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: KeyframeAttempt | None
    supported: bool
    reason: str
    training_track_ids: tuple[int, ...] = ()
    validation_track_ids: tuple[int, ...] = ()
    frame_checks: tuple[IntermediateFrameCheck, ...] = ()
    relative_rotation: (
        tuple[tuple[FiniteFloat, FiniteFloat, FiniteFloat], ...] | None
    ) = None
    unit_translation: tuple[FiniteFloat, FiniteFloat, FiniteFloat] | None = None

    @model_validator(mode="after")
    def _frozen_candidate(self):
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
                "shadow fitting and validation identities must be disjoint"
            )
        if self.candidate is None:
            if training or validation or self.frame_checks or self.supported:
                raise ValueError("shadow evidence requires a frozen candidate")
        else:
            motion = self.candidate.motion
            if not training <= set(motion.training_track_ids) or not validation <= set(
                motion.validation_track_ids
            ):
                raise ValueError("shadow identities cannot be reassigned or invented")
            if self.frame_checks and tuple(
                item.frame_slot for item in self.frame_checks
            ) != (self.candidate.end_offset + 1, self.candidate.end_offset + 2):
                raise ValueError("shadow checks must contain both subsequent frames")
        if self.supported:
            if (
                self.candidate is None
                or not self.candidate.motion.supported
                or not self.frame_checks
                or not all(item.supported for item in self.frame_checks)
                or len(training) < 8
                or len(validation) < 6
                or self.relative_rotation != self.candidate.motion.relative_rotation
                or self.unit_translation != self.candidate.motion.unit_translation
            ):
                raise ValueError(
                    "shadow validation may retain or veto, never replace a pose"
                )
            _pose(self.candidate.motion.candidate)
        elif self.relative_rotation is not None or self.unit_translation is not None:
            raise ValueError(
                "rejected shadow evidence cannot expose an executable pose"
            )
        return self


def validate_shadow_motion(
    candidate: KeyframeAttempt | None,
    extended: TemporalTracks | None,
    camera: np.ndarray,
    *,
    config: ShadowMotionConfig | None = None,
    seed: int = 0,
) -> ShadowMotionEvidence:
    config = config or ShadowMotionConfig()
    if type(seed) is not int or not 0 <= seed < 2**31 - config.temporal.maximum_frames:
        raise ValueError("shadow seed must reserve all signed frame seeds")
    base = {"candidate": candidate}

    def reject(reason):
        return ShadowMotionEvidence(supported=False, reason=reason, **base)

    if candidate is None or not candidate.motion.supported:
        return reject("no supported frozen local candidate")
    endpoint = candidate.end_offset
    required = endpoint + config.future_frames + 1
    if (
        extended is None
        or required > config.temporal.maximum_frames
        or extended.frame_count != required
    ):
        return reject("insufficient bounded subsequent images")
    original = candidate.tracks
    if (
        extended.image_shape != original.image_shape
        or extended.seed_count != original.seed_count
    ):
        raise ValueError("shadow tracks must originate at the same real anchor")
    camera = np.asarray(camera, dtype=float)
    if (
        camera.shape != (3, 3)
        or not np.isfinite(camera).all()
        or camera[0, 0] <= 0
        or camera[1, 1] <= 0
        or not np.allclose(camera[2], (0, 0, 1))
        or abs(np.linalg.det(camera)) < 1e-9
    ):
        raise ValueError(
            "shadow validation requires finite nonsingular pinhole intrinsics"
        )
    old_indices = {identity: index for index, identity in enumerate(original.track_ids)}
    future_indices = {
        identity: index for index, identity in enumerate(extended.track_ids)
    }
    common = tuple(
        identity for identity in original.track_ids if identity in future_indices
    )
    if not common:
        return reject("original identities do not survive future images")
    old = np.asarray([old_indices[identity] for identity in common])
    new = np.asarray([future_indices[identity] for identity in common])
    old_pixels, pixels = np.asarray(original.pixels), np.asarray(extended.pixels)
    if not np.allclose(
        old_pixels[:, old],
        pixels[: endpoint + 1, new],
        rtol=0,
        atol=config.prefix_pixel_tolerance,
    ):
        return reject("future tracking changed frozen prefix observations")
    cv2 = _opencv()
    geometry = config.geometry
    rotation, translation = _pose(candidate.motion.candidate)
    scale = original.image_shape[1] / 224
    # Triangulation never consumes future pixels or a future-derived camera pose.
    points, valid = triangulate_landmarks(
        old_pixels[0, old],
        old_pixels[-1, old],
        np.eye(3),
        np.zeros(3),
        rotation,
        translation,
        camera,
        config=VisualOdometryConfig(
            minimum_parallax_degrees=geometry.two_view.minimum_parallax_degrees,
            reprojection_pixels=geometry.two_view.reference_reprojection_pixels * scale,
        ),
        cv2=cv2,
    )
    partitions = []
    for identities in (
        candidate.motion.training_track_ids,
        candidate.motion.validation_track_ids,
    ):
        identity_set = set(identities)
        partitions.append(
            np.asarray(
                [
                    index
                    for index, identity in enumerate(common)
                    if valid[index] and identity in identity_set
                ],
                dtype=int,
            )
        )
    training, validation = partitions
    base.update(
        training_track_ids=tuple(common[index] for index in training),
        validation_track_ids=tuple(common[index] for index in validation),
    )
    if (
        len(training) + len(validation) < geometry.minimum_tracks
        or len(training) < geometry.minimum_training_inliers
        or len(validation) < geometry.minimum_validation_tracks
    ):
        return reject("insufficient surviving original validation partitions")
    checks = []
    for slot in range(endpoint + 1, required):
        cv2.setRNGSeed(seed + slot)
        success, vector, offset, inliers = cv2.solvePnPRansac(
            points[training],
            pixels[slot, new[training]],
            camera,
            None,
            iterationsCount=200,
            reprojectionError=geometry.reference_pnp_pixels * scale,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        count = 0 if inliers is None else len(inliers)
        values = {
            "frame_slot": slot,
            "training_inliers": count,
            "validation_tracks": len(validation),
        }
        if not success or count < geometry.minimum_training_inliers:
            checks.append(
                IntermediateFrameCheck(
                    supported=False, reason="future PnP lacks fitting support", **values
                )
            )
            continue
        fitted = training[inliers.ravel()]
        vector, offset = cv2.solvePnPRefineLM(
            points[fitted], pixels[slot, new[fitted]], camera, None, vector, offset
        )
        if not np.isfinite(vector).all() or not np.isfinite(offset).all():
            checks.append(
                IntermediateFrameCheck(
                    supported=False, reason="future PnP is nonfinite", **values
                )
            )
            continue
        predicted = points[validation] @ cv2.Rodrigues(vector)[0].T + offset.ravel()
        projected = predicted @ camera.T
        projected = projected[:, :2] / np.maximum(projected[:, 2:], 1e-9)
        errors = np.linalg.norm(projected - pixels[slot, new[validation]], axis=1)
        positive = np.isfinite(errors) & (predicted[:, 2] > 0)
        passing = positive & (errors <= geometry.reference_validation_pixels * scale)
        error95 = float(np.quantile(errors[positive], 0.95)) if positive.any() else None
        fraction = float(passing.mean())
        cells = _cells(pixels[slot, new[validation[passing]]], *original.image_shape)
        supported = bool(
            positive.all()
            and fraction >= geometry.minimum_validation_acceptance
            and error95 is not None
            and error95 <= geometry.reference_validation_pixels * scale
            and cells >= geometry.minimum_validation_cells
        )
        checks.append(
            IntermediateFrameCheck(
                supported=supported,
                reason="future withheld-point geometry consistent"
                if supported
                else "future withheld-point geometry inconsistent",
                validation_acceptance=fraction,
                validation_error_95_pixels=error95,
                validation_cells=cells,
                **values,
            )
        )
    base["frame_checks"] = tuple(checks)
    if not all(item.supported for item in checks):
        return reject("independent subsequent-frame geometry failed")
    return ShadowMotionEvidence(
        supported=True,
        reason="subsequent-frame consistent; external accuracy unvalidated",
        relative_rotation=candidate.motion.relative_rotation,
        unit_translation=candidate.motion.unit_translation,
        **base,
    )
