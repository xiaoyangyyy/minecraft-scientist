"""Measure future view diversity conditional on a frozen provisional RGB map.

Neither camera-center ratios nor ray angles are privileged simulator telemetry.
Passing this diagnostic does not prove that competing maps are distinguishable.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .motion_stability import _pose
from .shadow_motion import (
    ShadowMotionConfig,
    ShadowMotionEvidence,
    validate_shadow_motion,
)
from .temporal_tracks import TemporalTracks
from .visual_geometry import _opencv
from .visual_odometry import VisualOdometryConfig, triangulate_landmarks


class ParallaxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_future_baseline_ratio: FiniteFloat = Field(default=0.25, gt=0)
    minimum_median_ray_angle_degrees: FiniteFloat = Field(default=1, gt=0, lt=30)
    minimum_turn_degrees: FiniteFloat = Field(default=45, gt=0, lt=90)
    maximum_turn_degrees: FiniteFloat = Field(default=135, gt=90, lt=180)


class ConditionalView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_slot: int = Field(ge=1, strict=True)
    baseline_ratio: FiniteFloat = Field(ge=0)
    turn_degrees: FiniteFloat = Field(ge=0, le=180)
    median_ray_angle_degrees: FiniteFloat = Field(ge=0, le=180)
    validation_tracks: int = Field(ge=6, strict=True)

    def diverse(self, config):
        return (
            self.baseline_ratio >= config.minimum_future_baseline_ratio
            and self.median_ray_angle_degrees >= config.minimum_median_ray_angle_degrees
            and config.minimum_turn_degrees
            <= self.turn_degrees
            <= config.maximum_turn_degrees
        )


class ParallaxEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: ParallaxConfig
    conditionally_diverse_view: bool
    reason: str
    views: tuple[ConditionalView, ...] = ()

    @model_validator(mode="after")
    def _consistent(self):
        if self.conditionally_diverse_view != any(
            item.diverse(self.config) for item in self.views
        ) or len({item.frame_slot for item in self.views}) != len(self.views):
            raise ValueError(
                "conditional view diversity must match the measured geometry"
            )
        return self


def measure_parallax(
    shadow: ShadowMotionEvidence,
    tracks: TemporalTracks | None,
    camera: np.ndarray,
    *,
    seed=0,
    config: ParallaxConfig | None = None,
):
    config = config or ParallaxConfig()
    checked = validate_shadow_motion(shadow.candidate, tracks, camera, seed=seed)
    if checked != shadow:
        raise ValueError(
            "parallax measurement requires reproducible original shadow evidence"
        )
    if not checked.supported:
        return ParallaxEvidence(
            config=config,
            conditionally_diverse_view=False,
            reason="no consistent frozen shadow map",
        )
    candidate = checked.candidate
    original = candidate.tracks
    old_indices = {identity: index for index, identity in enumerate(original.track_ids)}
    new_indices = {identity: index for index, identity in enumerate(tracks.track_ids)}
    identities = checked.training_track_ids + checked.validation_track_ids
    old = np.asarray([old_indices[identity] for identity in identities])
    new = np.asarray([new_indices[identity] for identity in identities])
    rotation, translation = _pose(candidate.motion.candidate)
    center = -rotation.T @ translation
    settings = ShadowMotionConfig().geometry
    cv2 = _opencv()
    camera = np.asarray(camera, dtype=float)
    pixels = np.asarray(original.pixels)
    points, valid = triangulate_landmarks(
        pixels[0, old],
        pixels[-1, old],
        np.eye(3),
        np.zeros(3),
        rotation,
        translation,
        camera,
        config=VisualOdometryConfig(
            minimum_parallax_degrees=settings.two_view.minimum_parallax_degrees,
            reprojection_pixels=settings.two_view.reference_reprojection_pixels
            * original.image_shape[1]
            / 224,
        ),
        cv2=cv2,
    )
    if not valid.all():
        raise ValueError(
            "conditional map differs from validated original triangulation"
        )
    train_count = len(checked.training_track_ids)
    train, withheld = np.arange(train_count), np.arange(train_count, len(identities))
    future_pixels = np.asarray(tracks.pixels)
    views = []
    for frame in checked.frame_checks:
        slot = frame.frame_slot
        cv2.setRNGSeed(seed + slot)
        success, vector, offset, inliers = cv2.solvePnPRansac(
            points[train],
            future_pixels[slot, new[train]],
            camera,
            None,
            iterationsCount=200,
            reprojectionError=settings.reference_pnp_pixels
            * original.image_shape[1]
            / 224,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if (
            not success
            or inliers is None
            or len(inliers) < settings.minimum_training_inliers
        ):
            raise ValueError("conditional future pose differs from validated PnP")
        selected = train[inliers.ravel()]
        vector, offset = cv2.solvePnPRefineLM(
            points[selected],
            future_pixels[slot, new[selected]],
            camera,
            None,
            vector,
            offset,
        )
        future_center = -cv2.Rodrigues(vector)[0].T @ offset.ravel()
        delta = future_center - center
        ratio = float(np.linalg.norm(delta) / np.linalg.norm(center))
        turn = (
            math.degrees(
                math.acos(
                    float(
                        np.clip(
                            center
                            @ delta
                            / (np.linalg.norm(center) * np.linalg.norm(delta)),
                            -1,
                            1,
                        )
                    )
                )
            )
            if np.linalg.norm(delta) > 1e-9
            else 0.0
        )
        rays = [points[withheld] - origin for origin in (center, future_center)]
        rays = [
            ray / np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-9)
            for ray in rays
        ]
        angles = np.degrees(
            np.arccos(np.clip(np.sum(rays[0] * rays[1], axis=1), -1, 1))
        )
        views.append(
            ConditionalView(
                frame_slot=slot,
                baseline_ratio=ratio,
                turn_degrees=turn,
                median_ray_angle_degrees=float(np.median(angles)),
                validation_tracks=len(withheld),
            )
        )
    diverse = any(item.diverse(config) for item in views)
    return ParallaxEvidence(
        config=config,
        conditionally_diverse_view=diverse,
        reason="conditional angular/baseline diversity; competing maps not discriminated"
        if diverse
        else "future images lack required conditional view diversity",
        views=tuple(views),
    )
