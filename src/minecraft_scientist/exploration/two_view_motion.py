"""Fail-closed relative-motion evidence from public matched pixels.

Support does not establish correct motion: repeated texture and dynamic objects
can still produce a plausible epipolar model. Independent motion validation is
mandatory before these candidates enter navigation or persistent memory.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from .dense_correspondence import DenseCorrespondences
from .visual_geometry import _cells, _opencv
from .visual_odometry import VisualOdometryConfig, triangulate_landmarks


class TwoViewMotionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_matches: int = Field(default=20, ge=8)
    minimum_positive_inliers: int = Field(default=15, ge=8)
    minimum_spatial_cells: int = Field(default=4, ge=2, le=16)
    minimum_parallax_degrees: float = Field(default=1.0, gt=0, lt=30)
    reference_ransac_pixels: float = Field(default=1.5, gt=0)
    reference_reprojection_pixels: float = Field(default=2.0, gt=0)
    pure_rotation_support_fraction: float = Field(default=0.9, gt=0, le=1)
    pure_rotation_singular_spread: float = Field(default=0.025, gt=0, lt=1)


class TwoViewMotionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    supported: bool
    reason: str
    matches: int = Field(ge=0)
    positive_inliers: int = Field(default=0, ge=0)
    triangulated_inliers: int = Field(default=0, ge=0)
    spatial_cells: int = Field(default=0, ge=0)
    homography_inliers: int = Field(default=0, ge=0)
    rotation_singular_spread: FiniteFloat | None = None
    median_parallax_degrees: FiniteFloat | None = None
    relative_rotation: (
        tuple[tuple[FiniteFloat, FiniteFloat, FiniteFloat], ...] | None
    ) = None
    unit_translation: tuple[FiniteFloat, FiniteFloat, FiniteFloat] | None = None


def estimate_two_view_motion(
    correspondences: DenseCorrespondences,
    camera: np.ndarray,
    *,
    config: TwoViewMotionConfig | None = None,
    seed: int = 0,
) -> TwoViewMotionEvidence:
    config = config or TwoViewMotionConfig()
    # The existing tracker validates the supported pinhole model; constructing
    # it would carry state, so validate intrinsics directly for this pure fit.
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
        raise ValueError("two-view evidence requires zero-skew pinhole intrinsics")
    if correspondences.image_shape0 != correspondences.image_shape1:
        raise ValueError("two-view evidence requires same-size images")
    first, second = (
        np.asarray(correspondences.points0).reshape(-1, 2),
        np.asarray(correspondences.points1).reshape(-1, 2),
    )
    count = len(first)
    base = {"matches": count}
    if count < config.minimum_matches:
        return TwoViewMotionEvidence(
            supported=False, reason="insufficient correspondences", **base
        )
    height, width = correspondences.image_shape0
    cells = min(_cells(first, height, width), _cells(second, height, width))
    if cells < config.minimum_spatial_cells:
        return TwoViewMotionEvidence(
            supported=False,
            reason="insufficient spatial coverage",
            spatial_cells=cells,
            **base,
        )
    cv2 = _opencv()
    cv2.setRNGSeed(seed)
    pixel_scale = width / 224
    homography, mask = cv2.findHomography(
        first, second, cv2.RANSAC, config.reference_ransac_pixels * pixel_scale
    )
    homography_inliers = 0 if mask is None else int(mask.sum())
    base["homography_inliers"] = homography_inliers
    if homography is not None and np.isfinite(homography).all():
        calibrated = np.linalg.inv(camera) @ homography @ camera
        singular = np.linalg.svd(calibrated, compute_uv=False)
        spread = float(
            (singular.max() - singular.min()) / max(float(np.median(singular)), 1e-9)
        )
        base["rotation_singular_spread"] = spread
        if (
            homography_inliers / count >= config.pure_rotation_support_fraction
            and spread <= config.pure_rotation_singular_spread
        ):
            return TwoViewMotionEvidence(
                supported=False, reason="pure rotation or stationary ambiguity", **base
            )
    cv2.setRNGSeed(seed)
    essential, mask = cv2.findEssentialMat(
        first,
        second,
        camera,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=config.reference_ransac_pixels * pixel_scale,
    )
    if essential is None or essential.shape != (3, 3) or mask is None:
        return TwoViewMotionEvidence(
            supported=False, reason="essential model unavailable", **base
        )
    count, rotation, translation, positive = cv2.recoverPose(
        essential, first, second, camera, mask=mask
    )
    base["positive_inliers"] = int(count)
    if count < config.minimum_positive_inliers:
        return TwoViewMotionEvidence(
            supported=False, reason="insufficient positive-depth support", **base
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
            reprojection_pixels=config.reference_reprojection_pixels * pixel_scale,
        ),
        cv2=cv2,
    )
    valid &= positive.ravel() > 0
    cells = min(
        _cells(first[valid], height, width), _cells(second[valid], height, width)
    )
    base.update(triangulated_inliers=int(valid.sum()), spatial_cells=cells)
    if (
        int(valid.sum()) < config.minimum_positive_inliers
        or cells < config.minimum_spatial_cells
    ):
        return TwoViewMotionEvidence(
            supported=False, reason="triangulation lacks parallax or coverage", **base
        )
    center = -rotation.T @ translation.ravel()
    first_rays = points[valid] / np.linalg.norm(points[valid], axis=1, keepdims=True)
    second_rays = points[valid] - center
    second_rays /= np.linalg.norm(second_rays, axis=1, keepdims=True)
    angles = np.degrees(
        np.arccos(np.clip(np.sum(first_rays * second_rays, axis=1), -1, 1))
    )
    return TwoViewMotionEvidence(
        supported=True,
        reason="supported relative motion; accuracy unvalidated",
        relative_rotation=tuple(map(tuple, rotation)),
        unit_translation=tuple(translation.ravel()),
        median_parallax_degrees=float(np.median(angles)),
        **base,
    )
