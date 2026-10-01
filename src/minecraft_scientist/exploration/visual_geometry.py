"""Optional RGB-only calibrated two-view evidence for visual navigation.

OpenCV is loaded lazily. No simulator coordinates, FOV telemetry, semantic labels
or pretrained weights enter this module. Rotation scans calibrate focal length;
feature correspondences estimate a unit direction, not metric travel distance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class VisualGeometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_budget: int = Field(default=800, ge=32)
    ratio_threshold: float = Field(default=0.75, gt=0, lt=1)
    minimum_matches: int = Field(default=12, ge=8)
    minimum_inliers: int = Field(default=8, ge=5)
    minimum_cells: int = Field(default=4, ge=2)
    minimum_focal_samples: int = Field(default=20, ge=5)
    minimum_consensus_views: int = Field(default=3, ge=2)
    consensus_degrees: float = Field(default=25, gt=0, lt=90)
    maximum_focal_relative_mad: float = Field(default=0.1, gt=0, lt=1)
    ransac_pixels: float = Field(default=1.5, gt=0)


class VisualGoalBearing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted: bool
    reason: str
    focal_pixels: float | None = None
    focal_sample_count: int = 0
    focal_relative_mad: float | None = None
    direction: tuple[float, float, float] | None = None
    supported_query_views: tuple[int, ...] = ()
    supported_goal_views: tuple[int, ...] = ()
    consensus_pair_count: int = 0
    candidate_pair_count: int = 0
    pairs: tuple[dict[str, Any], ...] = ()


def _opencv() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "visual geometry requires the optional OpenCV runtime"
        ) from error
    cv2.setNumThreads(1)
    return cv2


def _frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3 or min(array.shape[:2]) < 32:
        raise ValueError("visual geometry requires RGB frames of at least 32x32")
    if not np.isfinite(array).all():
        raise ValueError("visual geometry requires finite pixels")
    return np.clip(array[:, :, :3], 0, 255).astype(np.uint8)


def _features(
    frame: np.ndarray, config: VisualGeometryConfig, cv2: Any
) -> tuple[np.ndarray, Any]:
    rgb = _frame(frame)
    height, width = rgb.shape[:2]
    x0, x1 = int(width * 0.18), int(width * 0.82)
    y0, y1 = int(height * 0.2), int(height * 0.76)
    # Crop BEFORE filtering/equalization: pixels outside the public world ROI
    # must not influence descriptors through filter support or CLAHE tiles.
    gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    keypoints, descriptors = cv2.SIFT_create(
        nfeatures=config.feature_budget
    ).detectAndCompute(gray, None)
    points = np.asarray(
        [(item.pt[0] + x0, item.pt[1] + y0) for item in keypoints], dtype=np.float64
    ).reshape(-1, 2)
    return points, descriptors


def _matched_indices(
    left: tuple[np.ndarray, Any],
    right: tuple[np.ndarray, Any],
    config: VisualGeometryConfig,
    cv2: Any,
) -> tuple[np.ndarray, np.ndarray]:
    left_points, left_descriptors = left
    right_points, right_descriptors = right
    empty = np.empty(0, dtype=np.int64)
    if (
        left_descriptors is None
        or right_descriptors is None
        or min(len(left_points), len(right_points)) < 2
    ):
        return empty, empty
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    reverse = matcher.knnMatch(right_descriptors, left_descriptors, k=2)
    reverse_good = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse
        if len(pair) == 2
        and pair[0].distance < config.ratio_threshold * pair[1].distance
    }
    good = [
        pair[0]
        for pair in forward
        if len(pair) == 2
        and pair[0].distance < config.ratio_threshold * pair[1].distance
        and reverse_good.get(pair[0].trainIdx) == pair[0].queryIdx
    ]
    return (
        np.asarray([item.queryIdx for item in good], dtype=np.int64),
        np.asarray([item.trainIdx for item in good], dtype=np.int64),
    )


def _matches(
    left: tuple[np.ndarray, Any],
    right: tuple[np.ndarray, Any],
    config: VisualGeometryConfig,
    cv2: Any,
) -> tuple[np.ndarray, np.ndarray]:
    first, second = _matched_indices(left, right, config, cv2)
    return left[0][first], right[0][second]


def _calibrate_focal(
    features: Sequence[tuple[np.ndarray, Any]],
    width: int,
    config: VisualGeometryConfig,
    cv2: Any,
) -> tuple[float | None, int, float | None]:
    """Fit a pinhole focal from corresponding pixels under a full yaw scan.

    tan(delta) * f^2 - (x1-x2) * f + tan(delta) * x1*x2 = 0.
    The larger positive root is the physical focal. Small roots and mismatches are
    rejected; robust dispersion is checked before using the camera model.
    """
    tangent = math.tan(2 * math.pi / len(features))
    values = []
    for index, left in enumerate(features):
        p1, p2 = _matches(left, features[(index + 1) % len(features)], config, cv2)
        for first, second in zip(p1, p2, strict=True):
            x1, x2 = float(first[0] - width / 2), float(second[0] - width / 2)
            discriminant = (x1 - x2) ** 2 - 4 * tangent**2 * x1 * x2
            if discriminant <= 0 or x1 <= x2:
                continue
            focal = ((x1 - x2) + math.sqrt(discriminant)) / (2 * tangent)
            if width * 0.3 <= focal <= width * 2:
                values.append(focal)
    if len(values) < config.minimum_focal_samples:
        return None, len(values), None
    median = float(np.median(values))
    mad = float(np.median(np.abs(np.asarray(values) - median))) / median
    if mad > config.maximum_focal_relative_mad:
        return None, len(values), mad
    return median, len(values), mad


def _cells(points: np.ndarray, height: int, width: int) -> int:
    return len(
        {
            (min(3, int(point[0] * 4 / width)), min(3, int(point[1] * 4 / height)))
            for point in points
        }
    )


def estimate_visual_goal_bearing(
    goal_frames: Sequence[np.ndarray],
    query_frames: Sequence[np.ndarray],
    *,
    config: VisualGeometryConfig | None = None,
) -> VisualGoalBearing:
    """Estimate a consensus direction to a remembered camera center from RGB.

    Input frames must be a same-size ordered full yaw scan with at least eight
    equal turns. Focal calibration assumes fixed horizontal pitch. Estimates with
    inadequate cheirality, spatial support or cross-view agreement fail closed.
    """
    config = config or VisualGeometryConfig()
    cv2 = _opencv()
    if len(goal_frames) != len(query_frames) or len(goal_frames) < 8:
        raise ValueError(
            "geometry requires equal ordered scans with at least eight views"
        )
    frames = [_frame(item) for item in (*goal_frames, *query_frames)]
    if any(item.shape != frames[0].shape for item in frames):
        raise ValueError("geometry frame dimensions differ")
    period = len(goal_frames)
    height, width = frames[0].shape[:2]
    goal_features = [_features(item, config, cv2) for item in frames[:period]]
    query_features = [_features(item, config, cv2) for item in frames[period:]]
    focal, sample_count, mad = _calibrate_focal(goal_features, width, config, cv2)
    base = {
        "focal_pixels": focal,
        "focal_sample_count": sample_count,
        "focal_relative_mad": mad,
    }
    if focal is None:
        return VisualGoalBearing(
            accepted=False,
            reason="rotation calibration lacks consistent support",
            **base,
        )
    camera = np.asarray(
        ((focal, 0, width / 2), (0, focal, height / 2), (0, 0, 1)), dtype=np.float64
    )
    pairs = []
    for goal_index, goal_feature in enumerate(goal_features):
        for query_index, query_feature in enumerate(query_features):
            p1, p2 = _matches(goal_feature, query_feature, config, cv2)
            if len(p1) < config.minimum_matches:
                continue
            if (
                min(_cells(p1, height, width), _cells(p2, height, width))
                < config.minimum_cells
            ):
                continue
            cv2.setRNGSeed(goal_index * period + query_index)
            essential, mask = cv2.findEssentialMat(
                p1,
                p2,
                camera,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=config.ransac_pixels,
            )
            if essential is None or essential.shape != (3, 3) or mask is None:
                continue
            inliers, rotation, translation, positive_mask = cv2.recoverPose(
                essential, p1, p2, camera, mask=mask
            )
            if inliers < config.minimum_inliers:
                continue
            positive = positive_mask.ravel() > 0
            if (
                min(
                    _cells(p1[positive], height, width),
                    _cells(p2[positive], height, width),
                )
                < config.minimum_cells
            ):
                continue
            # Fixed-pitch panorama rotation must preserve the camera vertical axis.
            if float(rotation[1, 1]) < math.cos(math.radians(15)):
                continue
            angle = query_index * 2 * math.pi / period
            tx, ty, tz = (float(item) for item in translation.ravel())
            direction = (
                math.cos(angle) * tx + math.sin(angle) * tz,
                ty,
                -math.sin(angle) * tx + math.cos(angle) * tz,
            )
            pairs.append(
                {
                    "goal_view": goal_index,
                    "query_view": query_index,
                    "matches": len(p1),
                    "positive_inliers": int(inliers),
                    "direction": direction,
                }
            )
    if not pairs:
        return VisualGoalBearing(
            accepted=False, reason="no spatially supported two-view pose", **base
        )
    vectors = np.asarray([item["direction"] for item in pairs], dtype=np.float64)
    agreement = vectors @ vectors.T >= math.cos(math.radians(config.consensus_degrees))
    weights = np.asarray([item["positive_inliers"] for item in pairs])
    # Coverage precedes match count so one repeated facade cannot decide the bearing.
    best_index = max(
        range(len(pairs)),
        key=lambda index: (
            len(
                {pairs[item]["query_view"] for item in np.flatnonzero(agreement[index])}
            ),
            int(weights[agreement[index]].sum()),
            -index,
        ),
    )
    indices = np.flatnonzero(agreement[best_index])
    query_views = tuple(sorted({pairs[index]["query_view"] for index in indices}))
    goal_views = tuple(sorted({pairs[index]["goal_view"] for index in indices}))
    supported = min(len(query_views), len(goal_views)) >= config.minimum_consensus_views
    vector = np.sum(vectors[indices] * weights[indices, None], axis=0)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return VisualGoalBearing(
        accepted=supported,
        reason="cross-view consensus"
        if supported
        else "insufficient independent view consensus",
        direction=tuple(float(item) for item in vector) if supported else None,
        supported_query_views=query_views,
        supported_goal_views=goal_views,
        consensus_pair_count=len(indices),
        candidate_pair_count=len(pairs),
        pairs=tuple(pairs),
        **base,
    )
