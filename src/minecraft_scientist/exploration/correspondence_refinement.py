"""Bounded bidirectional RGB refinement of independently generated match seeds.

Neural scores remain uncalibrated scores. Refinement checks pixel consistency;
it neither assigns object identity nor supplies a metric-motion guarantee.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .dense_correspondence import DenseCorrespondences
from .visual_geometry import _frame, _opencv


class CorrespondenceRefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_window_pixels: int = Field(default=15, ge=5, le=41)
    pyramid_levels: int = Field(default=2, ge=0, le=4)
    reference_maximum_correction: FiniteFloat = Field(default=3, gt=0, le=10)
    reference_cycle_error: FiniteFloat = Field(default=0.5, gt=0, le=3)
    reference_patch_pixels: int = Field(default=7, ge=3, le=21)
    minimum_patch_correlation: FiniteFloat = Field(default=0.8, ge=-1, le=1)
    minimum_patch_std: FiniteFloat = Field(default=2, gt=0, le=50)
    reference_pixel_spacing: FiniteFloat = Field(default=2, gt=0)


class RefinedCorrespondences(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    correspondences: DenseCorrespondences
    seed_count: int = Field(ge=0)
    cycle_consistent_count: int = Field(ge=0)
    photometrically_consistent_count: int = Field(ge=0)
    retained_source_indices: tuple[int, ...]

    @model_validator(mode="after")
    def _consistent(self) -> RefinedCorrespondences:
        if (
            not self.seed_count
            >= self.cycle_consistent_count
            >= self.photometrically_consistent_count
            >= len(self.retained_source_indices)
        ):
            raise ValueError("refinement counts must be monotone")
        if len(self.retained_source_indices) != len(
            self.correspondences.points0
        ) or len(set(self.retained_source_indices)) != len(
            self.retained_source_indices
        ):
            raise ValueError("refinement must preserve unique aligned seed provenance")
        if any(
            not 0 <= index < self.seed_count for index in self.retained_source_indices
        ):
            raise ValueError("refinement seed index outside input")
        return self


def _crop_gray(frame: np.ndarray, cv2) -> tuple[np.ndarray, np.ndarray]:
    rgb = _frame(frame)
    height, width = rgb.shape[:2]
    x0, x1, y0, y1 = (
        int(width * 0.18),
        int(width * 0.82),
        int(height * 0.2),
        int(height * 0.76),
    )
    return cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY), np.array(
        (x0, y0), dtype=float
    )


def _odd(value: float) -> int:
    return max(3, round(value) // 2 * 2 + 1)


def refine_correspondences(
    first_frame: np.ndarray,
    second_frame: np.ndarray,
    seeds: DenseCorrespondences,
    *,
    config: CorrespondenceRefinementConfig | None = None,
) -> RefinedCorrespondences:
    config = config or CorrespondenceRefinementConfig()
    cv2 = _opencv()
    first, offset0 = _crop_gray(first_frame, cv2)
    second, offset1 = _crop_gray(second_frame, cv2)
    if (
        tuple(np.asarray(first_frame).shape[:2]) != seeds.image_shape0
        or tuple(np.asarray(second_frame).shape[:2]) != seeds.image_shape1
        or seeds.image_shape0 != seeds.image_shape1
    ):
        raise ValueError("refinement requires matched same-size RGB images")
    scale = seeds.image_shape0[1] / 224
    window, patch = (
        _odd(config.reference_window_pixels * scale),
        _odd(config.reference_patch_pixels * scale),
    )
    margin = max(window, patch) / 2 + 1
    points0 = np.asarray(seeds.points0, dtype=np.float32).reshape(
        -1, 2
    ) - offset0.astype(np.float32)
    points1 = np.asarray(seeds.points1, dtype=np.float32).reshape(
        -1, 2
    ) - offset1.astype(np.float32)

    def inside(points, image):
        return (
            np.isfinite(points).all(axis=1)
            & (points >= margin).all(axis=1)
            & (points[:, 0] < image.shape[1] - margin)
            & (points[:, 1] < image.shape[0] - margin)
        )

    indices = np.flatnonzero(inside(points0, first) & inside(points1, second))

    def finish(retained, targets, cycle_count=0, photo_count=0):
        return RefinedCorrespondences(
            correspondences=DenseCorrespondences(
                points0=tuple(seeds.points0[index] for index in retained),
                points1=tuple(map(tuple, targets)),
                confidence=tuple(seeds.confidence[index] for index in retained),
                image_shape0=seeds.image_shape0,
                image_shape1=seeds.image_shape1,
                raw_match_count=seeds.raw_match_count,
            ),
            seed_count=len(seeds.points0),
            cycle_consistent_count=cycle_count,
            photometrically_consistent_count=photo_count,
            retained_source_indices=tuple(int(index) for index in retained),
        )

    if not len(indices):
        return finish([], np.empty((0, 2)))
    kwargs = {
        "winSize": (window, window),
        "maxLevel": config.pyramid_levels,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01),
        "flags": cv2.OPTFLOW_USE_INITIAL_FLOW,
    }
    source = points0[indices].reshape(-1, 1, 2)
    target, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        first, second, source, points1[indices].reshape(-1, 1, 2).copy(), **kwargs
    )
    if target is None or forward_status is None:
        return finish([], np.empty((0, 2)))
    # Feed only valid forward tracks into the reverse solver. Failed LK outputs
    # are not guaranteed meaningful or finite and must never seed another solve.
    valid = (forward_status.ravel() > 0) & inside(target.reshape(-1, 2), second)
    valid &= (
        np.linalg.norm(target.reshape(-1, 2) - points1[indices], axis=1)
        <= config.reference_maximum_correction * scale
    )
    indices, target = indices[valid], target[valid]
    if not len(indices):
        return finish([], np.empty((0, 2)))
    source = points0[indices].reshape(-1, 1, 2)
    reverse, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        second, first, target, source.copy(), **kwargs
    )
    if reverse is None or backward_status is None:
        return finish([], np.empty((0, 2)))
    valid = (backward_status.ravel() > 0) & inside(reverse.reshape(-1, 2), first)
    valid &= (
        np.linalg.norm(reverse.reshape(-1, 2) - source.reshape(-1, 2), axis=1)
        <= config.reference_cycle_error * scale
    )
    indices, target = indices[valid], target[valid].reshape(-1, 2)
    cycle_count = len(indices)
    photo = []
    for local_index, (index, point) in enumerate(zip(indices, target, strict=True)):
        left = (
            cv2.getRectSubPix(first, (patch, patch), tuple(map(float, points0[index])))
            .astype(float)
            .ravel()
        )
        right = (
            cv2.getRectSubPix(second, (patch, patch), tuple(map(float, point)))
            .astype(float)
            .ravel()
        )
        if min(float(left.std()), float(right.std())) < config.minimum_patch_std:
            continue
        left, right = left - left.mean(), right - right.mean()
        correlation = float(
            left @ right / (np.linalg.norm(left) * np.linalg.norm(right))
        )
        if correlation >= config.minimum_patch_correlation:
            photo.append(local_index)
    indices, target = indices[photo], target[photo] + offset1
    order = sorted(
        range(len(indices)),
        key=lambda index: (
            -seeds.confidence[indices[index]],
            *seeds.points0[indices[index]],
            *target[index],
        ),
    )
    occupied, retained = set(), []
    for index in order:
        key = tuple(
            np.floor(target[index] / (config.reference_pixel_spacing * scale)).astype(
                int
            )
        )
        if key not in occupied:
            occupied.add(key)
            retained.append(index)
    return finish(indices[retained], target[retained], cycle_count, len(photo))
