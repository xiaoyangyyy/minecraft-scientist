"""Bounded, identity-preserving tracks over consecutive public RGB images.

Track indices refer to the first image's detector/seed order. Failed tracks never
reappear under another identity, and outside-ROI pixels never enter a pyramid.
Pixel consistency is not semantic identity or correct metric movement.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .correspondence_refinement import _crop_gray, _odd
from .dense_correspondence import DenseCorrespondences
from .visual_geometry import _opencv

Pixel = tuple[FiniteFloat, FiniteFloat]


class TemporalTrackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_budget: int = Field(default=800, ge=32, le=2000)
    maximum_frames: int = Field(default=32, ge=3, le=128)
    corner_quality: FiniteFloat = Field(default=0.01, gt=0, lt=1)
    reference_feature_spacing: FiniteFloat = Field(default=3, gt=0)
    reference_window_pixels: int = Field(default=15, ge=5, le=41)
    pyramid_levels: int = Field(default=3, ge=0, le=4)
    reference_maximum_step: FiniteFloat = Field(default=25, gt=0, le=80)
    reference_cycle_error: FiniteFloat = Field(default=0.5, gt=0, le=3)
    reference_chain_cycle_error: FiniteFloat = Field(default=1, gt=0, le=5)
    reference_patch_pixels: int = Field(default=7, ge=3, le=21)
    minimum_patch_correlation: FiniteFloat = Field(default=0.8, ge=0, le=1)
    minimum_patch_std: FiniteFloat = Field(default=2, gt=0, le=50)
    median_filter: bool = True


class TemporalTracks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_shape: tuple[int, int]
    frame_count: int = Field(ge=2, le=128)
    seed_count: int = Field(ge=0, le=2000)
    surviving_counts: tuple[int, ...]
    track_ids: tuple[int, ...]
    pixels: tuple[tuple[Pixel, ...], ...]
    minimum_patch_correlations: tuple[FiniteFloat, ...]
    chain_cycle_errors: tuple[FiniteFloat, ...]

    @model_validator(mode="after")
    def _aligned(self) -> TemporalTracks:
        count = len(self.track_ids)
        if (
            len(self.pixels) != self.frame_count
            or len(self.surviving_counts) != self.frame_count
        ):
            raise ValueError("track frames and survival counts must align")
        if (
            any(len(points) != count for points in self.pixels)
            or len(self.minimum_patch_correlations) != count
            or len(self.chain_cycle_errors) != count
        ):
            raise ValueError(
                "all frames must preserve aligned surviving track identities"
            )
        if len(set(self.track_ids)) != count or any(
            not 0 <= index < self.seed_count for index in self.track_ids
        ):
            raise ValueError("track IDs must be unique first-frame seed indices")
        if (
            not self.surviving_counts
            or self.surviving_counts[0] > self.seed_count
            or self.surviving_counts[-1] < count
            or any(not 0 <= value <= self.seed_count for value in self.surviving_counts)
            or any(
                left < right
                for left, right in zip(self.surviving_counts, self.surviving_counts[1:])
            )
        ):
            raise ValueError(
                "track survival counts cannot increase or exceed seed budget"
            )
        height, width = self.image_shape
        if min(self.image_shape) < 32 or any(
            not 0 <= x < width or not 0 <= y < height
            for frame in self.pixels
            for x, y in frame
        ):
            raise ValueError("temporal track lies outside its public image")
        if any(not 0 <= score <= 1 for score in self.minimum_patch_correlations) or any(
            value < 0 for value in self.chain_cycle_errors
        ):
            raise ValueError("invalid pixel consistency statistic")
        return self

    def endpoint_correspondences(self) -> DenseCorrespondences:
        # This shared pixel-pair IR is not exclusively neural. Here the score is
        # minimum local patch correlation, not neural confidence or probability.
        return DenseCorrespondences(
            points0=self.pixels[0],
            points1=self.pixels[-1],
            confidence=self.minimum_patch_correlations,
            image_shape0=self.image_shape,
            image_shape1=self.image_shape,
            raw_match_count=self.seed_count,
        )


def track_frame_chain(
    frames: Sequence[np.ndarray],
    *,
    config: TemporalTrackConfig | None = None,
    seed_pixels: Sequence[Pixel] | None = None,
) -> TemporalTracks:
    """Track only first-frame points; validate adjacent and whole-chain cycles.

    Caller supplies temporally adjacent observations in order. Images deliberately
    repeated for identity controls are allowed; observation slots are not poses.
    No action intent, previous semantic identity or privileged state is consumed.
    """
    config = config or TemporalTrackConfig()
    if not 2 <= len(frames) <= config.maximum_frames:
        raise ValueError(
            "temporal chain requires a bounded number of at least two images"
        )
    cv2 = _opencv()
    grays_offsets = [_crop_gray(frame, cv2) for frame in frames]
    image_shape = tuple(np.asarray(frames[0]).shape[:2])
    if any(tuple(np.asarray(frame).shape[:2]) != image_shape for frame in frames):
        raise ValueError("temporal chain images must have identical dimensions")
    grays = [
        cv2.medianBlur(gray, 3) if config.median_filter else gray
        for gray, _ in grays_offsets
    ]
    offset = grays_offsets[0][1]
    scale = image_shape[1] / 224
    window, patch = (
        _odd(config.reference_window_pixels * scale),
        _odd(config.reference_patch_pixels * scale),
    )
    margin = max(window, patch) / 2 + 1

    def inside(points):
        return (
            np.isfinite(points).all(axis=1)
            & (points >= margin).all(axis=1)
            & (points[:, 0] < grays[0].shape[1] - margin)
            & (points[:, 1] < grays[0].shape[0] - margin)
        )

    if seed_pixels is None:
        detected = cv2.goodFeaturesToTrack(
            grays[0],
            maxCorners=config.feature_budget,
            qualityLevel=config.corner_quality,
            minDistance=config.reference_feature_spacing * scale,
        )
        seeds = (
            np.empty((0, 2), dtype=np.float32)
            if detected is None
            else detected.reshape(-1, 2)
        )
    else:
        seeds = np.asarray(seed_pixels, dtype=np.float32)
        if seeds.size == 0:
            seeds = seeds.reshape(0, 2)
        if (
            seeds.ndim != 2
            or seeds.shape[1] != 2
            or not np.isfinite(seeds).all()
            or len(seeds) > config.feature_budget
        ):
            raise ValueError(
                "explicit seed pixels must be finite pairs within feature budget"
            )
        seeds = seeds - offset.astype(np.float32)
    seed_count = len(seeds)
    ids = np.flatnonzero(inside(seeds))
    history = [seeds[ids].copy()]
    scores = np.ones(len(ids))
    counts = [len(ids)]
    kwargs = {
        "winSize": (window, window),
        "maxLevel": config.pyramid_levels,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01),
    }

    for previous_gray, current_gray in pairwise(grays):
        if not len(ids):
            history.append(np.empty((0, 2), dtype=np.float32))
            counts.append(0)
            continue
        previous = history[-1].reshape(-1, 1, 2)
        current, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, previous, None, **kwargs
        )
        if current is None or status is None:
            valid = np.zeros(len(ids), dtype=bool)
            current = np.zeros_like(previous)
        else:
            valid = (status.ravel() > 0) & inside(current.reshape(-1, 2))
            valid &= (
                np.linalg.norm(current.reshape(-1, 2) - previous.reshape(-1, 2), axis=1)
                <= config.reference_maximum_step * scale
            )
        selected = np.flatnonzero(valid)
        if len(selected):
            reverse, back_status, _ = cv2.calcOpticalFlowPyrLK(
                current_gray, previous_gray, current[selected], None, **kwargs
            )
            if reverse is None or back_status is None:
                valid[:] = False
            else:
                valid[selected] &= (back_status.ravel() > 0) & inside(
                    reverse.reshape(-1, 2)
                )
                valid[selected] &= (
                    np.linalg.norm(
                        reverse.reshape(-1, 2) - previous[selected].reshape(-1, 2),
                        axis=1,
                    )
                    <= config.reference_cycle_error * scale
                )
        correlations = np.zeros(len(ids))
        for index in np.flatnonzero(valid):
            left = (
                cv2.getRectSubPix(
                    previous_gray, (patch, patch), tuple(map(float, previous[index, 0]))
                )
                .astype(float)
                .ravel()
            )
            right = (
                cv2.getRectSubPix(
                    current_gray, (patch, patch), tuple(map(float, current[index, 0]))
                )
                .astype(float)
                .ravel()
            )
            if min(float(left.std()), float(right.std())) < config.minimum_patch_std:
                valid[index] = False
                continue
            left, right = left - left.mean(), right - right.mean()
            correlations[index] = float(
                np.clip(
                    left @ right / (np.linalg.norm(left) * np.linalg.norm(right)), -1, 1
                )
            )
            valid[index] &= correlations[index] >= config.minimum_patch_correlation
        ids, scores = ids[valid], np.minimum(scores[valid], correlations[valid])
        history = [points[valid] for points in history]
        history.append(current.reshape(-1, 2)[valid])
        counts.append(len(ids))

    # Reverse through EVERY intermediate image, not a shortcut endpoint solve.
    # No failed frame is skipped and no replacement corner can fill its identity.
    reverse = history[-1].reshape(-1, 1, 2).copy()
    valid = np.ones(len(ids), dtype=bool)
    for current_gray, previous_gray in zip(grays[:0:-1], grays[-2::-1]):
        selected = np.flatnonzero(valid)
        if not len(selected):
            break
        returned, status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, reverse[selected], None, **kwargs
        )
        if returned is None or status is None:
            valid[:] = False
            break
        good = (status.ravel() > 0) & inside(returned.reshape(-1, 2))
        valid[selected] &= good
        reverse[selected[good]] = returned[good]
    errors = np.linalg.norm(reverse.reshape(-1, 2) - history[0], axis=1)
    valid &= np.isfinite(errors) & (
        errors <= config.reference_chain_cycle_error * scale
    )
    return TemporalTracks(
        image_shape=image_shape,
        frame_count=len(frames),
        seed_count=seed_count,
        surviving_counts=tuple(counts),
        track_ids=tuple(int(value) for value in ids[valid]),
        pixels=tuple(tuple(map(tuple, points[valid] + offset)) for points in history),
        minimum_patch_correlations=tuple(scores[valid]),
        chain_cycle_errors=tuple(errors[valid]),
    )
