"""Select background seeds using only bounded pre-intervention RGB context."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .correspondence_refinement import _crop_gray, _odd
from .temporal_tracks import TemporalTrackConfig, track_frame_chain
from .visual_geometry import _opencv


class StationaryContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_frames: int = Field(default=3, ge=3, le=8, strict=True)
    reference_patch_pixels: int = Field(default=7, ge=3, le=21, strict=True)
    maximum_patch_range: FiniteFloat = Field(default=16, gt=0, le=255)
    maximum_patch_mad: FiniteFloat = Field(default=4, gt=0, le=128)
    reference_stationary_displacement: FiniteFloat = Field(default=0.5, gt=0, le=2)
    temporal: TemporalTrackConfig = TemporalTrackConfig()

    @model_validator(mode="after")
    def _budget(self):
        if self.context_frames + 1 > self.temporal.maximum_frames:
            raise ValueError(
                "context and virtual initialization exceed the tracker frame budget"
            )
        return self


class StationarySeeds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_shape: tuple[int, int]
    context_frames: int = Field(ge=3, le=8)
    detector_count: int = Field(ge=0, le=2000)
    context_consistent_count: int = Field(ge=0, le=2000)
    seed_ids: tuple[int, ...]
    pixels: tuple[tuple[FiniteFloat, FiniteFloat], ...]
    maximum_displacements: tuple[FiniteFloat, ...]
    patch_range_mean: tuple[FiniteFloat, ...]
    patch_mad_mean: tuple[FiniteFloat, ...]

    @model_validator(mode="after")
    def _aligned(self):
        count = len(self.seed_ids)
        if not count <= self.context_consistent_count <= self.detector_count:
            raise ValueError(
                "context selections cannot exceed the original detector budget"
            )
        if len(set(self.seed_ids)) != count or any(
            not 0 <= index < self.detector_count for index in self.seed_ids
        ):
            raise ValueError(
                "seed provenance must preserve original detector identities"
            )
        if any(
            len(values) != count
            for values in (
                self.pixels,
                self.maximum_displacements,
                self.patch_range_mean,
                self.patch_mad_mean,
            )
        ):
            raise ValueError("stationary context statistics must align with seeds")
        height, width = self.image_shape
        if min(self.image_shape) < 32 or any(
            not 0 <= x < width or not 0 <= y < height for x, y in self.pixels
        ):
            raise ValueError("background seed outside the public image")
        if any(
            value < 0
            for values in (
                self.maximum_displacements,
                self.patch_range_mean,
                self.patch_mad_mean,
            )
            for value in values
        ):
            raise ValueError("context statistics must be nonnegative")
        return self


def stationary_background_seeds(
    frames: Sequence[np.ndarray], *, config: StationaryContextConfig | None = None
) -> StationarySeeds:
    """Return last-real-frame pixels, never coordinates in a synthetic image.

    A temporal median is used only for detection. Every point must cycle through
    all real context images with bounded displacement. This is pixel stationarity,
    not proof that the body is motionless or that the environment is static.
    Caller must supply past observations ending at the intervention's first frame.
    """
    config = config or StationaryContextConfig()
    if len(frames) != config.context_frames:
        raise ValueError(
            "stationary context must contain exactly its bounded frame budget"
        )
    cv2 = _opencv()
    images = [np.asarray(image) for image in frames]
    shape = images[0].shape
    if (
        len(shape) != 3
        or shape[2] != 3
        or any(image.shape != shape or image.dtype != np.uint8 for image in images)
    ):
        raise ValueError("stationary context requires equal-size uint8 RGB images")
    cropped = [_crop_gray(image, cv2) for image in images]
    grays = np.stack(
        [
            cv2.medianBlur(gray, 3) if config.temporal.median_filter else gray
            for gray, _ in cropped
        ]
    ).astype(np.float32)
    median = np.median(grays, axis=0)
    variation = np.ptp(grays, axis=0)
    mad = np.median(np.abs(grays - median), axis=0)
    scale = shape[1] / 224
    patch = _odd(config.reference_patch_pixels * scale)
    ranges = cv2.boxFilter(variation, -1, (patch, patch), normalize=True)
    deviations = cv2.boxFilter(mad, -1, (patch, patch), normalize=True)
    mask = (
        (ranges <= config.maximum_patch_range)
        & (deviations <= config.maximum_patch_mad)
    ).astype(np.uint8) * 255
    detected = cv2.goodFeaturesToTrack(
        median.astype(np.uint8),
        maxCorners=config.temporal.feature_budget,
        qualityLevel=config.temporal.corner_quality,
        minDistance=config.temporal.reference_feature_spacing * scale,
        mask=mask,
    )
    points = (
        np.empty((0, 2), dtype=np.float32)
        if detected is None
        else detected.reshape(-1, 2)
    )
    offset = cropped[0][1]
    # Median has no physical pose: it is a virtual detector/track initialization
    # slot only and is never passed to geometric motion estimation.
    virtual = np.median(np.stack(images), axis=0).astype(np.uint8)
    tracks = track_frame_chain(
        [virtual, *images],
        # Context assumes subpixel stationarity, not arbitrary displacement.
        # Coarse pyramids on the small ROI let a large moving foreground steer
        # otherwise static background seeds. Validate locally at native scale;
        # subsequent intervention tracking still uses the full declared pyramid.
        config=config.temporal.model_copy(update={"pyramid_levels": 0}),
        seed_pixels=tuple(map(tuple, points + offset)),
    )
    pixels = np.asarray(tracks.pixels, dtype=float).reshape(len(images) + 1, -1, 2)
    motion = np.linalg.norm(pixels - pixels[-1:], axis=2).max(axis=0)
    selected = np.isfinite(motion) & (
        motion <= config.reference_stationary_displacement * scale
    )
    ids = np.asarray(tracks.track_ids, dtype=int)[selected]
    locations = np.rint(points[ids]).astype(int)
    return StationarySeeds(
        image_shape=shape[:2],
        context_frames=len(images),
        detector_count=len(points),
        context_consistent_count=len(tracks.track_ids),
        seed_ids=tuple(map(int, ids)),
        pixels=tuple(map(tuple, pixels[-1, selected])),
        maximum_displacements=tuple(motion[selected]),
        patch_range_mean=tuple(float(ranges[y, x]) for x, y in locations),
        patch_mad_mean=tuple(float(deviations[y, x]) for x, y in locations),
    )
