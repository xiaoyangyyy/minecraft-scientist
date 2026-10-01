"""Explicit public-RGB resolution normalization for remembered visual features."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SceneFeatureEncoder(Protocol):
    @property
    def dimensions(self) -> int: ...

    def encode(self, frame: np.ndarray) -> tuple[int, ...]: ...


class ResolutionNormalizedSceneEncoder:
    """Keep reference feature scale while preserving raw frames for geometry.

    Width and height come from public reference images, not simulator telemetry.
    Integer downsampling uses exact area averages. Other size ratios require the
    optional Pillow runtime. This does not assert viewpoint/terrain invariance.
    """

    def __init__(self, encoder: SceneFeatureEncoder, *, size: tuple[int, int]) -> None:
        if len(size) != 2 or any(type(value) is not int or value < 1 for value in size):
            raise ValueError(
                "reference resolution must contain positive integer dimensions"
            )
        self.encoder = encoder
        self.size = size

    @property
    def dimensions(self) -> int:
        return self.encoder.dimensions

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
            raise ValueError("resolution normalization requires finite RGB pixels")
        height, width = rgb.shape[:2]
        target_width, target_height = self.size
        if height == 0 or width == 0:
            raise ValueError("RGB frames cannot be empty")
        if (width, height) == self.size:
            return rgb.copy()
        if width % target_width == 0 and height % target_height == 0:
            pooled = rgb.reshape(
                target_height,
                height // target_height,
                target_width,
                width // target_width,
                3,
            ).mean(axis=(1, 3))
            return np.rint(np.clip(pooled, 0, 255)).astype(np.uint8)
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "noninteger resolution normalization requires Pillow"
            ) from error
        return np.asarray(
            Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize(
                self.size, Image.Resampling.BOX
            )
        )

    def encode(self, frame: np.ndarray) -> tuple[int, ...]:
        return self.encoder.encode(self.normalize(frame))
