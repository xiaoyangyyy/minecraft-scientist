"""Optional learned RGB correspondences with a strict public-pixel boundary.

LoFTR is not a navigation oracle. Its scores are uncalibrated match scores, and
the output must undergo independent geometry and motion validation. Checkpoints
are explicit local files; importing or calling this module never downloads them.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .visual_geometry import _frame, _opencv

Score = Annotated[FiniteFloat, Field(ge=0, le=1)]
Pixel = tuple[FiniteFloat, FiniteFloat]


class DenseCorrespondenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    working_dimension: int = Field(default=320, ge=64, le=640)
    minimum_confidence: float = Field(default=0.5, ge=0, le=1)
    maximum_matches: int = Field(default=1024, ge=16, le=8192)
    minimum_pixel_spacing: float = Field(default=2.0, gt=0)
    equalize_contrast: bool = True


class DenseCorrespondences(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    points0: tuple[Pixel, ...]
    points1: tuple[Pixel, ...]
    confidence: tuple[Score, ...]
    image_shape0: tuple[int, int]
    image_shape1: tuple[int, int]
    raw_match_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _aligned(self) -> DenseCorrespondences:
        if len(self.points0) != len(self.points1) or len(self.points0) != len(
            self.confidence
        ):
            raise ValueError("dense correspondence lengths differ")
        if self.raw_match_count < len(self.points0):
            raise ValueError("retained correspondences exceed raw match count")
        for points, shape in (
            (self.points0, self.image_shape0),
            (self.points1, self.image_shape1),
        ):
            height, width = shape
            if min(shape) < 32:
                raise ValueError("dense correspondence images must be at least 32x32")
            if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
                raise ValueError("dense correspondence lies outside its image")
        return self


@dataclass(frozen=True, slots=True)
class DenseView:
    gray: np.ndarray
    mask: np.ndarray
    original_shape: tuple[int, int]
    offset: tuple[int, int]
    crop_size: tuple[int, int]
    resized_size: tuple[int, int]

    def restore_pixels(self, points: np.ndarray) -> np.ndarray:
        scale = np.asarray(self.resized_size) / np.asarray(self.crop_size)
        return (points + 0.5) / scale - 0.5 + np.asarray(self.offset)


def prepare_dense_view(
    frame: np.ndarray, config: DenseCorrespondenceConfig
) -> DenseView:
    cv2 = _opencv()
    rgb = _frame(frame)
    height, width = rgb.shape[:2]
    x0, x1 = int(width * 0.18), int(width * 0.82)
    y0, y1 = int(height * 0.2), int(height * 0.76)
    # Crop before any neural or classical preprocessing. Outside HUD/text cannot
    # affect histograms, interpolation, convolution or cross-image attention.
    gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    if config.equalize_contrast:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    crop_height, crop_width = gray.shape
    factor = config.working_dimension / max(gray.shape)
    resized_width, resized_height = (
        max(8, round(crop_width * factor)),
        max(8, round(crop_height * factor)),
    )
    gray = cv2.resize(
        gray, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    padded_width, padded_height = (
        ((resized_width + 7) // 8) * 8,
        ((resized_height + 7) // 8) * 8,
    )
    image = np.zeros((padded_height, padded_width), dtype=np.float32)
    image[:resized_height, :resized_width] = gray.astype(np.float32) / 255
    mask = np.zeros_like(image, dtype=bool)
    mask[:resized_height, :resized_width] = True
    return DenseView(
        image,
        mask,
        (height, width),
        (x0, y0),
        (crop_width, crop_height),
        (resized_width, resized_height),
    )


def filter_dense_matches(
    raw: Mapping[str, np.ndarray],
    left: DenseView,
    right: DenseView,
    config: DenseCorrespondenceConfig,
) -> DenseCorrespondences:
    first, second = (
        np.asarray(raw["keypoints0"], dtype=float),
        np.asarray(raw["keypoints1"], dtype=float),
    )
    confidence = np.asarray(raw["confidence"], dtype=float)
    if (
        first.ndim != 2
        or first.shape[1:] != (2,)
        or second.shape != first.shape
        or confidence.shape != (len(first),)
    ):
        raise ValueError("dense backend output dimensions differ")
    valid = (
        np.isfinite(first).all(axis=1)
        & np.isfinite(second).all(axis=1)
        & np.isfinite(confidence)
    )
    valid &= (confidence >= config.minimum_confidence) & (confidence <= 1)
    for points, view in ((first, left), (second, right)):
        width, height = view.resized_size
        valid &= (points[:, 0] >= 0) & (points[:, 0] <= width - 1)
        valid &= (points[:, 1] >= 0) & (points[:, 1] <= height - 1)
    restored0, restored1 = left.restore_pixels(first), right.restore_pixels(second)
    for points, view in ((restored0, left), (restored1, right)):
        x0, y0 = view.offset
        width, height = view.crop_size
        valid &= (points[:, 0] >= x0 + 1) & (points[:, 0] < x0 + width - 1)
        valid &= (points[:, 1] >= y0 + 1) & (points[:, 1] < y0 + height - 1)
    indices = sorted(
        np.flatnonzero(valid),
        key=lambda index: (-confidence[index], *restored0[index], *restored1[index]),
    )
    retained, occupied0, occupied1 = [], set(), set()
    for index in indices:
        key0 = tuple(
            np.floor(restored0[index] / config.minimum_pixel_spacing).astype(int)
        )
        key1 = tuple(
            np.floor(restored1[index] / config.minimum_pixel_spacing).astype(int)
        )
        if key0 in occupied0 or key1 in occupied1:
            continue
        occupied0.add(key0)
        occupied1.add(key1)
        retained.append(index)
        if len(retained) >= config.maximum_matches:
            break
    return DenseCorrespondences(
        points0=tuple(map(tuple, restored0[retained])),
        points1=tuple(map(tuple, restored1[retained])),
        confidence=tuple(confidence[retained]),
        image_shape0=left.original_shape,
        image_shape1=right.original_shape,
        raw_match_count=len(first),
    )


class DenseRGBMatcher:
    def __init__(
        self,
        backend: Callable[[DenseView, DenseView], Mapping[str, np.ndarray]],
        *,
        config: DenseCorrespondenceConfig | None = None,
    ):
        self.backend = backend
        self.config = config or DenseCorrespondenceConfig()

    def match(self, first: np.ndarray, second: np.ndarray) -> DenseCorrespondences:
        left, right = (
            prepare_dense_view(first, self.config),
            prepare_dense_view(second, self.config),
        )
        return filter_dense_matches(self.backend(left, right), left, right, self.config)


class LocalLoFTRBackend:
    """CPU-only optional backend; verified bytes and safe tensor loading only."""

    def __init__(self, checkpoint: Path, *, sha256: str, threads: int = 4):
        if not re.fullmatch("[0-9a-f]{64}", sha256) or not 1 <= threads <= 16:
            raise ValueError("a SHA256 and bounded CPU thread count are required")
        if checkpoint.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("LoFTR checkpoint exceeds 128 MiB")
        checkpoint_bytes = checkpoint.read_bytes()
        if hashlib.sha256(checkpoint_bytes).hexdigest() != sha256:
            raise ValueError("LoFTR checkpoint hash mismatch")
        try:
            import kornia
            import torch
            from kornia.feature.loftr.loftr import LoFTR, default_cfg
        except ImportError as error:
            raise RuntimeError(
                "LoFTR requires optional Kornia and PyTorch runtimes"
            ) from error
        self.torch = torch
        self.threads = threads
        self.model = (
            LoFTR(pretrained=None, config=deepcopy(default_cfg)).eval().to("cpu")
        )
        payload = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.provenance = {
            "checkpoint_sha256": sha256,
            "checkpoint_bytes": len(checkpoint_bytes),
            "torch_version": torch.__version__,
            "kornia_version": kornia.__version__,
            "device": "cpu",
            "threads": threads,
            "backend_source_sha256": hashlib.sha256(
                Path(LoFTR.__init__.__code__.co_filename).read_bytes()
            ).hexdigest(),
        }

    def __call__(self, left: DenseView, right: DenseView) -> Mapping[str, np.ndarray]:
        torch = self.torch
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self.threads)
            with torch.inference_mode():
                output = self.model(
                    {
                        "image0": torch.from_numpy(left.gray)[None, None],
                        "image1": torch.from_numpy(right.gray)[None, None],
                        # Kornia 0.8.x resizes these masks before coercing them
                        # to boolean attention masks; torch cannot resize bool.
                        "mask0": torch.from_numpy(left.mask).float()[None],
                        "mask1": torch.from_numpy(right.mask).float()[None],
                    }
                )
            return {
                name: output[name].detach().cpu().numpy()
                for name in ("keypoints0", "keypoints1", "confidence")
            }
        finally:
            torch.set_num_threads(previous_threads)
