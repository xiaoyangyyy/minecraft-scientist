"""Camera geometry for a publicly known full-frame resize pipeline.

The horizontal focal is learned from public image correspondences. No simulator
FOV is read. A square-pixel native pinhole raster becomes anisotropic when the
client stretches a wide render to a square observation.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

Dimension = Annotated[int, Field(strict=True, ge=32, le=16384)]


class CameraResizeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    render_size: tuple[Dimension, Dimension]
    observation_size: tuple[Dimension, Dimension]
    transform: Literal["full_frame_resize"] = "full_frame_resize"
    native_pixel_geometry: Literal["square_pixels"] = "square_pixels"
    principal_point: Literal["pixel_grid_center"] = "pixel_grid_center"

    def camera(self, horizontal_focal_pixels: float) -> np.ndarray:
        focal = FocalCalibration(
            horizontal_focal_pixels=horizontal_focal_pixels
        ).horizontal_focal_pixels
        render_width, render_height = self.render_size
        width, _ = self.observation_size
        native_focal = focal * render_width / width
        native = np.array(
            (
                (native_focal, 0, (render_width - 1) / 2),
                (0, native_focal, (render_height - 1) / 2),
                (0, 0, 1),
            )
        )
        return resize_intrinsics(native, self.render_size, self.observation_size)


class FocalCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    horizontal_focal_pixels: FiniteFloat = Field(gt=0)


def resize_intrinsics(
    camera: np.ndarray, source_size: tuple[int, int], target_size: tuple[int, int]
) -> np.ndarray:
    """Apply full-frame pixel-center resizing independently along both axes."""
    spec = CameraResizeSpec(render_size=source_size, observation_size=target_size)
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
        raise ValueError("resize requires finite zero-skew pinhole intrinsics")
    sx, sy = np.asarray(spec.observation_size) / np.asarray(spec.render_size)
    transform = np.array(((sx, 0, (sx - 1) / 2), (0, sy, (sy - 1) / 2), (0, 0, 1)))
    return transform @ camera
