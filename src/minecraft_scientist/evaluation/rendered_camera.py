"""Evaluator-only pinhole decomposition of actual world-render matrices.

The public scientist never receives these matrices, camera centers or intrinsics.
Shader/shadow frames are unsupported. Matrix consistency is not navigation proof.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .relative_motion import score_motion_truth

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Vector3 = tuple[Number, Number, Number]
Matrix3 = tuple[Vector3, Vector3, Vector3]
Vector4 = tuple[Number, Number, Number, Number]
Matrix4 = tuple[Vector4, Vector4, Vector4, Vector4]


class WorldRenderSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1, le=10000, strict=True)
    partial_ticks: Number = Field(ge=0, le=1)
    model_view: Matrix4
    projection: Matrix4
    base_position: Vector3
    shader_pack_loaded: bool = Field(strict=True)
    shadow_pass: bool = Field(strict=True)

    @model_validator(mode="after")
    def _view(self):
        if self.shader_pack_loaded or self.shadow_pass:
            raise ValueError("shader/shadow world matrices are unsupported")
        view = np.asarray(self.model_view)
        rotation = view[:3, :3]
        if (
            not np.allclose(view[3], (0, 0, 0, 1), atol=1e-7)
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
        ):
            raise ValueError("world model-view must be rigid and proper")
        if abs(np.linalg.det(np.asarray(self.projection))) < 1e-12:
            raise ValueError("world projection is singular")
        return self


class RenderedCamera(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    render_sequence: int = Field(ge=1, le=10000, strict=True)
    rotation: Matrix3
    center: Vector3
    camera: Matrix3
    camera_center_offset: Vector3
    reconstruction_error: Number = Field(ge=0, le=1e-6)
    navigation_accepted: Literal[False] = False

    @model_validator(mode="after")
    def _proper(self):
        rotation, camera = np.array(self.rotation), np.array(self.camera)
        if (
            not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7)
            or camera[0, 0] <= 0
            or camera[1, 1] <= 0
            or not np.allclose(camera[2], (0, 0, 1), atol=1e-7)
            or not np.allclose(np.tril(camera, -1), 0, atol=1e-7)
        ):
            raise ValueError(
                "rendered camera requires proper rotation and positive upper-triangular intrinsics"
            )
        return self


def decompose_rendered_camera(snapshot, *, observation_size):
    """Convert row-major GL clip/view matrices to edge-origin OpenCV pixels.

    World vertices are relative to ActiveRenderInfo's base position before the
    world model-view. Effects already in the projection are retained, not removed.
    RQ is computed with NumPy QR; no optional simulator/OpenCV dependency is used.
    """
    snapshot = WorldRenderSnapshot.model_validate(
        snapshot.model_dump() if isinstance(snapshot, WorldRenderSnapshot) else snapshot
    )
    if len(observation_size) != 2 or any(
        type(value) is not int or not 32 <= value <= 4096 for value in observation_size
    ):
        raise ValueError(
            "rendered camera observation size must be bounded integer width/height"
        )
    width, height = observation_size
    viewport = np.array(
        ((width / 2, 0, 0, width / 2), (0, -height / 2, 0, height / 2), (0, 0, 0, 1))
    )
    projection = (
        viewport @ np.array(snapshot.projection) @ np.array(snapshot.model_view)
    )
    matrix, shift = projection[:, :3], projection[:, 3]
    if abs(np.linalg.det(matrix)) < 1e-12 or np.linalg.cond(matrix) > 1e10:
        raise ValueError("rendered pixel projection is unobservable or ill-conditioned")
    if np.linalg.det(matrix) < 0:
        raise ValueError("rendered pixel projection has reversed depth handedness")
    q, r = np.linalg.qr(np.flipud(matrix).T)
    camera, rotation = np.fliplr(np.flipud(r.T)), np.flipud(q.T)
    signs = np.diag(np.where(np.diag(camera) < 0, -1.0, 1.0))
    camera, rotation = camera @ signs, signs @ rotation
    scale = camera[2, 2]
    camera /= scale
    offset = -np.linalg.solve(matrix, shift)
    rebuilt = scale * camera @ np.column_stack((rotation, -rotation @ offset))
    error = float(
        np.linalg.norm(rebuilt - projection) / max(np.linalg.norm(projection), 1)
    )
    if not np.isfinite(error) or error > 1e-6:
        raise ValueError("rendered camera decomposition does not reconstruct clip rays")
    return RenderedCamera(
        render_sequence=snapshot.sequence,
        rotation=tuple(map(tuple, rotation)),
        camera=tuple(map(tuple, camera)),
        center=tuple(np.array(snapshot.base_position) + offset),
        camera_center_offset=tuple(offset),
        reconstruction_error=error,
    )


def rendered_motion_truth(first, second):
    first, second = [
        RenderedCamera.model_validate(
            item.model_dump() if isinstance(item, RenderedCamera) else item
        )
        for item in (first, second)
    ]
    if second.render_sequence <= first.render_sequence:
        raise ValueError("rendered motion requires chronological distinct renders")
    rotation = np.array(second.rotation)
    translation = rotation @ (np.array(first.center) - np.array(second.center))
    distance = float(np.linalg.norm(translation))
    return {
        "distance": distance,
        "relative_rotation": rotation @ np.array(first.rotation).T,
        "unit_translation": translation / distance if distance > 1e-9 else None,
    }


def score_rendered_motion(prediction, first, second, **thresholds):
    return score_motion_truth(
        prediction, rendered_motion_truth(first, second), **thresholds
    )
