"""Label-free fixed-width scene descriptors for agent-visible observations."""

from __future__ import annotations

import json
import math

import numpy as np

from minecraft_scientist.protocols import ActionKind, Observation


class AnonymousSceneEncoder:
    """Encode local solid geometry without IDs, names, or absolute coordinates.

    The descriptor intentionally uses only public numeric block features. It is
    invariant to opaque-ID renaming, object order, and whole-cell joint translation
    of the agent and scene. It is not a pixel encoder and must not be presented as
    one.
    """

    DIMENSIONS = 34

    def __init__(self, *, cell_size: int = 4, extent: int = 4) -> None:
        if cell_size < 1 or extent < 1:
            raise ValueError("scene cell size and extent must be positive")
        self.cell_size = cell_size
        self.extent = extent

    def encode(self, observation: Observation) -> tuple[int, ...]:
        origin = tuple(float(observation.agent_state[axis]) for axis in ("x", "y", "z"))
        anchor = tuple(
            math.floor(value / self.cell_size) * self.cell_size + self.cell_size / 2
            for value in origin
        )
        blocks = [
            item
            for item in observation.objects
            if item.affordances != (ActionKind.MOVE,)
            and "diggable" in item.observable_features
            and all(axis in item.observable_features for axis in ("x", "y", "z"))
            and all(
                abs(float(item.observable_features[axis]) - anchor[index])
                <= self.extent
                for index, axis in enumerate(("x", "y", "z"))
            )
        ]
        radial = [0] * 8
        vertical = [0] * 9
        sectors = [0] * 8
        hardness = [0] * 6
        diggable = [0] * 2
        for item in blocks:
            features = item.observable_features
            dx = float(features["x"]) - anchor[0]
            dy = float(features["y"]) - anchor[1]
            dz = float(features["z"]) - anchor[2]
            horizontal = math.hypot(dx, dz)
            radial[min(7, int(horizontal // 2))] += 1
            vertical[min(8, max(0, round(dy) + 4))] += 1
            quadrant = (int(dx >= 0) << 1) | int(dz >= 0)
            sectors[quadrant * 2 + int(horizontal > 4)] += 1
            value = features.get("hardness")
            if value is None or float(value) < 0:
                hardness_bin = 0
            elif float(value) < 0.5:
                hardness_bin = 1
            elif float(value) < 1.5:
                hardness_bin = 2
            elif float(value) < 3:
                hardness_bin = 3
            elif float(value) < 5:
                hardness_bin = 4
            else:
                hardness_bin = 5
            hardness[hardness_bin] += 1
            diggable[int(bool(features["diggable"]))] += 1
        descriptor = (
            len(blocks),
            *radial,
            *vertical,
            *sectors,
            *hardness,
            *diggable,
        )
        if len(descriptor) != self.DIMENSIONS:
            raise AssertionError("anonymous scene descriptor width changed")
        return descriptor


class RGBSceneEncoder:
    """Encode RGB appearance with a normalized central CIELAB spatial grid.

    The encoder is deterministic, label-free, and has no learned weights. A central
    crop suppresses HUD and peripheral scenery; per-channel normalization reduces
    illumination shifts while retaining obstacle layout. Calibration remains
    mandatory because this hand-built metric is not universally invariant.
    """

    VERSION = "rgb-lab-grid-v1"

    def __init__(
        self,
        *,
        grid_size: int = 16,
        horizontal_crop: float = 0.18,
        bottom_fraction: float = 0.76,
        quantization: float = 8,
        stabilization: float = 2,
    ) -> None:
        if grid_size < 2:
            raise ValueError("RGB grid size must be at least two")
        if not 0 <= horizontal_crop < 0.5:
            raise ValueError("RGB horizontal crop must be in [0, 0.5)")
        if not 0 < bottom_fraction <= 1:
            raise ValueError("RGB bottom fraction must be in (0, 1]")
        if quantization <= 0 or stabilization <= 0:
            raise ValueError("RGB quantization and stabilization must be positive")
        self.grid_size = grid_size
        self.horizontal_crop = float(horizontal_crop)
        self.bottom_fraction = float(bottom_fraction)
        self.quantization = float(quantization)
        self.stabilization = float(stabilization)

    @property
    def dimensions(self) -> int:
        return self.grid_size * self.grid_size * 3

    def encode(self, frame: np.ndarray) -> tuple[int, ...]:
        array = np.asarray(frame)
        if (
            array.ndim != 3
            or array.shape[2] < 3
            or array.shape[0] < 2
            or array.shape[1] < 2
        ):
            raise ValueError("RGB frame is too small or has invalid dimensions")
        rgb = np.asarray(array[:, :, :3], dtype=np.float64)
        if not np.isfinite(rgb).all():
            raise ValueError("RGB frame must contain finite values")
        rgb = np.clip(rgb, 0, 255)
        height, width = rgb.shape[:2]
        left = int(width * self.horizontal_crop)
        right = int(width * (1 - self.horizontal_crop))
        bottom = int(height * self.bottom_fraction)
        crop = rgb[:bottom, left:right]
        if crop.shape[0] < self.grid_size or crop.shape[1] < self.grid_size:
            y_indices = np.linspace(0, crop.shape[0] - 1, self.grid_size).astype(int)
            x_indices = np.linspace(0, crop.shape[1] - 1, self.grid_size).astype(int)
            crop = crop[y_indices][:, x_indices]
        pooled = self._area_pool(self._rgb_to_lab(crop))
        mean = pooled.mean(axis=(0, 1), keepdims=True)
        scale = pooled.std(axis=(0, 1), keepdims=True) + self.stabilization
        normalized = (pooled - mean) / scale
        signature = tuple(
            int(value)
            for value in np.rint(normalized * self.quantization)
            .astype(np.int16)
            .ravel()
        )
        if len(signature) != self.dimensions:
            raise AssertionError("RGB scene descriptor width changed")
        return signature

    def serialize(self, signature: tuple[int, ...]) -> str:
        if len(signature) != self.dimensions:
            raise ValueError("RGB signature width does not match encoder")
        return json.dumps(signature, separators=(",", ":"))

    def deserialize(self, payload: str) -> tuple[int, ...]:
        try:
            signature = tuple(int(value) for value in json.loads(payload))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid serialized RGB signature") from error
        if len(signature) != self.dimensions:
            raise ValueError("serialized RGB signature width does not match encoder")
        return signature

    def _area_pool(self, array: np.ndarray) -> np.ndarray:
        y_edges = np.linspace(0, array.shape[0], self.grid_size + 1).astype(int)
        x_edges = np.linspace(0, array.shape[1], self.grid_size + 1).astype(int)
        return np.stack(
            [
                [
                    array[
                        y_edges[row] : y_edges[row + 1],
                        x_edges[column] : x_edges[column + 1],
                    ].mean(axis=(0, 1))
                    for column in range(self.grid_size)
                ]
                for row in range(self.grid_size)
            ]
        )

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        srgb = rgb / 255
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            ((srgb + 0.055) / 1.055) ** 2.4,
        )
        matrix = np.asarray(
            (
                (0.4124564, 0.3575761, 0.1804375),
                (0.2126729, 0.7151522, 0.0721750),
                (0.0193339, 0.1191920, 0.9503041),
            )
        )
        xyz = linear @ matrix.T
        xyz /= np.asarray((0.95047, 1.0, 1.08883))
        delta = 6 / 29
        transformed = np.where(
            xyz > delta**3,
            np.cbrt(xyz),
            xyz / (3 * delta**2) + 4 / 29,
        )
        return np.stack(
            (
                116 * transformed[..., 1] - 16,
                500 * (transformed[..., 0] - transformed[..., 1]),
                200 * (transformed[..., 1] - transformed[..., 2]),
            ),
            axis=-1,
        )


def rgb_signature_from_observation(
    observation: Observation,
    encoder: RGBSceneEncoder | None = None,
) -> tuple[int, ...]:
    """Read and validate the public RGB signature produced by a visual adapter."""

    selected = encoder or RGBSceneEncoder()
    if observation.agent_state.get("appearance_encoder") != selected.VERSION:
        raise ValueError("observation uses an unsupported appearance encoder")
    if observation.agent_state.get("appearance_dimensions") != selected.dimensions:
        raise ValueError("observation appearance dimensions are inconsistent")
    payload = observation.agent_state.get("appearance_signature")
    if not isinstance(payload, str):
        raise TypeError("observation is missing its appearance signature")
    return selected.deserialize(payload)
