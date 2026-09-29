"""Label-free fixed-width scene descriptors for agent-visible observations."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minecraft_scientist.protocols import ActionKind, Observation


class PanoramaAlignment(BaseModel):
    """RGB-only cyclic alignment from a query panorama to a stored heading."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_offset: int = Field(ge=0)
    right_turn_steps: int = Field(ge=0)
    mean_distance: float = Field(ge=0)
    second_best_distance: float = Field(ge=0)
    separation: float = Field(ge=0)


def align_panorama_heading(
    reference: Iterable[Iterable[int]],
    query: Iterable[Iterable[int]],
) -> PanoramaAlignment:
    """Find the cyclic RGB alignment needed to restore a learned heading.

    If query view zero corresponds to reference view ``k``, turning right by
    ``period - k`` steps restores reference view zero.  The full panorama is used
    for scoring, so the result does not depend on one potentially ambiguous frame.
    """

    stored = tuple(tuple(int(value) for value in item) for item in reference)
    observed = tuple(tuple(int(value) for value in item) for item in query)
    if len(stored) < 2 or len(observed) != len(stored):
        raise ValueError("panorama alignment requires equal multi-view periods")
    dimensions = len(stored[0])
    if dimensions < 1 or any(
        len(item) != dimensions for item in (*stored, *observed)
    ):
        raise ValueError("panorama alignment views must have equal nonzero width")
    period = len(stored)
    scores = []
    for offset in range(period):
        score = sum(
            math.dist(observed[index], stored[(index + offset) % period])
            for index in range(period)
        ) / period
        scores.append((score, offset))
    ranked = sorted(scores)
    best_distance, offset = ranked[0]
    second_best = ranked[1][0]
    return PanoramaAlignment(
        reference_offset=offset,
        right_turn_steps=(-offset) % period,
        mean_distance=best_distance,
        second_best_distance=second_best,
        separation=second_best - best_distance,
    )


def temporal_median_rgb(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Suppress transient RGB artifacts with an odd-frame pixelwise median."""

    arrays = tuple(np.asarray(frame) for frame in frames)
    if not arrays or len(arrays) % 2 == 0:
        raise ValueError("temporal RGB aggregation requires a positive odd sample count")
    shape = arrays[0].shape
    if (
        len(shape) != 3
        or shape[2] < 3
        or any(array.shape != shape for array in arrays)
        or any(not np.isfinite(array).all() for array in arrays)
    ):
        raise ValueError("temporal RGB samples must be finite equal-shaped images")
    return np.median(np.stack(arrays), axis=0).astype(np.uint8)


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

    VERSION = "rgb-lab-grid-v2"

    def __init__(
        self,
        *,
        grid_size: int = 16,
        horizontal_crop: float = 0.18,
        top_fraction: float = 0.2,
        bottom_fraction: float = 0.76,
        quantization: float = 8,
        stabilization: float = 2,
    ) -> None:
        if grid_size < 2:
            raise ValueError("RGB grid size must be at least two")
        if not 0 <= horizontal_crop < 0.5:
            raise ValueError("RGB horizontal crop must be in [0, 0.5)")
        if not 0 <= top_fraction < bottom_fraction <= 1:
            raise ValueError(
                "RGB vertical fractions must satisfy 0 <= top < bottom <= 1"
            )
        if quantization <= 0 or stabilization <= 0:
            raise ValueError("RGB quantization and stabilization must be positive")
        self.grid_size = grid_size
        self.horizontal_crop = float(horizontal_crop)
        self.top_fraction = float(top_fraction)
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
        top = int(height * self.top_fraction)
        bottom = int(height * self.bottom_fraction)
        crop = rgb[top:bottom, left:right]
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


class RGBEdgeSceneEncoder:
    """Encode illumination-robust scene structure from anonymous RGB edges.

    A small median filter suppresses rain streaks and isolated particles.  The
    remaining luminance gradients are converted to percentile-ranked strong-edge
    occupancy plus unsigned horizontal/vertical orientation channels.  Because the
    threshold is fitted independently per frame, global brightness and contrast do
    not enter the descriptor.
    """

    VERSION = "rgb-edge-grid-v2"

    def __init__(
        self,
        *,
        grid_size: int = 16,
        horizontal_crop: float = 0.18,
        top_fraction: float = 0.2,
        bottom_fraction: float = 0.76,
        edge_quantile: float = 0.7,
        quantization: float = 32,
    ) -> None:
        if grid_size < 2:
            raise ValueError("edge grid size must be at least two")
        if not 0 <= horizontal_crop < 0.5:
            raise ValueError("edge horizontal crop must be in [0, 0.5)")
        if not 0 <= top_fraction < bottom_fraction <= 1:
            raise ValueError(
                "edge vertical fractions must satisfy 0 <= top < bottom <= 1"
            )
        if not 0 < edge_quantile < 1:
            raise ValueError("edge quantile must be in (0, 1)")
        if quantization <= 0:
            raise ValueError("edge quantization must be positive")
        self.grid_size = grid_size
        self.horizontal_crop = float(horizontal_crop)
        self.top_fraction = float(top_fraction)
        self.bottom_fraction = float(bottom_fraction)
        self.edge_quantile = float(edge_quantile)
        self.quantization = float(quantization)

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
        height, width = rgb.shape[:2]
        left = int(width * self.horizontal_crop)
        right = int(width * (1 - self.horizontal_crop))
        top = int(height * self.top_fraction)
        bottom = int(height * self.bottom_fraction)
        crop = np.clip(rgb[top:bottom, left:right], 0, 255)
        if crop.shape[0] < self.grid_size or crop.shape[1] < self.grid_size:
            y_indices = np.linspace(0, crop.shape[0] - 1, self.grid_size).astype(int)
            x_indices = np.linspace(0, crop.shape[1] - 1, self.grid_size).astype(int)
            crop = crop[y_indices][:, x_indices]
        luminance = (
            0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]
        )
        padded = np.pad(luminance, 1, mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        smooth = np.median(windows, axis=(-2, -1))
        gradient_x = np.zeros_like(smooth)
        gradient_y = np.zeros_like(smooth)
        gradient_x[:, 1:-1] = smooth[:, 2:] - smooth[:, :-2]
        gradient_y[1:-1, :] = smooth[2:, :] - smooth[:-2, :]
        magnitude = np.hypot(gradient_x, gradient_y)
        threshold = float(np.quantile(magnitude, self.edge_quantile))
        strong = magnitude > max(threshold, 1e-9)
        scale = np.where(strong, 1 / np.maximum(magnitude, 1e-9), 0)
        channels = np.stack(
            (
                strong.astype(np.float64),
                np.abs(gradient_x) * scale,
                np.abs(gradient_y) * scale,
            ),
            axis=-1,
        )
        pooled = self._area_pool(channels)
        signature = tuple(
            int(value)
            for value in np.rint(pooled * self.quantization).astype(np.int16).ravel()
        )
        if len(signature) != self.dimensions:
            raise AssertionError("RGB edge descriptor width changed")
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


class RGBPanoramaEncoder:
    """Encode a full visual scan independently of its starting camera yaw.

    A place scan is represented by the per-coordinate mean and standard deviation
    of its single-view RGB signatures.  Both statistics are invariant to cyclic
    reordering (and, more generally, any ordering) of the captured headings.  This
    deliberately treats a panorama as a bag of views: it is useful for place
    retrieval but does not retain a compass direction or reconstruct geometry.
    """

    VERSION = "rgb-lab-panorama-v2"

    def __init__(
        self,
        scene_encoder: RGBSceneEncoder | RGBEdgeSceneEncoder | None = None,
    ) -> None:
        self.scene_encoder = scene_encoder or RGBSceneEncoder()

    @property
    def dimensions(self) -> int:
        return self.scene_encoder.dimensions * 2

    def encode(self, frames: Iterable[np.ndarray]) -> tuple[int, ...]:
        arrays = tuple(np.asarray(frame) for frame in frames)
        if len(arrays) < 2:
            raise ValueError("panorama encoding requires at least two views")
        return self.encode_signatures(
            tuple(self.scene_encoder.encode(frame) for frame in arrays)
        )

    def encode_signatures(self, signatures: Iterable[Iterable[int]]) -> tuple[int, ...]:
        items = tuple(tuple(int(value) for value in item) for item in signatures)
        if len(items) < 2:
            raise ValueError("panorama encoding requires at least two signatures")
        if any(len(item) != self.scene_encoder.dimensions for item in items):
            raise ValueError("panorama view signature width does not match encoder")
        matrix = np.asarray(items, dtype=np.float64)
        descriptor = tuple(
            int(value)
            for value in np.concatenate(
                (
                    np.rint(matrix.mean(axis=0)),
                    np.rint(matrix.std(axis=0)),
                )
            ).astype(np.int16)
        )
        if len(descriptor) != self.dimensions:
            raise AssertionError("RGB panorama descriptor width changed")
        return descriptor

    def serialize(self, signature: tuple[int, ...]) -> str:
        if len(signature) != self.dimensions:
            raise ValueError("RGB panorama signature width does not match encoder")
        return json.dumps(signature, separators=(",", ":"))

    def deserialize(self, payload: str) -> tuple[int, ...]:
        try:
            signature = tuple(int(value) for value in json.loads(payload))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid serialized RGB panorama signature") from error
        if len(signature) != self.dimensions:
            raise ValueError("serialized RGB panorama width does not match encoder")
        return signature


class RGBEdgePanoramaEncoder(RGBPanoramaEncoder):
    """Order-independent panorama built from illumination-robust edge views."""

    VERSION = "rgb-edge-panorama-v2"

    def __init__(self, scene_encoder: RGBEdgeSceneEncoder | None = None) -> None:
        super().__init__(scene_encoder or RGBEdgeSceneEncoder())


class RGBEdgeHistogramSceneEncoder:
    """Translation-tolerant edge distribution for one anonymous RGB view.

    The underlying edge encoder still removes illumination and transient pixel
    noise.  Spatial cells are then histogrammed per channel, deliberately trading
    exact screen position for tolerance to the sub-block viewpoint shifts that
    occur during natural-terrain homing.
    """

    VERSION = "rgb-edge-histogram-view-v1"

    def __init__(self, *, grid_size: int = 16, bin_count: int = 6) -> None:
        if bin_count < 2:
            raise ValueError("edge histogram requires at least two bins")
        self.edge_encoder = RGBEdgeSceneEncoder(grid_size=grid_size)
        self.grid_size = grid_size
        self.bin_count = bin_count

    @property
    def dimensions(self) -> int:
        return self.bin_count * 3

    def encode(self, frame: np.ndarray) -> tuple[int, ...]:
        edge = np.asarray(
            self.edge_encoder.encode(frame), dtype=np.int16
        ).reshape(self.grid_size, self.grid_size, 3)
        # RGBEdgeSceneEncoder quantizes all three channels into [0, 32].
        boundaries = np.linspace(0, 33, self.bin_count + 1)
        signature = tuple(
            int(value)
            for channel in range(3)
            for value in np.histogram(edge[:, :, channel], bins=boundaries)[0]
        )
        if len(signature) != self.dimensions:
            raise AssertionError("RGB edge histogram descriptor width changed")
        return signature


class RGBEdgeHistogramPanoramaEncoder(RGBPanoramaEncoder):
    """Yaw/order-invariant panorama over translation-tolerant edge histograms."""

    VERSION = "rgb-edge-histogram-panorama-v1"

    def __init__(
        self, scene_encoder: RGBEdgeHistogramSceneEncoder | None = None
    ) -> None:
        super().__init__(scene_encoder or RGBEdgeHistogramSceneEncoder())


class RGBEdgeSpectrumSceneEncoder:
    """Translation-invariant edge spectrum retaining coarse spatial structure."""

    VERSION = "rgb-edge-spectrum-view-v1"

    def __init__(
        self, *, grid_size: int = 16, low_frequency_radius: int = 4
    ) -> None:
        if grid_size < 2 or not 1 <= low_frequency_radius < grid_size // 2:
            raise ValueError("invalid edge spectrum grid or frequency radius")
        self.edge_encoder = RGBEdgeSceneEncoder(grid_size=grid_size)
        self.grid_size = grid_size
        self.low_frequency_radius = low_frequency_radius

    @property
    def dimensions(self) -> int:
        width = self.low_frequency_radius * 2 + 1
        return width * width * 3

    def encode(self, frame: np.ndarray) -> tuple[int, ...]:
        edge = np.asarray(
            self.edge_encoder.encode(frame), dtype=np.float64
        ).reshape(self.grid_size, self.grid_size, 3)
        center = self.grid_size // 2
        radius = self.low_frequency_radius
        features = []
        for channel in range(3):
            spectrum = np.fft.fftshift(np.abs(np.fft.fft2(edge[:, :, channel])))
            crop = np.log1p(
                spectrum[
                    center - radius : center + radius + 1,
                    center - radius : center + radius + 1,
                ]
            )
            normalized = crop / (float(np.linalg.norm(crop)) + 1e-9)
            features.extend(np.rint(normalized * 100).astype(np.int16).ravel())
        signature = tuple(int(value) for value in features)
        if len(signature) != self.dimensions:
            raise AssertionError("RGB edge spectrum descriptor width changed")
        return signature


class RGBEdgeSpectrumPanoramaEncoder(RGBPanoramaEncoder):
    """Yaw/order-invariant panorama over translation-invariant edge spectra."""

    VERSION = "rgb-edge-spectrum-panorama-v1"

    def __init__(
        self, scene_encoder: RGBEdgeSpectrumSceneEncoder | None = None
    ) -> None:
        super().__init__(scene_encoder or RGBEdgeSpectrumSceneEncoder())


class RGBEdgeCompactSpectrumSceneEncoder(RGBEdgeSpectrumSceneEncoder):
    """Lower-band edge spectrum for sub-block natural-terrain viewpoint shifts.

    The fixed radius is intentionally versioned separately from the wider spectrum:
    changing descriptor bandwidth changes the modality's calibrated geometry and
    must never silently reuse persisted place boundaries.
    """

    VERSION = "rgb-edge-spectrum-r3-view-v1"

    def __init__(self) -> None:
        super().__init__(low_frequency_radius=3)


class RGBEdgeCompactSpectrumPanoramaEncoder(RGBPanoramaEncoder):
    """Yaw-invariant panorama over the fixed compact edge spectrum."""

    VERSION = "rgb-edge-spectrum-r3-panorama-v1"

    def __init__(
        self, scene_encoder: RGBEdgeCompactSpectrumSceneEncoder | None = None
    ) -> None:
        super().__init__(scene_encoder or RGBEdgeCompactSpectrumSceneEncoder())


class RGBEdgeExpandedSpectrumSceneEncoder(RGBEdgeSpectrumSceneEncoder):
    """Higher-band edge spectrum for sloped-terrain arrival recognition."""

    VERSION = "rgb-edge-spectrum-r5-view-v1"

    def __init__(self) -> None:
        super().__init__(low_frequency_radius=5)


class RGBEdgeExpandedSpectrumPanoramaEncoder(RGBPanoramaEncoder):
    """Yaw-invariant panorama over the fixed expanded edge spectrum."""

    VERSION = "rgb-edge-spectrum-r5-panorama-v1"

    def __init__(
        self, scene_encoder: RGBEdgeExpandedSpectrumSceneEncoder | None = None
    ) -> None:
        super().__init__(scene_encoder or RGBEdgeExpandedSpectrumSceneEncoder())


class RGBEdgeDualBandSpectrumSceneEncoder:
    """Concatenate complementary global and structured edge-frequency bands."""

    VERSION = "rgb-edge-spectrum-r1-r6-view-v1"

    def __init__(self) -> None:
        self.encoders = (
            RGBEdgeSpectrumSceneEncoder(low_frequency_radius=1),
            RGBEdgeSpectrumSceneEncoder(low_frequency_radius=6),
        )

    @property
    def dimensions(self) -> int:
        return sum(encoder.dimensions for encoder in self.encoders)

    def encode(self, frame: np.ndarray) -> tuple[int, ...]:
        signature = tuple(
            value
            for encoder in self.encoders
            for value in encoder.encode(frame)
        )
        if len(signature) != self.dimensions:
            raise AssertionError("RGB dual-band spectrum descriptor width changed")
        return signature


class RGBEdgeDualBandSpectrumPanoramaEncoder(RGBPanoramaEncoder):
    """Yaw-invariant panorama over fixed radius-1 and radius-6 spectra."""

    VERSION = "rgb-edge-spectrum-r1-r6-panorama-v1"

    def __init__(
        self, scene_encoder: RGBEdgeDualBandSpectrumSceneEncoder | None = None
    ) -> None:
        super().__init__(scene_encoder or RGBEdgeDualBandSpectrumSceneEncoder())


def infer_visual_rotation_period(
    anchor: tuple[int, ...],
    scan: Iterable[Iterable[int]],
    *,
    minimum_steps: int = 4,
) -> tuple[int, tuple[float, ...]]:
    """Infer a full-turn period only from recurrence of public RGB signatures.

    ``scan`` contains the views after successive equal camera interventions.  The
    first eligible global distance minimum is selected deterministically.  An
    evaluator should still verify the resulting physical yaw when telemetry is
    available; this function intentionally has no access to it.
    """

    views = tuple(tuple(int(value) for value in item) for item in scan)
    if minimum_steps < 2:
        raise ValueError("minimum rotation period must be at least two steps")
    if len(views) < minimum_steps:
        raise ValueError("rotation scan is shorter than its minimum period")
    if not anchor or any(len(item) != len(anchor) for item in views):
        raise ValueError("rotation scan signatures must have equal nonzero width")
    distances = tuple(math.dist(anchor, item) for item in views)
    eligible = distances[minimum_steps - 1 :]
    period = minimum_steps + min(
        range(len(eligible)), key=lambda index: (eligible[index], index)
    )
    return period, distances


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
