"""Label-free fixed-width scene descriptors for agent-visible observations."""

from __future__ import annotations

import math

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
