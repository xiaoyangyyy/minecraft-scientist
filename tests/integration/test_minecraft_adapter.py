from __future__ import annotations

import json
from typing import Any

import pytest

from minecraft_scientist.minecraft import (
    MinecraftConnectionConfig,
    MinecraftWorldAdapter,
)
from minecraft_scientist.protocols import ActionKind, PrimitiveAction


class FakeBridge:
    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.held_item: dict[str, Any] | None = None
        self.last_action: dict[str, Any] | None = None
        self.connect_params: dict[str, Any] | None = None

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del timeout
        if method == "connect":
            assert params is not None and params["version"] == "1.21.11"
            self.connect_params = params
            self.connected = True
            return {"username": "ScientistBot", "version": "1.21.11"}
        if method == "observe":
            return self._observation()
        if method == "act":
            assert params is not None
            self.last_action = params["action"]
            if params["action"]["kind"] == "select_slot":
                self.held_item = {
                    "id": 101,
                    "name": "diamond_pickaxe",
                    "count": 1,
                    "slot": 36,
                }
            return {
                "action": params["action"]["kind"],
                "observation": self._observation(step=1),
            }
        raise AssertionError(method)

    def close(self) -> None:
        self.closed = True

    def _observation(self, *, step: int = 0) -> dict[str, Any]:
        return {
            "step": step,
            "position": {"x": 0.5, "y": 64.0, "z": -2.0},
            "health": 20,
            "food": 20,
            "time": 6000,
            "inventory": [
                {
                    "id": 101,
                    "name": "diamond_pickaxe",
                    "count": 1,
                    "slot": 36,
                }
            ],
            "held_item": self.held_item,
            "blocks": [
                {
                    "id": 202,
                    "name": "diamond_ore",
                    "position": {"x": 1, "y": 64, "z": -2},
                    "diggable": True,
                    "hardness": 3.0,
                }
            ],
            "entities": [
                {
                    "id": 303,
                    "name": "zombie",
                    "type": "mob",
                    "position": {"x": 4, "y": 64, "z": -2},
                }
            ],
        }


def test_adapter_hides_names_and_executes_opaque_action() -> None:
    bridge = FakeBridge()
    world = MinecraftWorldAdapter(bridge_factory=lambda: bridge)
    observation = world.reset(seed=17)
    serialized = observation.model_dump_json()

    assert "diamond" not in serialized
    assert "zombie" not in serialized
    assert all(item.object_id.startswith("obj_") for item in observation.objects)
    select = next(
        action
        for action in world.available_actions()
        if action.kind is ActionKind.SELECT
    )

    transition = world.step(select)

    assert transition.intervention_compliance is True
    assert transition.state_after.selected_id == select.target_id
    map_payload = json.dumps(
        [
            data["data"].model_dump(mode="json")
            for _, data in world.cognitive_map.graph.nodes(data=True)
        ]
    )
    assert "diamond" not in map_payload
    assert "zombie" not in map_payload
    world.close()
    assert bridge.closed is True


def test_adapter_ids_are_seeded_and_deterministic() -> None:
    first = MinecraftWorldAdapter(bridge_factory=FakeBridge).reset(seed=5)
    second = MinecraftWorldAdapter(bridge_factory=FakeBridge).reset(seed=5)
    third = MinecraftWorldAdapter(bridge_factory=FakeBridge).reset(seed=6)

    assert [item.object_id for item in first.objects] == [
        item.object_id for item in second.objects
    ]
    assert [item.object_id for item in first.objects] != [
        item.object_id for item in third.objects
    ]


def test_adapter_rejects_unknown_target_before_controller_call() -> None:
    world = MinecraftWorldAdapter(bridge_factory=FakeBridge)
    world.reset(seed=0)
    with pytest.raises(ValueError, match="absent"):
        world.step(PrimitiveAction(kind=ActionKind.INTERACT, target_id="obj_unknown"))


def test_adapter_exposes_live_curriculum_block_without_leaking_its_name() -> None:
    class InteractiveBridge(FakeBridge):
        def _observation(self, *, step: int = 0) -> dict[str, Any]:
            payload = super()._observation(step=step)
            payload["blocks"][0]["name"] = "lodestone"
            return payload

    bridge = InteractiveBridge()
    world = MinecraftWorldAdapter(bridge_factory=lambda: bridge)
    observation = world.reset(seed=23)
    target = next(
        item
        for item in observation.objects
        if set(item.affordances)
        == {ActionKind.PLACE_INPUT, ActionKind.ACTIVATE, ActionKind.INTERACT}
        and "distance" in item.observable_features
    )
    assert "lodestone" not in observation.model_dump_json()

    world.step(PrimitiveAction(kind=ActionKind.INTERACT, target_id=target.object_id))

    assert bridge.last_action is not None
    assert bridge.last_action["kind"] == "use_block"


def test_adapter_exposes_opaque_open_world_movement_frontiers() -> None:
    bridge = FakeBridge()
    world = MinecraftWorldAdapter(
        MinecraftConnectionConfig(exploration_stride=8),
        bridge_factory=lambda: bridge,
    )
    observation = world.reset(seed=31)
    assert bridge.connect_params is not None
    assert bridge.connect_params["pathfinder_timeout_ms"] == 15000
    movement_objects = [
        item for item in observation.objects if item.affordances == (ActionKind.MOVE,)
    ]

    assert len(movement_objects) == 4
    assert all(item.object_id.startswith("obj_") for item in movement_objects)
    assert all(
        item.observable_features["distance"] == 8.0
        and item.observable_features["previously_visited"] is False
        for item in movement_objects
    )
    serialized = observation.model_dump_json()
    assert all(label not in serialized for label in ("north", "south", "east", "west"))

    action = next(
        candidate
        for candidate in world.available_actions()
        if candidate.kind is ActionKind.MOVE
    )
    transition = world.step(action)

    assert transition.intervention_compliance is True
    assert transition.event.event_type == "minecraft_move"
    assert bridge.last_action is not None
    assert bridge.last_action["kind"] == "move_to"
    assert bridge.last_action["range"] == 1


def test_movement_frontier_ids_survive_small_revisit_position_drift() -> None:
    class OffsetBridge(FakeBridge):
        def __init__(self, x: float, z: float) -> None:
            super().__init__()
            self.x = x
            self.z = z

        def _observation(self, *, step: int = 0) -> dict[str, Any]:
            payload = super()._observation(step=step)
            payload["position"] = {"x": self.x, "y": 64.0, "z": self.z}
            return payload

    config = MinecraftConnectionConfig(exploration_stride=8)
    first = MinecraftWorldAdapter(
        config,
        bridge_factory=lambda: OffsetBridge(0.5, -2.0),
    ).reset(seed=31)
    revisited = MinecraftWorldAdapter(
        config,
        bridge_factory=lambda: OffsetBridge(1.5, -1.0),
    ).reset(seed=31)

    first_ids = sorted(
        item.object_id
        for item in first.objects
        if item.affordances == (ActionKind.MOVE,)
    )
    revisited_ids = sorted(
        item.object_id
        for item in revisited.objects
        if item.affordances == (ActionKind.MOVE,)
    )
    assert revisited_ids == first_ids
