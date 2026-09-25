"""Agent-safe StructuredWorld adapter over the Mineflayer process bridge."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field

from minecraft_scientist.cognitive_map import CognitiveMap
from minecraft_scientist.protocols import (
    ActionKind,
    ObjectObservation,
    Observation,
    PrimitiveAction,
    TransitionRecord,
    WorldEvent,
)

from .bridge import MinecraftBridgeClient, MinecraftBridgeError
from .server import MINECRAFT_VERSION

INTERACTIVE_BLOCK_NAMES = frozenset(
    {
        "lodestone",
        "copper_block",
        "orange_glazed_terracotta",
        "emerald_block",
        "chiseled_copper",
        "chiseled_tuff",
        "polished_blackstone",
    }
)


class Bridge(Protocol):
    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class MinecraftConnectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = "127.0.0.1"
    port: int = Field(default=25565, ge=1, le=65535)
    username: str = "ScientistBot"
    auth: str = "offline"
    version: str = MINECRAFT_VERSION
    radius: int = Field(default=8, ge=1, le=32)
    max_blocks: int = Field(default=128, ge=1, le=512)
    timeout: float = Field(default=30.0, gt=0)
    pathfinder_timeout: float = Field(default=15.0, gt=0, le=120)
    allowed_block_positions: tuple[tuple[int, int, int], ...] | None = None
    exploration_stride: int | None = Field(default=None, ge=1, le=32)


class MinecraftWorldAdapter:
    """Translate live semantic data into opaque, capability-limited observations."""

    def __init__(
        self,
        config: MinecraftConnectionConfig | None = None,
        *,
        bridge_factory: Callable[[], Bridge] = MinecraftBridgeClient,
        cognitive_map: CognitiveMap | None = None,
    ) -> None:
        self.config = config or MinecraftConnectionConfig()
        self._bridge_factory = bridge_factory
        self._bridge: Bridge | None = None
        self._map = cognitive_map or CognitiveMap()
        self._seed = 0
        self._episode = 0
        self._transition = 0
        self._bindings: dict[str, dict[str, Any]] = {}
        self._controller_ids: dict[tuple[str, str], set[str]] = {}
        self._observation: Observation | None = None

    @property
    def cognitive_map(self) -> CognitiveMap:
        return self._map

    def reset(self, seed: int) -> Observation:
        self.close()
        self._seed = seed
        self._episode += 1
        self._transition = 0
        self._controller_ids.clear()
        self._bridge = self._bridge_factory()
        self._bridge.request(
            "connect",
            {
                "host": self.config.host,
                "port": self.config.port,
                "username": self.config.username,
                "auth": self.config.auth,
                "version": self.config.version,
                "timeout_ms": int(self.config.timeout * 1000),
                "pathfinder_timeout_ms": int(self.config.pathfinder_timeout * 1000),
            },
            timeout=self.config.timeout + 5,
        )
        raw = self._bridge.request("observe", self._observe_params())
        self._observation = self._encode(raw)
        return self._observation

    def observe(self) -> Observation:
        if self._bridge is None:
            raise RuntimeError("world must be reset before observation")
        raw = self._bridge.request("observe", self._observe_params())
        self._observation = self._encode(raw)
        return self._observation

    def available_actions(self) -> tuple[PrimitiveAction, ...]:
        if self._observation is None:
            raise RuntimeError("world must be reset before requesting actions")
        actions = [PrimitiveAction(kind=ActionKind.WAIT, ticks=1)]
        for object_id, binding in sorted(self._bindings.items()):
            if binding["operation"] == "select_slot":
                actions.append(
                    PrimitiveAction(kind=ActionKind.SELECT, target_id=object_id)
                )
            elif binding["operation"] in {"dig", "attack"}:
                actions.append(
                    PrimitiveAction(kind=ActionKind.INTERACT, target_id=object_id)
                )
            elif binding["operation"] == "use_block":
                actions.extend(
                    (
                        PrimitiveAction(
                            kind=ActionKind.PLACE_INPUT, target_id=object_id
                        ),
                        PrimitiveAction(kind=ActionKind.ACTIVATE, target_id=object_id),
                        PrimitiveAction(kind=ActionKind.INTERACT, target_id=object_id),
                    )
                )
            elif binding["operation"] == "move_to":
                actions.append(
                    PrimitiveAction(kind=ActionKind.MOVE, target_id=object_id)
                )
        actions.append(PrimitiveAction(kind=ActionKind.ACTIVATE))
        return tuple(actions)

    def step(self, action: PrimitiveAction) -> TransitionRecord:
        if self._bridge is None or self._observation is None:
            raise RuntimeError("world must be reset before stepping")
        before = self._observation
        self._transition += 1
        bridge_action = self._translate(action)
        failure: str | None = None
        compliant = True
        try:
            if action.item_id is not None:
                item_binding = self._bindings.get(action.item_id)
                if item_binding is None or item_binding["operation"] != "select_slot":
                    raise ValueError(
                        "action item is absent from the current inventory observation"
                    )
                self._bridge.request(
                    "act",
                    {
                        "action": {
                            "kind": "select_slot",
                            "slot": item_binding["slot"],
                        },
                        "observe": self._observe_params(),
                    },
                    timeout=self.config.timeout,
                )
            result = self._bridge.request(
                "act",
                {"action": bridge_action, "observe": self._observe_params()},
                timeout=max(
                    self.config.timeout,
                    self.config.pathfinder_timeout + 5,
                    action.ticks * 0.05 + 5,
                ),
            )
            raw = result["observation"]
            outcome_success = bool(result.get("success", True))
        except (MinecraftBridgeError, KeyError) as exc:
            failure = str(exc)
            raw = self._bridge.request("observe", self._observe_params())
            compliant = False
            outcome_success = False
        after = self._encode(raw)
        self._observation = after
        delta = self._delta(before, after)
        positive_inventory = [
            item_id for item_id, count in delta["inventory"].items() if count > 0
        ]
        event = WorldEvent(
            event_type=f"minecraft_{action.kind.value}",
            target_id=action.target_id,
            success=outcome_success,
            duration=action.ticks,
            output_id=(positive_inventory[0] if len(positive_inventory) == 1 else None),
            details={"controller": "mineflayer"},
        )
        return TransitionRecord(
            transition_id=f"minecraft-{self._episode}-{self._transition}",
            context={"backend": "minecraft", "version": self.config.version},
            state_before=before,
            executed_action=action,
            intervention_compliance=compliant,
            state_after=after,
            observed_delta=delta,
            event=event,
            controller_failure=failure,
            provenance={
                "backend": "mineflayer",
                "episode": self._episode,
                "seed": self._seed,
            },
        )

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        self._observation = None
        self._bindings.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _encode(self, raw: dict[str, Any]) -> Observation:
        self._bindings = {}
        position = raw["position"]
        origin = (float(position["x"]), float(position["y"]), float(position["z"]))
        inventory: dict[str, int] = {}
        objects: list[ObjectObservation] = []
        public_map_observation = {
            "step": int(raw.get("step", 0)),
            "position": position,
            "health": float(raw.get("health", 20.0)),
            "food": float(raw.get("food", 20.0)),
            "blocks": [],
            "entities": [],
        }
        for item in raw.get("inventory", ()):
            object_id = self._opaque("inventory", str(item["id"]))
            self._remember_controller_id("inventory", str(item.get("name")), object_id)
            inventory[object_id] = inventory.get(object_id, 0) + int(item["count"])
            self._bindings.setdefault(
                object_id,
                {"operation": "select_slot", "slot": int(item["slot"])},
            )
            objects.append(
                ObjectObservation(
                    object_id=object_id,
                    affordances=(ActionKind.SELECT,),
                    observable_features={
                        "count": inventory[object_id],
                        "durable": int(item.get("max_durability", 0)) > 0,
                    },
                )
            )
        for block in raw.get("blocks", ()):
            block_position = block["position"]
            integer_position = tuple(
                int(block_position[axis]) for axis in ("x", "y", "z")
            )
            if (
                self.config.allowed_block_positions is not None
                and integer_position not in self.config.allowed_block_positions
            ):
                continue
            key = ":".join(
                str(value)
                for value in (
                    block_position["x"],
                    block_position["y"],
                    block_position["z"],
                )
            )
            object_id = self._opaque("block", key)
            self._remember_controller_id("block", str(block.get("name")), object_id)
            coordinates = {
                axis: float(block_position[axis]) for axis in ("x", "y", "z")
            }
            operation = (
                "use_block"
                if str(block.get("name")) in INTERACTIVE_BLOCK_NAMES
                else "dig"
            )
            self._bindings[object_id] = {
                "operation": operation,
                **coordinates,
            }
            features: dict[str, str | int | float | bool] = {
                **coordinates,
                "distance": math.dist(origin, tuple(coordinates.values())),
                "diggable": bool(block.get("diggable", False)),
            }
            if block.get("hardness") is not None:
                features["hardness"] = float(block["hardness"])
            objects.append(
                ObjectObservation(
                    object_id=object_id,
                    affordances=(
                        (
                            ActionKind.PLACE_INPUT,
                            ActionKind.ACTIVATE,
                            ActionKind.INTERACT,
                        )
                        if operation == "use_block"
                        else (ActionKind.INTERACT,)
                    ),
                    observable_features=features,
                )
            )
            public_map_observation["blocks"].append({"name": object_id})
        for entity in raw.get("entities", ()):
            entity_position = entity["position"]
            object_id = self._opaque("entity", str(entity["id"]))
            self._remember_controller_id("entity", str(entity.get("name")), object_id)
            coordinates = {
                axis: float(entity_position[axis]) for axis in ("x", "y", "z")
            }
            self._bindings[object_id] = {
                "operation": "attack",
                "entity_id": int(entity["id"]),
            }
            objects.append(
                ObjectObservation(
                    object_id=object_id,
                    affordances=(ActionKind.INTERACT,),
                    observable_features={
                        **coordinates,
                        "distance": math.dist(origin, tuple(coordinates.values())),
                    },
                )
            )
            public_map_observation["entities"].append({"name": object_id})
        if self.config.exploration_stride is not None:
            stride = self.config.exploration_stride
            source_node = self._map.node_id_for(origin)
            frontier_offsets = (
                (-stride, 0),
                (0, -stride),
                (0, stride),
                (stride, 0),
            )
            for x_offset, z_offset in frontier_offsets:
                coordinates = {
                    "x": origin[0] + x_offset,
                    "y": origin[1],
                    "z": origin[2] + z_offset,
                }
                stable_key = f"{source_node}:{x_offset}:{z_offset}"
                object_id = self._opaque("frontier", stable_key)
                destination_node = self._map.node_id_for(tuple(coordinates.values()))
                self._bindings[object_id] = {
                    "operation": "move_to",
                    **coordinates,
                }
                objects.append(
                    ObjectObservation(
                        object_id=object_id,
                        affordances=(ActionKind.MOVE,),
                        observable_features={
                            **coordinates,
                            "distance": float(stride),
                            "previously_visited": destination_node in self._map.graph,
                        },
                    )
                )
        self._map.ingest(public_map_observation)
        return Observation(
            episode_id=self._episode,
            step=int(raw.get("step", 0)),
            inventory=inventory,
            selected_id=self._selected_id(raw),
            agent_state={
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position["z"]),
                "health": float(raw.get("health", 20.0)),
                "food": float(raw.get("food", 20.0)),
                "time": int(raw.get("time", 0)),
            },
            objects=tuple(sorted(objects, key=lambda item: item.object_id)),
        )

    def _selected_id(self, raw: dict[str, Any]) -> str | None:
        held = raw.get("held_item")
        if not held:
            return None
        return self._opaque("inventory", str(held["id"]))

    def _translate(self, action: PrimitiveAction) -> dict[str, Any]:
        if action.kind is ActionKind.WAIT:
            return {"kind": "wait", "ticks": action.ticks}
        if action.kind is ActionKind.ACTIVATE and action.target_id is None:
            return {"kind": "use_item", "duration_ms": action.ticks * 50}
        if action.target_id is None or action.target_id not in self._bindings:
            raise ValueError("action target is absent from the current observation")
        binding = self._bindings[action.target_id]
        if action.kind is ActionKind.SELECT and binding["operation"] == "select_slot":
            return {"kind": "select_slot", "slot": binding["slot"]}
        if action.kind is ActionKind.INTERACT and binding["operation"] == "dig":
            return {"kind": "dig", **self._coordinates(binding)}
        if action.kind is ActionKind.INTERACT and binding["operation"] == "use_block":
            return {
                "kind": "use_block",
                "intent": "interact",
                **self._coordinates(binding),
            }
        if (
            action.kind in {ActionKind.PLACE_INPUT, ActionKind.ACTIVATE}
            and binding["operation"] == "use_block"
        ):
            return {
                "kind": "use_block",
                "intent": action.kind.value,
                **self._coordinates(binding),
            }
        if action.kind is ActionKind.INTERACT and binding["operation"] == "attack":
            return {"kind": "attack", "entity_id": binding["entity_id"]}
        if action.kind is ActionKind.MOVE and binding["operation"] == "move_to":
            return {
                "kind": "move_to",
                "range": 1,
                **self._coordinates(binding),
            }
        raise ValueError("action kind is incompatible with its target")

    def _opaque(self, category: str, stable_key: str) -> str:
        digest = hashlib.blake2s(
            f"{self._seed}:{category}:{stable_key}".encode(), digest_size=8
        ).hexdigest()
        return f"obj_{digest}"

    def _controller_ids_for(self, category: str, semantic_name: str) -> tuple[str, ...]:
        """Privileged controller lookup; never expose this adapter to the agent."""

        return tuple(sorted(self._controller_ids.get((category, semantic_name), ())))

    def _remember_controller_id(
        self, category: str, semantic_name: str, object_id: str
    ) -> None:
        self._controller_ids.setdefault((category, semantic_name), set()).add(object_id)

    def _observe_params(self) -> dict[str, int]:
        return {
            "radius": self.config.radius,
            "max_blocks": self.config.max_blocks,
        }

    @staticmethod
    def _coordinates(binding: dict[str, Any]) -> dict[str, float]:
        return {axis: float(binding[axis]) for axis in ("x", "y", "z")}

    @staticmethod
    def _delta(before: Observation, after: Observation) -> dict[str, Any]:
        inventory_ids = set(before.inventory) | set(after.inventory)
        return {
            "inventory": {
                item_id: after.inventory.get(item_id, 0)
                - before.inventory.get(item_id, 0)
                for item_id in sorted(inventory_ids)
                if after.inventory.get(item_id, 0) != before.inventory.get(item_id, 0)
            },
            "selected_changed": before.selected_id != after.selected_id,
            "visible_object_delta": len(after.objects) - len(before.objects),
            "agent_state": {
                key: float(after.agent_state.get(key, 0))
                - float(before.agent_state.get(key, 0))
                for key in sorted(set(before.agent_state) | set(after.agent_state))
                if isinstance(before.agent_state.get(key, 0), (int, float))
                and isinstance(after.agent_state.get(key, 0), (int, float))
                and float(after.agent_state.get(key, 0))
                != float(before.agent_state.get(key, 0))
            },
        }
