"""Optional MineStudio visual backend with an opaque perception boundary.

MineStudio is imported lazily because its simulator stack is Linux/JDK8-oriented
and intentionally does not belong to the default deterministic test environment.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minecraft_scientist.protocols import (
    ActionKind,
    ObjectObservation,
    Observation,
    PrimitiveAction,
    TransitionRecord,
    WorldEvent,
)


class MineStudioSimulator(Protocol):
    def reset(self) -> tuple[np.ndarray, dict[str, Any]]: ...

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]: ...

    def noop_action(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class VisualScene:
    """Private codec result; controller_state never enters public observations."""

    objects: tuple[ObjectObservation, ...] = ()
    inventory: dict[str, int] | None = None
    agent_state: dict[str, str | int | float | bool] | None = None
    selected_id: str | None = None
    controller_state: Any = None


class PerceptionController(Protocol):
    """Pluggable vision/navigation policy spanning pixels and low-level actions."""

    def perceive(
        self, frame: np.ndarray, info: dict[str, Any], *, seed: int
    ) -> VisualScene: ...

    def encode_action(self, action: PrimitiveAction, scene: VisualScene) -> Any: ...


class MineStudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_type: str = "env"
    obs_size: tuple[int, int] = (224, 224)
    render_size: tuple[int, int] = (640, 360)
    seed: int = 0
    inventory: dict[str, int] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GridVisualController:
    """Label-free RGB baseline using stable screen cells as intervention targets."""

    rows: int = 3
    columns: int = 3
    hotbar_slots: int = 9

    def __post_init__(self) -> None:
        if self.rows < 1 or self.columns < 1 or not 1 <= self.hotbar_slots <= 9:
            raise ValueError("grid dimensions must be positive and hotbar_slots in 1..9")

    def perceive(
        self, frame: np.ndarray, info: dict[str, Any], *, seed: int
    ) -> VisualScene:
        del info
        array = np.asarray(frame, dtype=np.float32)
        if array.ndim == 2:
            array = array[..., None]
        height, width, _ = array.shape
        objects: list[ObjectObservation] = []
        bindings: dict[str, dict[str, Any]] = {}
        inventory: dict[str, int] = {}
        for slot in range(1, self.hotbar_slots + 1):
            object_id = self._opaque(seed, "hotbar", slot)
            inventory[object_id] = 1
            bindings[object_id] = {"kind": "hotbar", "slot": slot}
            objects.append(
                ObjectObservation(
                    object_id=object_id,
                    affordances=(ActionKind.SELECT,),
                    observable_features={"slot": slot},
                )
            )
        for row in range(self.rows):
            top = row * height // self.rows
            bottom = (row + 1) * height // self.rows
            for column in range(self.columns):
                left = column * width // self.columns
                right = (column + 1) * width // self.columns
                patch = array[top:bottom, left:right]
                object_id = self._opaque(seed, "cell", row, column)
                pitch = ((row + 0.5) / self.rows - 0.5) * 60.0
                yaw = ((column + 0.5) / self.columns - 0.5) * 90.0
                bindings[object_id] = {
                    "kind": "cell",
                    "camera": (pitch, yaw),
                }
                objects.append(
                    ObjectObservation(
                        object_id=object_id,
                        affordances=(
                            ActionKind.INTERACT,
                            ActionKind.PLACE_INPUT,
                            ActionKind.ACTIVATE,
                        ),
                        observable_features={
                            "screen_row": row,
                            "screen_column": column,
                            "mean_intensity": float(patch.mean()) if patch.size else 0.0,
                            "contrast": float(patch.std()) if patch.size else 0.0,
                        },
                    )
                )
        return VisualScene(
            objects=tuple(objects),
            inventory=inventory,
            agent_state={
                "frame_mean": float(array.mean()),
                "frame_contrast": float(array.std()),
            },
            controller_state=bindings,
        )

    def encode_action(self, action: PrimitiveAction, scene: VisualScene) -> Any:
        payload = self._noop()
        bindings = scene.controller_state
        if not isinstance(bindings, dict):
            raise ValueError("visual scene is missing controller bindings")
        if action.kind is ActionKind.WAIT:
            return payload
        if action.target_id is None or action.target_id not in bindings:
            raise ValueError("visual action target is absent from the current frame")
        binding = bindings[action.target_id]
        if action.kind is ActionKind.SELECT and binding["kind"] == "hotbar":
            payload[f"hotbar.{int(binding['slot'])}"] = np.array(1)
            return payload
        if binding["kind"] != "cell":
            raise ValueError("action kind is incompatible with visual target")
        payload["camera"] = np.asarray(binding["camera"], dtype=np.float32)
        if action.kind is ActionKind.INTERACT:
            payload["attack"] = np.array(1)
        elif action.kind is ActionKind.PLACE_INPUT:
            payload["use"] = np.array(1)
        elif action.kind is ActionKind.ACTIVATE:
            payload["sneak"] = np.array(1)
            payload["use"] = np.array(1)
        else:
            raise ValueError("action kind is incompatible with visual target")
        return payload

    @staticmethod
    def _opaque(seed: int, *parts: object) -> str:
        raw = ":".join((str(seed), *(str(part) for part in parts)))
        return "obj_" + hashlib.blake2s(raw.encode(), digest_size=8).hexdigest()

    @staticmethod
    def _noop() -> dict[str, np.ndarray]:
        keys = (
            "attack",
            "back",
            "drop",
            "forward",
            "jump",
            "left",
            "right",
            "sneak",
            "sprint",
            "use",
            "inventory",
            *(f"hotbar.{slot}" for slot in range(1, 10)),
        )
        payload = {key: np.array(0) for key in keys}
        payload["camera"] = np.asarray([0.0, 0.0], dtype=np.float32)
        return payload


def create_minecraft_sim(config: MineStudioConfig) -> MineStudioSimulator:
    """Create the real simulator only in an isolated environment with MineStudio."""

    try:
        from minestudio.simulator import MinecraftSim
    except ImportError as exc:
        raise RuntimeError(
            "MineStudio is not installed; use the isolated Linux visual environment"
        ) from exc
    return MinecraftSim(
        action_type=config.action_type,
        obs_size=config.obs_size,
        render_size=config.render_size,
        seed=config.seed,
        inventory=config.inventory,
    )


def check_minestudio_backend() -> dict[str, Any]:
    """Report prerequisites without importing or launching the heavy simulator."""

    java = shutil.which("java")
    java_version = None
    if java is not None:
        result = subprocess.run(
            [java, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first = (result.stderr or result.stdout).splitlines()
        java_version = first[0] if first else "unknown"
    linux = platform.system() == "Linux"
    installed = importlib.util.find_spec("minestudio") is not None
    java8 = java_version is not None and ('"1.8.' in java_version or '"8' in java_version)
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "linux": linux,
        "minestudio_installed": installed,
        "java": java,
        "java_version": java_version,
        "java8_detected": java8,
        "ready_for_visual_simulation": linux and installed and java8,
        "launch_attempted": False,
    }


class MineStudioVisualAdapter:
    """StructuredWorld-compatible adapter over frames and a perception controller."""

    def __init__(
        self,
        controller: PerceptionController,
        config: MineStudioConfig | None = None,
        *,
        simulator_factory: Callable[[MineStudioConfig], MineStudioSimulator] = (
            create_minecraft_sim
        ),
    ) -> None:
        self.config = config or MineStudioConfig()
        self._controller = controller
        self._simulator_factory = simulator_factory
        self._simulator: MineStudioSimulator | None = None
        self._scene: VisualScene | None = None
        self._observation: Observation | None = None
        self._seed = self.config.seed
        self._episode = 0
        self._step = 0

    def reset(self, seed: int) -> Observation:
        self.close()
        self._seed = seed
        self._episode += 1
        self._step = 0
        self._simulator = self._simulator_factory(
            self.config.model_copy(update={"seed": seed})
        )
        frame, info = self._simulator.reset()
        return self._encode(frame, info)

    def observe(self) -> Observation:
        if self._observation is None:
            raise RuntimeError("world must be reset before observation")
        return self._observation

    def available_actions(self) -> tuple[PrimitiveAction, ...]:
        if self._scene is None:
            raise RuntimeError("world must be reset before requesting actions")
        actions: list[PrimitiveAction] = [
            PrimitiveAction(kind=ActionKind.WAIT, ticks=1)
        ]
        for item in self._scene.objects:
            for affordance in item.affordances:
                actions.append(PrimitiveAction(kind=affordance, target_id=item.object_id))
        return tuple(dict.fromkeys(actions))

    def step(self, action: PrimitiveAction) -> TransitionRecord:
        if self._simulator is None or self._scene is None or self._observation is None:
            raise RuntimeError("world must be reset before stepping")
        before = self._observation
        payload = self._controller.encode_action(action, self._scene)
        noop = getattr(self._simulator, "noop_action", None)
        if callable(noop) and isinstance(payload, dict):
            complete_payload = noop()
            complete_payload.update(payload)
            payload = complete_payload
        frame, reward, terminated, truncated, info = self._simulator.step(payload)
        self._step += 1
        after = self._encode(frame, info)
        success = not bool(info.get("controller_failure", False))
        event = WorldEvent(
            event_type=f"minestudio_{action.kind.value}",
            target_id=action.target_id,
            success=success,
            duration=action.ticks,
            details={
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
        )
        return TransitionRecord(
            transition_id=f"minestudio-{self._episode}-{self._step}",
            context={"backend": "minestudio", "visual": True},
            state_before=before,
            executed_action=action,
            intervention_compliance=success,
            state_after=after,
            observed_delta={
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "frame_changed": (
                    before.agent_state.get("frame_sha256")
                    != after.agent_state.get("frame_sha256")
                ),
            },
            event=event,
            controller_failure=(
                str(info["controller_failure"])
                if info.get("controller_failure")
                else None
            ),
            provenance={
                "backend": "minestudio",
                "episode": self._episode,
                "seed": self._seed,
            },
        )

    def close(self) -> None:
        if self._simulator is not None:
            self._simulator.close()
        self._simulator = None
        self._scene = None
        self._observation = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _encode(self, frame: np.ndarray, info: dict[str, Any]) -> Observation:
        array = np.asarray(frame)
        if array.ndim not in {2, 3}:
            raise ValueError("MineStudio observation frame must be a 2D or 3D array")
        scene = self._controller.perceive(array, info, seed=self._seed)
        if any(not item.object_id.startswith("obj_") for item in scene.objects):
            raise ValueError("visual object identifiers must use the opaque obj_ prefix")
        shape = array.shape
        state: dict[str, str | int | float | bool] = {
            **(scene.agent_state or {}),
            "frame_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "frame_height": int(shape[0]),
            "frame_width": int(shape[1]),
            "frame_channels": int(shape[2]) if array.ndim == 3 else 1,
        }
        observation = Observation(
            episode_id=self._episode,
            step=self._step,
            inventory=scene.inventory or {},
            selected_id=scene.selected_id,
            agent_state=state,
            objects=tuple(sorted(scene.objects, key=lambda item: item.object_id)),
        )
        self._reject_semantic_leaks(info, observation)
        self._scene = scene
        self._observation = observation
        return observation

    @staticmethod
    def _reject_semantic_leaks(info: dict[str, Any], observation: Observation) -> None:
        """Catch common accidental name/type leakage from privileged simulator info."""

        sensitive_keys = {"name", "type", "label", "item_name", "block_name"}
        sensitive: set[str] = set()

        def visit(value: Any, key: str | None = None) -> None:
            if key in sensitive_keys and isinstance(value, str) and len(value) >= 3:
                sensitive.add(value)
            if isinstance(value, dict):
                for child_key, child in value.items():
                    visit(child, str(child_key))
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child, key)

        visit(info)
        serialized = json.dumps(observation.model_dump(mode="json"), sort_keys=True)
        leaked = sorted(token for token in sensitive if token in serialized)
        if leaked:
            raise ValueError(f"perception controller leaked semantic token: {leaked[0]}")
