from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Self

from minecraft_scientist.exploration import ExplorationMemory
from minecraft_scientist.minecraft import open_world as open_world_module
from minecraft_scientist.protocols import (
    ActionKind,
    ObjectObservation,
    Observation,
    PrimitiveAction,
    TransitionRecord,
    WorldEvent,
)


class _FakeServer:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        pass


class _FakeWorld:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.observation: Observation | None = None
        self.bindings: dict[str, tuple[float, float, float]] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def reset(self, seed: int) -> Observation:
        self.state["seed"] = seed
        self.observation = self._observe()
        return self.observation

    def available_actions(self) -> tuple[PrimitiveAction, ...]:
        assert self.observation is not None
        return tuple(
            PrimitiveAction(kind=ActionKind.MOVE, target_id=target_id)
            for target_id in sorted(self.bindings)
        )

    def step(self, action: PrimitiveAction) -> TransitionRecord:
        assert self.observation is not None and action.target_id is not None
        before = self.observation
        self.state["calls"] += 1
        failure = self.state["calls"] <= self.state["failures"]
        if not failure:
            self.state["position"] = list(self.bindings[action.target_id])
        self.state["step"] += 1
        after = self._observe()
        self.observation = after
        return TransitionRecord(
            transition_id=f"fake-{self.state['calls']}",
            context={"backend": "fake"},
            state_before=before,
            executed_action=action,
            intervention_compliance=not failure,
            state_after=after,
            observed_delta={},
            event=WorldEvent(
                event_type="minecraft_move",
                target_id=action.target_id,
                success=not failure,
            ),
            controller_failure="synthetic obstruction" if failure else None,
            provenance={"backend": "fake"},
        )

    def _observe(self) -> Observation:
        x, y, z = self.state["position"]
        stride = 4
        source = (math.floor(x / 8), math.floor(y / 8), math.floor(z / 8))
        self.bindings = {}
        objects: list[ObjectObservation] = []
        for dx, dz in ((-stride, 0), (0, -stride), (0, stride), (stride, 0)):
            target_id = f"obj_{source[0]}_{source[1]}_{source[2]}_{dx}_{dz}"
            target = (x + dx, y, z + dz)
            self.bindings[target_id] = target
            objects.append(
                ObjectObservation(
                    object_id=target_id,
                    affordances=(ActionKind.MOVE,),
                    observable_features={
                        "x": target[0],
                        "y": target[1],
                        "z": target[2],
                        "distance": 4.0,
                        "previously_visited": False,
                    },
                )
            )
        place_cell = (math.floor(x / stride), math.floor(z / stride))
        block_count = (place_cell[0] + 10) * 31 + place_cell[1] + 10
        for index in range(block_count):
            objects.append(
                ObjectObservation(
                    object_id=f"block_{place_cell[0]}_{place_cell[1]}_{index}",
                    affordances=(ActionKind.INTERACT,),
                    observable_features={
                        "x": x + (index % 7) - 3,
                        "y": y - 1,
                        "z": z + ((index // 7) % 7) - 3,
                        "distance": 1.0,
                        "diggable": True,
                        "hardness": float(index % 6),
                    },
                )
            )
        return Observation(
            episode_id=1,
            step=self.state["step"],
            inventory={},
            agent_state={"x": x, "y": y, "z": z, "health": 20, "food": 20},
            objects=tuple(sorted(objects, key=lambda item: item.object_id)),
        )


def test_real_explorer_reopens_memory_and_recognizes_restart_places(
    tmp_path: Path, monkeypatch
) -> None:
    state: dict[str, Any] = {
        "position": [0.5, 64.0, 0.5],
        "step": 0,
        "calls": 0,
        "failures": 1,
    }
    monkeypatch.setattr(open_world_module, "MinecraftServerProcess", _FakeServer)
    monkeypatch.setattr(
        open_world_module,
        "MinecraftWorldAdapter",
        lambda _config: _FakeWorld(state),
    )

    report = open_world_module.run_open_world_exploration(
        server_directory=tmp_path / "server",
        output=tmp_path / "run",
        seed=41,
        move_budget=6,
        restart_interval=2,
        minimum_successful_moves=5,
        minimum_unique_places=6,
    )

    assert report["successful_moves"] == 5
    assert report["controller_failures"] == 1
    assert report["sessions"] == 3
    assert report["restart_trials"] == 2
    assert report["restart_recognitions"] == 2
    assert report["perceptual_restart_recognitions"] == 2
    assert report["perceptual_restart_false_matches"] == 0
    assert report["memory"]["place_count"] == 6
    assert report["memory"]["perceptual_place_count"] == 6
    assert report["memory"]["transition_count"] == 5
    assert report["memory"]["integrity_valid"] is True
    assert report["gate"]["passed"] is True

    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["session"] for item in events] == [0, 0, 1, 1, 2, 2]
    assert sum(not item["movement_valid"] for item in events) == 1
    with ExplorationMemory.open(tmp_path / "run" / "exploration.sqlite3") as memory:
        assert memory.verify_integrity()
