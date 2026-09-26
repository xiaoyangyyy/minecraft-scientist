from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from minecraft_scientist.exploration import (
    RGBSceneEncoder,
    rgb_signature_from_observation,
)
from minecraft_scientist.minecraft import (
    GridVisualController,
    MineStudioVisualAdapter,
    VisualScene,
)
from minecraft_scientist.protocols import ActionKind, ObjectObservation, PrimitiveAction


class FakeSimulator:
    def __init__(self) -> None:
        self.closed = False
        self.actions: list[Any] = []
        self.counter = 0

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        return np.zeros((4, 5, 3), dtype=np.uint8), {
            "objects": [{"name": "diamond_ore", "type": "block"}]
        }

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        self.counter += 1
        frame = np.full((4, 5, 3), self.counter, dtype=np.uint8)
        return (
            frame,
            0.25,
            False,
            False,
            {"objects": [{"name": "diamond_ore", "type": "block"}]},
        )

    def close(self) -> None:
        self.closed = True


class FakeVisualController:
    def perceive(
        self, frame: np.ndarray, info: dict[str, Any], *, seed: int
    ) -> VisualScene:
        del frame, info
        return VisualScene(
            objects=(
                ObjectObservation(
                    object_id=f"obj_{seed:04d}",
                    affordances=(ActionKind.INTERACT,),
                    observable_features={"bbox_x": 0.25, "bbox_y": 0.5},
                ),
            ),
            agent_state={"visual_confidence": 0.9},
            controller_state={"low_level_target": 7},
        )

    def encode_action(self, action: PrimitiveAction, scene: VisualScene) -> Any:
        assert scene.controller_state == {"low_level_target": 7}
        return {"kind": action.kind.value, **scene.controller_state}


@pytest.mark.parametrize("terminal_at,expected", [(None, 5), (2, 2)])
def test_wait_runs_requested_frames_and_stops_at_terminal(terminal_at, expected):
    class TerminalSimulator(FakeSimulator):
        def step(self, action):
            frame, reward, _, truncated, info = super().step(action)
            return frame, reward, self.counter == terminal_at, truncated, info

    simulator = TerminalSimulator()
    with MineStudioVisualAdapter(
        FakeVisualController(), simulator_factory=lambda config: simulator
    ) as world:
        world.reset(0)
        transition = world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=5))
        assert len(simulator.actions) == expected
        assert transition.event.duration == expected
        assert transition.observed_delta["reward"] == expected * 0.25
        assert transition.intervention_compliance is (terminal_at is None)
        if terminal_at is not None:
            assert world.available_actions() == ()
            with pytest.raises(RuntimeError, match="episode has ended"):
                world.step(PrimitiveAction(kind=ActionKind.WAIT))


def test_real_observation_mapping_is_unwrapped_on_reset_and_step() -> None:
    class MappingSimulator(FakeSimulator):
        def reset(self):
            frame, info = super().reset()
            return {"image": frame, "private_label": "diamond_ore"}, info

        def step(self, action):
            frame, reward, terminated, truncated, info = super().step(action)
            return {"image": frame}, reward, terminated, truncated, info

    with MineStudioVisualAdapter(
        FakeVisualController(), simulator_factory=lambda config: MappingSimulator()
    ) as world:
        initial = world.reset(7)
        assert initial.agent_state["frame_width"] == 5
        assert "diamond_ore" not in initial.model_dump_json()
        result = world.step(PrimitiveAction(kind=ActionKind.WAIT))
        assert result.observed_delta["frame_changed"]


def test_wait_preserves_control_failure_and_stops_immediately() -> None:
    class FailingSimulator(FakeSimulator):
        def step(self, action):
            frame, reward, terminated, truncated, info = super().step(action)
            info["controller_failure"] = "input_not_applied"
            return frame, reward, terminated, truncated, info

    simulator = FailingSimulator()
    with MineStudioVisualAdapter(
        FakeVisualController(), simulator_factory=lambda config: simulator
    ) as world:
        world.reset(0)
        result = world.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=5))
        assert simulator.counter == 1
        assert result.controller_failure == "input_not_applied"
        assert not result.intervention_compliance
        assert not result.event.success


def test_reset_reopens_actions_after_terminal_episode() -> None:
    class EndingSimulator(FakeSimulator):
        def step(self, action):
            frame, reward, _, truncated, info = super().step(action)
            return frame, reward, True, truncated, info

    with MineStudioVisualAdapter(
        FakeVisualController(), simulator_factory=lambda config: EndingSimulator()
    ) as world:
        world.reset(0)
        result = world.step(PrimitiveAction(kind=ActionKind.WAIT))
        assert result.intervention_compliance  # Full requested duration completed.
        assert world.available_actions() == ()
        world.reset(1)
        assert world.available_actions()


def test_minestudio_adapter_exposes_pixels_only_through_opaque_perception() -> None:
    simulator = FakeSimulator()
    world = MineStudioVisualAdapter(
        FakeVisualController(), simulator_factory=lambda config: simulator
    )
    initial = world.reset(17)

    assert "diamond_ore" not in initial.model_dump_json()
    assert initial.objects[0].object_id == "obj_0017"
    assert initial.agent_state["frame_height"] == 4
    action = next(
        item for item in world.available_actions() if item.kind is ActionKind.INTERACT
    )
    transition = world.step(action)

    assert transition.intervention_compliance is True
    assert transition.observed_delta["frame_changed"] is True
    assert simulator.actions == [{"kind": "interact", "low_level_target": 7}]
    world.close()
    assert simulator.closed is True


def test_minestudio_adapter_rejects_semantic_or_nonopaque_outputs() -> None:
    class LeakingController(FakeVisualController):
        def perceive(
            self, frame: np.ndarray, info: dict[str, Any], *, seed: int
        ) -> VisualScene:
            del frame, info, seed
            return VisualScene(
                objects=(
                    ObjectObservation(
                        object_id="obj_safe",
                        affordances=(ActionKind.INTERACT,),
                        observable_features={"label": "diamond_ore"},
                    ),
                )
            )

    world = MineStudioVisualAdapter(
        LeakingController(), simulator_factory=lambda config: FakeSimulator()
    )
    with pytest.raises(ValueError, match="leaked semantic token"):
        world.reset(0)

    class NamedController(LeakingController):
        def perceive(
            self, frame: np.ndarray, info: dict[str, Any], *, seed: int
        ) -> VisualScene:
            del frame, info, seed
            return VisualScene(
                objects=(
                    ObjectObservation(
                        object_id="diamond_ore",
                        affordances=(ActionKind.INTERACT,),
                    ),
                )
            )

    named = MineStudioVisualAdapter(
        NamedController(), simulator_factory=lambda config: FakeSimulator()
    )
    with pytest.raises(ValueError, match="opaque obj_ prefix"):
        named.reset(0)


def test_grid_visual_controller_maps_pixels_to_env_actions_without_labels() -> None:
    controller = GridVisualController(rows=2, columns=2, hotbar_slots=2)
    frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape((6, 8, 3))
    scene = controller.perceive(
        frame,
        {"objects": [{"name": "forbidden_semantic_name"}]},
        seed=5,
    )

    assert len(scene.objects) == 6
    assert scene.inventory == {}
    assert scene.agent_state["inventory_observed"] is False
    assert all(item.object_id.startswith("obj_") for item in scene.objects)
    assert "forbidden" not in repr(scene.objects)
    assert scene.agent_state is not None
    assert scene.agent_state["appearance_encoder"] == RGBSceneEncoder.VERSION
    signature = RGBSceneEncoder().deserialize(
        str(scene.agent_state["appearance_signature"])
    )
    assert len(signature) == RGBSceneEncoder().dimensions
    slot = next(
        item for item in scene.objects if item.affordances == (ActionKind.SELECT,)
    )
    cell = next(
        item for item in scene.objects if ActionKind.INTERACT in item.affordances
    )
    select = controller.encode_action(
        PrimitiveAction(kind=ActionKind.SELECT, target_id=slot.object_id), scene
    )
    interact = controller.encode_action(
        PrimitiveAction(kind=ActionKind.INTERACT, target_id=cell.object_id), scene
    )
    look = controller.encode_action(
        PrimitiveAction(kind=ActionKind.LOOK, target_id=cell.object_id), scene
    )
    activate = controller.encode_action(
        PrimitiveAction(kind=ActionKind.ACTIVATE, target_id=cell.object_id), scene
    )

    assert sum(int(select[f"hotbar.{index}"]) for index in range(1, 10)) == 1
    assert int(interact["attack"]) == 1
    assert np.any(interact["camera"] != 0)
    assert int(look["attack"]) == 0 and int(look["use"]) == 0
    assert np.any(look["camera"] != 0)
    assert int(activate["use"]) == 1 and int(activate["sneak"]) == 1


def test_grid_visual_controller_exposes_opaque_movement_handles() -> None:
    controller = GridVisualController(
        rows=1,
        columns=1,
        hotbar_slots=1,
        movement_controls=("forward", "left"),
    )
    scene = controller.perceive(np.zeros((8, 8, 3), dtype=np.uint8), {}, seed=9)
    handles = [item for item in scene.objects if item.affordances == (ActionKind.MOVE,)]

    assert len(handles) == 2
    assert all(item.object_id.startswith("obj_") for item in handles)
    assert "forward" not in repr(handles) and "left" not in repr(handles)
    payloads = [
        controller.encode_action(
            PrimitiveAction(kind=ActionKind.MOVE, target_id=item.object_id), scene
        )
        for item in handles
    ]
    assert {
        key
        for payload in payloads
        for key in ("forward", "left")
        if int(payload[key]) == 1
    } == {"forward", "left"}
    assert all(
        int(payload["attack"]) == int(payload["use"]) == 0 for payload in payloads
    )


def test_grid_visual_controller_supports_opaque_compound_movement_programs() -> None:
    controller = GridVisualController(
        rows=1,
        columns=1,
        hotbar_slots=1,
        movement_programs=(("forward",), ("forward", "jump")),
    )
    scene = controller.perceive(np.zeros((8, 8, 3), dtype=np.uint8), {}, seed=19)
    handles = [item for item in scene.objects if item.affordances == (ActionKind.MOVE,)]

    assert len(handles) == 2
    assert "forward" not in repr(handles) and "jump" not in repr(handles)
    payloads = [
        controller.encode_action(
            PrimitiveAction(kind=ActionKind.MOVE, target_id=item.object_id), scene
        )
        for item in handles
    ]
    assert [(int(item["forward"]), int(item["jump"])) for item in payloads] == [
        (1, 0),
        (1, 1),
    ]


def test_grid_controller_rgb_signature_survives_visual_adapter_boundary() -> None:
    simulator = FakeSimulator()
    world = MineStudioVisualAdapter(
        GridVisualController(rows=1, columns=1, hotbar_slots=1),
        simulator_factory=lambda config: simulator,
    )

    observation = world.reset(12)
    signature = rgb_signature_from_observation(observation)

    assert len(signature) == RGBSceneEncoder().dimensions
    assert "diamond_ore" not in observation.model_dump_json()


@pytest.mark.parametrize(
    "controls",
    [("forward", "forward"), ("jump",)],
)
def test_grid_visual_controller_rejects_invalid_movement_controls(controls) -> None:
    with pytest.raises(ValueError, match="unique cardinal env actions"):
        GridVisualController(movement_controls=controls)


@pytest.mark.parametrize(
    "programs",
    [
        ((),),
        (("forward", "forward"),),
        (("forward", "fly"),),
        (("forward", "jump"), ("forward", "jump")),
    ],
)
def test_grid_visual_controller_rejects_invalid_movement_programs(programs) -> None:
    with pytest.raises(ValueError, match="unique nonempty env-action sets"):
        GridVisualController(movement_programs=programs)


def test_move_repeats_requested_simulator_frames() -> None:
    simulator = FakeSimulator()
    controller = GridVisualController(
        rows=1, columns=1, hotbar_slots=1, movement_controls=("forward",)
    )
    world = MineStudioVisualAdapter(
        controller, simulator_factory=lambda config: simulator
    )
    world.reset(2)
    handle = next(
        item
        for item in world.observe().objects
        if item.affordances == (ActionKind.MOVE,)
    )

    result = world.step(
        PrimitiveAction(kind=ActionKind.MOVE, target_id=handle.object_id, ticks=4)
    )

    assert len(simulator.actions) == 4
    assert result.event.duration == 4
    assert result.intervention_compliance


def test_move_stops_if_episode_ends_before_requested_duration() -> None:
    class TerminalSimulator(FakeSimulator):
        def step(self, action):
            frame, reward, _, truncated, info = super().step(action)
            return frame, reward, self.counter == 2, truncated, info

    simulator = TerminalSimulator()
    controller = GridVisualController(
        rows=1, columns=1, hotbar_slots=1, movement_controls=("forward",)
    )
    world = MineStudioVisualAdapter(
        controller, simulator_factory=lambda config: simulator
    )
    world.reset(3)
    handle = next(
        item
        for item in world.observe().objects
        if item.affordances == (ActionKind.MOVE,)
    )

    result = world.step(
        PrimitiveAction(kind=ActionKind.MOVE, target_id=handle.object_id, ticks=4)
    )

    assert len(simulator.actions) == 2
    assert result.event.duration == 2
    assert not result.intervention_compliance
    assert result.controller_failure == "episode_ended_before_requested_duration"
