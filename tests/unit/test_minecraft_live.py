from __future__ import annotations

from dataclasses import dataclass

from minecraft_scientist.minecraft.live import (
    LiveWorldEvaluator,
    ResettableMinecraftWorld,
)
from minecraft_scientist.minecraft.fall_live import _fall_safe_height, _fall_slope
from minecraft_scientist.minecraft.scenarios import ScientistCurriculumScenario
from minecraft_scientist.protocols import ActionKind, Observation, PrimitiveAction


@dataclass
class FakeChannel:
    values: list[str]

    def command(self, value: str) -> None:
        self.values.append(value)


class FakeAdapter:
    def __init__(self) -> None:
        self.actions: list[PrimitiveAction] = []
        names = (
            "wooden_pickaxe",
            "stone_pickaxe",
            "iron_pickaxe",
            "raw_copper",
            "oak_log",
            "oak_planks",
            "stick",
            "copper_ingot",
            "torch",
        )
        blocks = (
            "stone",
            "iron_ore",
            "diamond_ore",
            "lodestone",
            "chiseled_copper",
            "chiseled_tuff",
            "polished_blackstone",
        )
        self.ids = {
            **{("inventory", name): (f"item-{index}",) for index, name in enumerate(names)},
            **{("block", name): (f"block-{index}",) for index, name in enumerate(blocks)},
        }

    @staticmethod
    def observe() -> Observation:
        return Observation(episode_id=0, step=0, inventory={}, objects=())

    def step(self, action: PrimitiveAction):  # type: ignore[no-untyped-def]
        self.actions.append(action)
        return object()

    @staticmethod
    def available_actions() -> tuple[PrimitiveAction, ...]:
        return ()

    def _controller_ids_for(self, category: str, name: str) -> tuple[str, ...]:
        return self.ids.get((category, name), ())


def test_resettable_live_world_switches_variant_only_at_change_episode() -> None:
    adapter = FakeAdapter()
    channel = FakeChannel([])
    world = ResettableMinecraftWorld(
        adapter,  # type: ignore[arg-type]
        channel,  # type: ignore[arg-type]
        ScientistCurriculumScenario(seed=8),
        change_episode=1,
    )

    world.begin_episode(0)
    assert channel.values[:2] == ["scientist reset 8", "scientist variant standard"]
    channel.values.clear()
    world.begin_episode(1)
    assert channel.values[:2] == ["scientist reset 8", "scientist variant shifted"]
    assert adapter.actions[-1] == PrimitiveAction(kind=ActionKind.WAIT, ticks=5)


def test_live_evaluator_keeps_truth_outside_agent_observation() -> None:
    adapter = FakeAdapter()
    evaluator = LiveWorldEvaluator(adapter, change_episode=1)  # type: ignore[arg-type]

    assert evaluator.active_requirements(0) == {
        "block-0": 0,
        "block-1": 1,
        "block-2": 2,
    }
    assert evaluator.active_requirements(1) == {
        "block-0": 2,
        "block-1": 0,
        "block-2": 1,
    }
    assert evaluator.machine_rule() == (
        "item-3",
        ("place", "activate", "wait:1", "interact"),
        "item-7",
    )


def test_live_fall_law_shift_changes_breakpoint_and_slope_assignment() -> None:
    assert _fall_safe_height(0, False) == 2
    assert _fall_safe_height(0, True) == 3
    standard = tuple(_fall_slope(0, False, index) for index in range(3))
    shifted = tuple(_fall_slope(0, True, index) for index in range(3))
    assert standard == (0.4, 0.7, 1.0)
    assert shifted == (0.7, 1.0, 0.4)
