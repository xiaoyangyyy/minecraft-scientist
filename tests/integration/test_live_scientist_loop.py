from __future__ import annotations

from typing import Any

from minecraft_scientist.agent import ScientistAgent
from minecraft_scientist.minecraft import MinecraftConnectionConfig, MinecraftWorldAdapter
from minecraft_scientist.minecraft.fall_live import (
    LiveFallBundle,
    LiveFallEvaluator,
    LiveFallWorld,
)
from minecraft_scientist.minecraft.live import (
    LiveWorldBundle,
    LiveWorldController,
    LiveWorldEvaluator,
    ResettableMinecraftWorld,
)
from minecraft_scientist.minecraft.scenarios import ScientistCurriculumScenario
from minecraft_scientist.protocols import RunConfig, TheoryStatus


ITEM_IDS = {
    name: index + 100
    for index, name in enumerate(
        (
            "wooden_pickaxe",
            "stone_pickaxe",
            "iron_pickaxe",
            "raw_copper",
            "oak_log",
            "oak_planks",
            "stick",
            "copper_ingot",
            "gold_ingot",
            "torch",
            "leather_boots",
            "chainmail_boots",
            "iron_boots",
        )
    )
}
BLOCK_IDS = {
    name: index + 200
    for index, name in enumerate(
        (
            "stone",
            "iron_ore",
            "diamond_ore",
            "lodestone",
            "copper_block",
            "orange_glazed_terracotta",
            "emerald_block",
            "chiseled_copper",
            "chiseled_tuff",
            "polished_blackstone",
            "white_concrete",
        )
    )
}


class FakeFabricState:
    def __init__(self) -> None:
        self.variant = "standard"
        self.seed = 0
        self.tick = 0
        self.health = 20.0
        self.blocks: dict[tuple[int, int, int], str] = {}
        self.inventory: dict[str, int] = {}
        self.selected: str | None = None
        self.machine_ready: dict[tuple[int, int, int], tuple[str, int]] = {}
        self.transforms: dict[tuple[int, int, int], tuple[str, str, int]] = {}

    def command(self, value: str) -> None:
        parts = value.split()
        if parts[:2] == ["scientist", "reset"]:
            self.seed = int(parts[2])
            self.variant = "standard"
            self.machine_ready.clear()
            self.transforms.clear()
        elif parts[:2] == ["scientist", "variant"]:
            self.variant = parts[2]
        elif parts[:2] == ["scientist", "fall"]:
            armor = int(parts[3])
            height = int(parts[4])
            standard_safe = 2 + (self.seed % 3)
            safe = (
                {2: 3, 3: 4, 4: 2}[standard_safe]
                if self.variant == "shifted"
                else standard_safe
            )
            slopes = (0.4, 0.7, 1.0)
            rotation = int(self.seed / 3) % 3
            slope = slopes[(armor + rotation + int(self.variant == "shifted")) % 3]
            self.health = round(20.0 - slope * max(0, height - safe), 3)
        elif parts[0] == "clear":
            self.inventory.clear()
            self.selected = None
            self.health = 20.0
        elif parts[0] == "setblock":
            position = tuple(int(value) for value in parts[1:4])
            self.blocks[position] = parts[4]
        elif parts[0] == "give":
            self.inventory[parts[2]] = self.inventory.get(parts[2], 0) + int(parts[3])

    def advance(self, ticks: int = 1) -> None:
        self.tick += ticks
        for position, (_output, ready) in tuple(self.machine_ready.items()):
            if self.tick >= ready:
                self.blocks[position] = "emerald_block"

    def act(self, action: dict[str, Any]) -> bool:
        kind = action["kind"]
        if kind == "wait":
            self.advance(int(action.get("ticks", 1)))
            return True
        self.advance()
        if kind == "select_slot":
            slot_map = {
                index: name
                for index, name in enumerate(sorted(self.inventory), start=36)
                if self.inventory[name] > 0
            }
            self.selected = slot_map[int(action["slot"])]
            return True
        position = tuple(int(action[axis]) for axis in ("x", "y", "z"))
        if kind == "dig":
            target = self.blocks[position]
            ranks = {"wooden_pickaxe": 0, "stone_pickaxe": 1, "iron_pickaxe": 2}
            requirements = (
                {"stone": 0, "iron_ore": 1, "diamond_ore": 2}
                if self.variant == "standard"
                else {"stone": 2, "iron_ore": 0, "diamond_ore": 1}
            )
            allowed = ranks.get(self.selected or "", -1) >= requirements[target]
            if allowed:
                del self.blocks[position]
            return allowed
        if kind != "use_block":
            return True
        intent = action.get("intent")
        block = self.blocks[position]
        machine_input = "raw_copper" if self.variant == "standard" else "oak_log"
        machine_output = "copper_ingot" if self.variant == "standard" else "gold_ingot"
        if block == "lodestone" and intent == "place_input" and self.selected == machine_input:
            self._consume(self.selected)
            self.blocks[position] = "copper_block"
            return True
        if block == "copper_block" and intent == "activate":
            self.blocks[position] = "orange_glazed_terracotta"
            self.machine_ready[position] = (machine_output, self.tick + 2)
            return True
        if block == "emerald_block" and intent == "interact":
            output, _ready = self.machine_ready.pop(position)
            self._grant(output)
            self.blocks[position] = "lodestone"
            return True
        transforms = self._transform_specification()
        if block not in transforms:
            return False
        expected, output = transforms[block]
        active = self.transforms.get(position)
        if active is None and intent == "place_input" and self.selected == expected:
            self._consume(self.selected)
            self.transforms[position] = ("loaded", output, -1)
            return True
        if active is not None and active[0] == "loaded" and intent == "activate":
            self.transforms[position] = ("active", output, self.tick + 2)
            return True
        if (
            active is not None
            and active[0] == "active"
            and intent == "interact"
            and self.tick >= active[2]
        ):
            self._grant(active[1])
            del self.transforms[position]
            return True
        return False

    def _transform_specification(self) -> dict[str, tuple[str, str]]:
        if self.variant == "standard":
            return {
                "chiseled_copper": ("oak_log", "oak_planks"),
                "chiseled_tuff": ("oak_planks", "stick"),
                "polished_blackstone": ("stick", "torch"),
            }
        return {
            "chiseled_copper": ("stick", "torch"),
            "chiseled_tuff": ("oak_log", "oak_planks"),
            "polished_blackstone": ("oak_planks", "stick"),
        }

    def _consume(self, name: str) -> None:
        self.inventory[name] -= 1

    def _grant(self, name: str) -> None:
        self.inventory[name] = self.inventory.get(name, 0) + 1


class FakeServer:
    def __init__(self, state: FakeFabricState) -> None:
        self.state = state

    def command(self, value: str) -> None:
        self.state.command(value)


class FakeFabricBridge:
    def __init__(self, state: FakeFabricState) -> None:
        self.state = state

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del timeout
        params = params or {}
        if method == "connect":
            return {"username": "ScientistBot", "version": "1.21.11"}
        if method == "observe":
            return self._observation()
        if method == "act":
            success = self.state.act(params["action"])
            return {
                "action": params["action"]["kind"],
                "success": success,
                "observation": self._observation(),
            }
        raise AssertionError(method)

    def close(self) -> None:
        pass

    def _observation(self) -> dict[str, Any]:
        slot_by_name = {
            name: index
            for index, name in enumerate(sorted(self.state.inventory), start=36)
            if self.state.inventory[name] > 0
        }
        items = [
            {
                "id": ITEM_IDS[name],
                "name": name,
                "count": count,
                "slot": slot_by_name[name],
                "max_durability": 100 if name.endswith("pickaxe") else 0,
            }
            for name, count in sorted(self.state.inventory.items())
            if count > 0
        ]
        held = next(
            (item for item in items if item["name"] == self.state.selected),
            None,
        )
        return {
            "step": self.state.tick,
            "position": {"x": 0.0, "y": 80.0, "z": 0.0},
            "health": self.state.health,
            "food": 20.0,
            "time": 6000,
            "inventory": items,
            "held_item": held,
            "blocks": [
                {
                    "id": BLOCK_IDS[name],
                    "name": name,
                    "position": {"x": x, "y": y, "z": z},
                    "diggable": True,
                    "hardness": 1.0,
                }
                for (x, y, z), name in sorted(self.state.blocks.items())
            ],
            "entities": [],
        }


def test_regular_scientist_recovers_fake_fabric_mechanisms_end_to_end() -> None:
    state = FakeFabricState()
    server = FakeServer(state)
    bridge = FakeFabricBridge(state)
    scenario = ScientistCurriculumScenario(seed=0)
    adapter = MinecraftWorldAdapter(
        MinecraftConnectionConfig(
            allowed_block_positions=scenario.experimental_positions()
        ),
        bridge_factory=lambda: bridge,
    )
    adapter.reset(0)
    world = ResettableMinecraftWorld(
        adapter,
        server,  # type: ignore[arg-type]
        scenario,
        change_episode=1,
        settle_ticks=1,
    )
    bundle = LiveWorldBundle(
        agent_view=world,
        controller=LiveWorldController(world),
        evaluator=LiveWorldEvaluator(adapter, change_episode=1),
    )

    result = ScientistAgent(
        RunConfig(seed=0, intervention_budget=80, change_episode=1),
        bundle=bundle,  # type: ignore[arg-type]
    ).run()

    assert result.report.hidden_intervention_accuracy == 1.0
    assert result.report.task_success_rate == 1.0
    tool_history = [
        theory for theory in result.theories if theory.theory_id == "tool-capability"
    ]
    assert [theory.status for theory in tool_history] == [
        TheoryStatus.SUPERSEDED,
        TheoryStatus.ACCEPTED,
    ]


def test_live_fall_view_recovers_standard_and_shifted_equations() -> None:
    from minecraft_scientist.evaluation.full import (
        _behavioral_accuracy,
        _discover_fall,
    )
    from minecraft_scientist.minecraft.scenarios import FallEquationScenario

    state = FakeFabricState()
    server = FakeServer(state)
    bridge = FakeFabricBridge(state)
    standard_scenario = FallEquationScenario(seed=7, variant="standard")
    adapter = MinecraftWorldAdapter(
        MinecraftConnectionConfig(
            radius=12,
            allowed_block_positions=tuple(
                standard_scenario.platform_positions().values()
            ),
        ),
        bridge_factory=lambda: bridge,
    )
    adapter.reset(7)
    standard_world = LiveFallWorld(adapter, server, standard_scenario)  # type: ignore[arg-type]
    standard = LiveFallBundle(
        agent_view=standard_world,
        evaluator=LiveFallEvaluator(
            standard_world,
            adapter,
            seed=7,
            shifted=False,
        ),
    )
    standard_discovery = _discover_fall(standard, budget=40)
    assert _behavioral_accuracy(standard_discovery.ir, standard) == 1.0

    shifted_scenario = FallEquationScenario(seed=7, variant="shifted")
    shifted_world = LiveFallWorld(adapter, server, shifted_scenario)  # type: ignore[arg-type]
    shifted = LiveFallBundle(
        agent_view=shifted_world,
        evaluator=LiveFallEvaluator(
            shifted_world,
            adapter,
            seed=7,
            shifted=True,
        ),
    )
    shifted_discovery = _discover_fall(shifted, budget=40)
    assert _behavioral_accuracy(shifted_discovery.ir, shifted) == 1.0
    assert standard_discovery.ir.parameters != shifted_discovery.ir.parameters
