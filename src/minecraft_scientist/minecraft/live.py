"""Live Fabric/Mineflayer execution of the symbolic discovery loop."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from minecraft_scientist.protocols import (
    ActionKind,
    Observation,
    PrimitiveAction,
    RunConfig,
    TransitionRecord,
)

from .adapter import MinecraftConnectionConfig, MinecraftWorldAdapter
from .scenarios import ScientistCurriculumScenario
from .server import MinecraftServerProcess, create_server_process


class ResettableMinecraftWorld:
    """Agent view whose trial resets are performed through a privileged channel."""

    __slots__ = (
        "__adapter",
        "__channel",
        "__base_scenario",
        "__change_episode",
        "__episode",
        "__settle_ticks",
    )

    def __init__(
        self,
        adapter: MinecraftWorldAdapter,
        channel: MinecraftServerProcess,
        scenario: ScientistCurriculumScenario,
        *,
        change_episode: int | None,
        settle_ticks: int = 5,
    ) -> None:
        self.__adapter = adapter
        self.__channel = channel
        self.__base_scenario = scenario
        self.__change_episode = change_episode
        self.__episode = 0
        self.__settle_ticks = settle_ticks

    def begin_episode(self, episode_id: int) -> Observation:
        self.__episode = episode_id
        return self.begin_trial()

    def begin_trial(self) -> Observation:
        shifted = (
            self.__change_episode is not None
            and self.__episode >= self.__change_episode
        )
        scenario = replace(
            self.__base_scenario,
            variant="shifted" if shifted else "standard",
        )
        scenario.apply(self.__channel)
        self.__adapter.step(
            PrimitiveAction(kind=ActionKind.WAIT, ticks=self.__settle_ticks)
        )
        return self.__adapter.observe()

    def observe(self) -> Observation:
        return self.__adapter.observe()

    def available_actions(self) -> tuple[PrimitiveAction, ...]:
        return self.__adapter.available_actions()

    def step(self, action: PrimitiveAction) -> TransitionRecord:
        return self.__adapter.step(action)


class LiveWorldController:
    __slots__ = ("__world",)

    def __init__(self, world: ResettableMinecraftWorld) -> None:
        self.__world = world

    def begin_episode(self, episode_id: int) -> Observation:
        return self.__world.begin_episode(episode_id)


class LiveWorldEvaluator:
    """Privileged semantic binding used only after agent-side discovery."""

    __slots__ = ("__adapter", "__change_episode")

    def __init__(
        self,
        adapter: MinecraftWorldAdapter,
        *,
        change_episode: int | None,
    ) -> None:
        self.__adapter = adapter
        self.__change_episode = change_episode

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(
            self.__one("inventory", name)
            for name in ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe")
        )

    @property
    def blocks(self) -> tuple[str, ...]:
        return tuple(
            self.__one("block", name)
            for name in ("stone", "iron_ore", "diamond_ore")
        )

    @property
    def processors(self) -> tuple[str, ...]:
        return tuple(
            self.__one("block", name)
            for name in ("chiseled_copper", "chiseled_tuff", "polished_blackstone")
        )

    @property
    def resources(self) -> tuple[str, ...]:
        return tuple(
            self.__one("inventory", name)
            for name in ("raw_copper", "oak_log", "oak_planks", "stick")
        )

    @property
    def machine_id(self) -> str:
        return self.__one("block", "lodestone")

    def tool_outcome(self, tool_id: str, block_id: str, episode_id: int) -> bool:
        ranks = {tool: rank for rank, tool in enumerate(self.tools)}
        return ranks[tool_id] >= self.active_requirements(episode_id)[block_id]

    def active_requirements(self, episode_id: int) -> dict[str, int]:
        values = (0, 1, 2)
        if self.__change_episode is not None and episode_id >= self.__change_episode:
            values = (2, 0, 1)
        return dict(zip(self.blocks, values, strict=True))

    def machine_rule(self) -> tuple[str, tuple[str, ...], str]:
        return (
            self.__one("inventory", "raw_copper"),
            ("place", "activate", "wait:1", "interact"),
            self.__one("inventory", "copper_ingot"),
        )

    def transformations(self) -> dict[str, tuple[str, str]]:
        return {
            self.__one("block", "chiseled_copper"): (
                self.__one("inventory", "oak_log"),
                self.__one("inventory", "oak_planks"),
            ),
            self.__one("block", "chiseled_tuff"): (
                self.__one("inventory", "oak_planks"),
                self.__one("inventory", "stick"),
            ),
            self.__one("block", "polished_blackstone"): (
                self.__one("inventory", "stick"),
                self.__one("inventory", "torch"),
            ),
        }

    def __one(self, category: str, name: str) -> str:
        matches = self.__adapter._controller_ids_for(category, name)
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one privileged {category} binding for {name}, got {matches}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class LiveWorldBundle:
    agent_view: ResettableMinecraftWorld
    controller: LiveWorldController
    evaluator: LiveWorldEvaluator


def run_live_discovery(
    *,
    server_directory: Path,
    output: Path,
    seed: int,
    intervention_budget: int = 80,
    change_episode: int | None = 1,
    username: str = "ScientistBot",
    overwrite: bool = False,
    server_mode: str = "fabric",
    planner: str = "information_gain",
    planner_checkpoint: Path | None = None,
) -> Any:
    """Run the regular scientist against the real Fabric-backed arena."""

    from minecraft_scientist.agent import ScientistAgent

    scenario = ScientistCurriculumScenario(username=username, seed=seed)
    connection = MinecraftConnectionConfig(
        username=username,
        allowed_block_positions=scenario.experimental_positions(),
    )
    with (
        create_server_process(server_directory, mode=server_mode) as server,
        MinecraftWorldAdapter(connection) as adapter,
    ):
        adapter.reset(seed)
        agent_view = ResettableMinecraftWorld(
            adapter,
            server,
            scenario,
            change_episode=change_episode,
        )
        bundle = LiveWorldBundle(
            agent_view=agent_view,
            controller=LiveWorldController(agent_view),
            evaluator=LiveWorldEvaluator(adapter, change_episode=change_episode),
        )
        config = RunConfig(
            seed=seed,
            intervention_budget=intervention_budget,
            change_episode=change_episode,
            world_name="minecraft-live",
            planner=planner,
            planner_checkpoint=(
                str(planner_checkpoint) if planner_checkpoint is not None else None
            ),
        )
        agent = ScientistAgent(config, bundle=bundle)  # type: ignore[arg-type]
        try:
            result = agent.run()
        except Exception as exc:
            output.mkdir(parents=True, exist_ok=True)
            (output / "partial-events.jsonl").write_text(
                "".join(
                    json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
                    for item in agent.experiments
                ),
                encoding="utf-8",
            )
            (output / "partial-theories.jsonl").write_text(
                "".join(
                    json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
                    for item in agent.registry.all_versions()
                ),
                encoding="utf-8",
            )
            (output / "failure.json").write_text(
                json.dumps(
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "experiment_count": len(agent.experiments),
                        "server_log_tail": list(server.log_tail),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raise
        result.persist(output, overwrite=overwrite)
        return result
