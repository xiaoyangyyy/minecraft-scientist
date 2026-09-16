"""Inspect one real Fabric fall-equation intervention end to end."""

from __future__ import annotations

import json
from pathlib import Path

from minecraft_scientist.evaluation.full import _public_inventory, _trial
from minecraft_scientist.minecraft.adapter import (
    MinecraftConnectionConfig,
    MinecraftWorldAdapter,
)
from minecraft_scientist.minecraft.fall_live import (
    LiveFallBundle,
    LiveFallEvaluator,
    LiveFallWorld,
)
from minecraft_scientist.minecraft.scenarios import FallEquationScenario
from minecraft_scientist.minecraft.server import create_server_process
from minecraft_scientist.symbolic.equation import FallEquationVersionSpace


def main() -> None:
    scenario = FallEquationScenario(seed=0)
    connection = MinecraftConnectionConfig(
        radius=12,
        max_blocks=512,
        allowed_block_positions=tuple(scenario.platform_positions().values()),
    )
    with (
        create_server_process(Path("mod/scientist-world-rules"), mode="gradle") as server,
        MinecraftWorldAdapter(connection) as adapter,
    ):
        adapter.reset(0)
        world = LiveFallWorld(adapter, server, scenario)
        evaluator = LiveFallEvaluator(world, adapter, seed=0, shifted=False)
        bundle = LiveFallBundle(agent_view=world, evaluator=evaluator)
        armor_ids, platforms = _public_inventory(world.observe())
        space = FallEquationVersionSpace(armor_ids, tuple(sorted(platforms)))
        samples: list[dict[str, object]] = []
        while not space.resolved():
            armor_id, height = space.next_experiment() or (armor_ids[0], 7)
            records, sample = _trial(bundle, armor_id, height, platforms[height])
            samples.append(
                {
                    "candidate": [armor_id, height],
                    "observed_damage": sample.outcome["damage"],
                    "expected_damage": evaluator.damage(armor_id, height),
                    "health": records[-1].state_after.agent_state["health"],
                    "food": records[-1].state_after.agent_state["food"],
                    "event": records[-1].event.model_dump(mode="json"),
                }
            )
            space.observe(
                armor_id,
                height,
                float(sample.outcome["damage"]),
                sample.sample_id,
            )
        print(
            json.dumps(
                {
                    "armor_ids": armor_ids,
                    "samples": samples,
                    "theory": space.to_ir().model_dump(mode="json"),
                    "server_tail": server.log_tail[-20:],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
