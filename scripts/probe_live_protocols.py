"""Deterministic end-to-end probe for the live Fabric mechanism protocol.

This is intentionally separate from the offline test suite: it starts the real
Gradle/Fabric server, connects Mineflayer, and verifies that packet
acknowledgement latency cannot stand in for a symbolic ``wait`` intervention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minecraft_scientist.minecraft.adapter import (
    MinecraftConnectionConfig,
    MinecraftWorldAdapter,
)
from minecraft_scientist.minecraft.live import (
    LiveWorldEvaluator,
    ResettableMinecraftWorld,
)
from minecraft_scientist.minecraft.scenarios import ScientistCurriculumScenario
from minecraft_scientist.minecraft.server import create_server_process
from minecraft_scientist.planning.experiments import ExperimentExecutor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-directory",
        type=Path,
        default=Path("mod/scientist-world-rules"),
    )
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    scenario = ScientistCurriculumScenario(seed=arguments.seed)
    connection = MinecraftConnectionConfig(
        allowed_block_positions=scenario.experimental_positions()
    )
    results: list[dict[str, object]] = []
    with (
        create_server_process(arguments.project_directory, mode="gradle") as server,
        MinecraftWorldAdapter(connection) as adapter,
    ):
        adapter.reset(arguments.seed)
        world = ResettableMinecraftWorld(
            adapter,
            server,
            scenario,
            change_episode=None,
        )
        world.begin_episode(0)
        evaluator = LiveWorldEvaluator(adapter, change_episode=None)
        executor = ExperimentExecutor(world)
        machine_input = evaluator.resources[0]
        protocol = ("place", "activate", "wait:1", "interact")
        machine_output: str | None = None

        for repetition in range(3):
            experiment = executor.run_protocol_test(
                evaluator.machine_id,
                machine_input,
                protocol,
            )
            results.append(
                {
                    "case": f"machine-correct-{repetition + 1}",
                    "compliant": experiment.compliant,
                    **experiment.outcome,
                }
            )
            if machine_output is None:
                machine_output = experiment.outcome["output_id"]

        no_wait = tuple(token for token in protocol if not token.startswith("wait:"))
        experiment = executor.run_protocol_test(
            evaluator.machine_id,
            machine_input,
            no_wait,
        )
        results.append(
            {"case": "machine-no-wait", "compliant": experiment.compliant, **experiment.outcome}
        )

        processors = evaluator.processors
        resources = evaluator.resources
        transform_cases = (
            (processors[0], resources[1], resources[2]),
            (processors[1], resources[2], resources[3]),
            (processors[2], resources[3], None),
        )
        for processor, input_id, output_id in transform_cases:
            experiment = executor.run_protocol_test(
                processor,
                input_id,
                protocol,
                mechanism="transformation",
            )
            results.append(
                {
                    "case": f"transformation-{processor}",
                    "expected_output_id": output_id,
                    "compliant": experiment.compliant,
                    **experiment.outcome,
                }
            )

    failures = [
        result
        for result in results
        if not result["compliant"]
        or (
            result["case"] == "machine-no-wait"
            and (result["success"] or result["output_id"] is not None)
        )
        or (
            result["case"] != "machine-no-wait"
            and (
                not result["success"]
                or result["output_id"] is None
                or (
                    result.get("expected_output_id", machine_output) is not None
                    and result["output_id"]
                    != result.get("expected_output_id", machine_output)
                )
            )
        )
    ]
    print(json.dumps({"results": results, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
