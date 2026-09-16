"""Privileged hidden-intervention evaluation."""

from __future__ import annotations

from minecraft_scientist.protocols import (
    EvaluationReport,
    SymbolPrimitive,
    TheoryVersion,
)
from minecraft_scientist.theory import TheoryInterpreter
from minecraft_scientist.world import WorldEvaluator


def evaluate_run(
    *,
    seed: int,
    evaluator: WorldEvaluator,
    latest_theories: dict[str, TheoryVersion],
    symbols: tuple[SymbolPrimitive, ...],
    experiment_count: int,
    change_episode: int | None,
    change_detection_delay: int | None,
    revision_experiments: int,
) -> EvaluationReport:
    active_episode = change_episode if change_episode is not None else 0
    tool_theory = latest_theories["tool-capability"]
    tool_rows = [
        (
            bool(
                TheoryInterpreter.predict(
                    tool_theory.ir,
                    tool_id=tool,
                    target_id=target,
                )["success"]
            ),
            evaluator.tool_outcome(tool, target, active_episode),
        )
        for tool in evaluator.tools
        for target in evaluator.blocks
    ]
    tool_accuracy = sum(pred == true for pred, true in tool_rows) / len(tool_rows)

    machine_theory = latest_theories["machine-protocol"]
    true_input, true_protocol, true_output = evaluator.machine_rule()
    machine_accuracy = float(
        machine_theory.ir.parameters["input_id"] == true_input
        and tuple(machine_theory.ir.parameters["protocol"]) == true_protocol
        and machine_theory.ir.effects["output_id"] == true_output
    )

    chain_theory = latest_theories["transformation-chain"]
    learned_mapping = {
        processor: tuple(pair)
        for processor, pair in chain_theory.ir.parameters["mapping"].items()
    }
    true_mapping = evaluator.transformations()
    transform_accuracy = sum(
        learned_mapping.get(processor) == pair
        for processor, pair in true_mapping.items()
    ) / len(true_mapping)

    mechanism_accuracy = {
        "machine_protocol": machine_accuracy,
        "tool_capability": tool_accuracy,
        "transformation_chain": transform_accuracy,
    }
    overall = sum(mechanism_accuracy.values()) / len(mechanism_accuracy)
    raw_description_length = sum(
        TheoryInterpreter.description_length(theory.ir)
        for theory in latest_theories.values()
    )
    description_length = max(
        0,
        raw_description_length
        - int(sum(symbol.compression_gain for symbol in symbols)),
    )
    return EvaluationReport(
        seed=seed,
        hidden_intervention_accuracy=overall,
        experiment_count=experiment_count,
        experiment_efficiency=overall / max(experiment_count, 1),
        change_detection_delay=change_detection_delay,
        revision_experiments=revision_experiments,
        description_length=description_length,
        symbol_count=len(symbols),
        symbol_reuse=sum(symbol.cross_world_reuse for symbol in symbols),
        task_success_rate=overall,
        mechanism_accuracy=mechanism_accuracy,
        theory_versions={
            key: theory.version for key, theory in latest_theories.items()
        },
        notes=(
            "All identifiers exposed to the scientist are opaque.",
            "Hidden accuracy is computed by the privileged evaluator.",
            "Description length includes accepted library compression.",
        ),
    )
