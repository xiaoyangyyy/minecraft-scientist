"""Paired, same-seed ablations for the scientist's major subsystems."""

from __future__ import annotations

import json
import hashlib
import math
import statistics
from pathlib import Path
from typing import Any

from minecraft_scientist.agent import ScientistAgent, ScientistRun
from minecraft_scientist.protocols import RunConfig, TheoryStatus, SymbolPrimitive
from minecraft_scientist.discovery import HeuristicHypothesisGenerator, PROTOCOL_CANDIDATES
from minecraft_scientist.planning import ExperimentExecutor
from minecraft_scientist.world import WorldFactory

def _transfer_probe(seed: int, symbols: tuple[SymbolPrimitive, ...], budget: int) -> int:
    """Execute the source's learned program in an independently named world."""
    bundle = WorldFactory.create(seed + 10_000, change_episode=None)
    observation = bundle.controller.begin_episode(0)
    groups = HeuristicHypothesisGenerator().classify(observation)
    primitive = next((s for s in symbols if s.name == "transform" and s.status == "accepted"), None)
    if primitive is None:
        protocols = tuple(sorted(PROTOCOL_CANDIDATES))
    else:
        protocols = (tuple(
            "place" if token == "place_input" else "wait:1" if token == "wait" else token
            for token in primitive.definition_ir["body"]
        ),)
    executor = ExperimentExecutor(bundle.agent_view)
    count = 0
    for protocol in protocols:
        for item in groups["resources"]:
            if count >= budget:
                raise RuntimeError("transfer intervention budget exhausted")
            count += 1
            result = executor.run_protocol_test(
                groups["processors"][0], item, protocol, mechanism="transformation_chain"
            )
            if result.compliant and result.outcome["success"]:
                return count
    raise RuntimeError("transfer probe failed")


def _summary(values: list[float]) -> dict[str, float | int | list[float]]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(values)) if values else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": std,
        "ci95": [mean - margin, mean + margin],
    }


def _compact(run: ScientistRun) -> dict[str, Any]:
    report = run.report
    tool_history = tuple(
        theory for theory in run.theories if theory.theory_id == "tool-capability"
    )
    return {
        "hidden_intervention_accuracy": report.hidden_intervention_accuracy,
        "experiment_count": report.experiment_count,
        "experiment_efficiency": report.experiment_efficiency,
        "task_success_rate": report.task_success_rate,
        "change_detection_delay": report.change_detection_delay,
        "revision_experiments": report.revision_experiments,
        "description_length": report.description_length,
        "symbol_count": report.symbol_count,
        "symbol_reuse": report.symbol_reuse,
        "tool_theory_versions": len(tool_history),
        "tool_theory_statuses": [theory.status.value for theory in tool_history],
        "crisis_completed": any(
            theory.status is TheoryStatus.SUPERSEDED for theory in tool_history
        ),
    }


def _run(seed: int, budget: int, **updates: Any) -> ScientistRun:
    config = RunConfig(
        seed=seed,
        intervention_budget=budget,
        change_episode=1,
        library_learning=True,
        theory_crisis=True,
        planner="information_gain",
    ).model_copy(update=updates)
    return ScientistAgent(config).run()


def run_ablation_suite(
    *,
    seed: int,
    seeds: int,
    intervention_budget: int,
    planner_checkpoint: Path,
    output: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Run matched pairs where exactly one named capability is withheld."""

    if seeds < 1:
        raise ValueError("ablation seed count must be positive")
    if not planner_checkpoint.is_file():
        raise FileNotFoundError(
            f"learned planner checkpoint does not exist: {planner_checkpoint}"
        )
    report_path = output / "ablation-matrix.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"ablation report already exists: {report_path}")

    values = tuple(seed + offset for offset in range(seeds))
    rows: list[dict[str, Any]] = []
    for current_seed in values:
        full = _run(current_seed, intervention_budget)
        no_active = _run(
            current_seed,
            intervention_budget,
            planner="random",
        )
        no_crisis = _run(
            current_seed,
            intervention_budget,
            theory_crisis=False,
        )
        no_library = _run(
            current_seed,
            intervention_budget,
            library_learning=False,
        )
        learned = _run(
            current_seed,
            intervention_budget,
            planner="learned",
            planner_checkpoint=str(planner_checkpoint),
        )
        memory_transfer = _transfer_probe(current_seed, full.symbols, intervention_budget)
        no_memory_transfer = _transfer_probe(current_seed, (), intervention_budget)
        rows.append(
            {
                "seed": current_seed,
                "full_information_gain": _compact(full),
                "no_active_planning": _compact(no_active),
                "no_theory_crisis": _compact(no_crisis),
                "no_library_learning": _compact(no_library),
                "learned_planner": _compact(learned),
                "cross_world_transfer": {
                    "with_memory_experiments": memory_transfer,
                    "without_memory_experiments": no_memory_transfer,
                },
            }
        )

    def deltas(left: str, right: str, metric: str) -> list[float]:
        return [
            float(row[left][metric]) - float(row[right][metric]) for row in rows
        ]

    payload: dict[str, Any] = {
        "suite": "strict-paired-ablations",
        "seed": seed,
        "seeds": list(values),
        "intervention_budget_per_condition": intervention_budget,
        "planner_checkpoint": str(planner_checkpoint),
        "planner_checkpoint_sha256": hashlib.sha256(planner_checkpoint.read_bytes()).hexdigest(),
        "design": {
            "paired_randomized_id_worlds": True,
            "same_seed": True,
            "same_intervention_budget": True,
            "one_named_capability_withheld_per_pair": True,
            "active_reference": "information_gain",
            "learned_planner_reference": "learned_planner",
            "confidence_interval": "normal approximation, paired seed differences",
            "random_selector": "uniform among untested informative actions",
            "crisis_ablation": "repair withheld; anomaly detection retained",
            "transfer_scope": "learned transform program on a new opaque-ID world",
        },
        "effects": {
            "active_information_gain": {
                "reference": "full_information_gain",
                "ablation": "no_active_planning",
                "experiment_savings": _summary(
                    deltas("no_active_planning", "full_information_gain", "experiment_count")
                ),
                "accuracy_gain": _summary(
                    deltas(
                        "full_information_gain",
                        "no_active_planning",
                        "hidden_intervention_accuracy",
                    )
                ),
            },
            "theory_crisis": {
                "reference": "full_information_gain",
                "ablation": "no_theory_crisis",
                "post_change_accuracy_recovery": _summary(
                    deltas(
                        "full_information_gain",
                        "no_theory_crisis",
                        "hidden_intervention_accuracy",
                    )
                ),
                "completed_crises": sum(
                    bool(row["full_information_gain"]["crisis_completed"])
                    for row in rows
                ),
                "ablation_completed_crises": sum(
                    bool(row["no_theory_crisis"]["crisis_completed"])
                    for row in rows
                ),
            },
            "library_learning": {
                "reference": "full_information_gain",
                "ablation": "no_library_learning",
                "description_length_reduction": _summary(
                    deltas(
                        "no_library_learning",
                        "full_information_gain",
                        "description_length",
                    )
                ),
                "symbol_gain": _summary(
                    deltas("full_information_gain", "no_library_learning", "symbol_count")
                ),
                "reuse_gain": _summary(
                    deltas("full_information_gain", "no_library_learning", "symbol_reuse")
                ),
            },
            "cross_world_memory": {
                "reference": "symbol_available",
                "ablation": "symbol_withheld",
                "transfer_experiment_savings": _summary(
                    [
                        float(row["cross_world_transfer"]["without_memory_experiments"])
                        - float(row["cross_world_transfer"]["with_memory_experiments"])
                        for row in rows
                    ]
                ),
            },
            "learned_planner": {
                "reference": "learned_planner",
                "ablation": "full_information_gain",
                "experiment_savings": _summary(
                    deltas("full_information_gain", "learned_planner", "experiment_count")
                ),
                "accuracy_gain": _summary(
                    deltas(
                        "learned_planner",
                        "full_information_gain",
                        "hidden_intervention_accuracy",
                    )
                ),
            },
        },
        "per_seed": rows,
    }
    payload["integrity"] = {
        "all_conditions_completed": len(rows) == seeds,
        "all_full_runs_succeeded": all(
            row["full_information_gain"]["task_success_rate"] == 1.0 for row in rows
        ),
        "all_learned_runs_succeeded": all(
            row["learned_planner"]["task_success_rate"] == 1.0 for row in rows
        ),
        "all_rule_shifts_detected_without_repair": all(
            "challenged" in row["no_theory_crisis"]["tool_theory_statuses"]
            for row in rows
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload
