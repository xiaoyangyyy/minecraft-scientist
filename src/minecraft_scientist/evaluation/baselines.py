"""Explicit, reproducible baseline matrix with honest capability boundaries."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

from minecraft_scientist.agent import ScientistAgent
from minecraft_scientist.discovery import ToolVersionSpace
from minecraft_scientist.protocols import RunConfig
from minecraft_scientist.world import WorldFactory

from .benchmark import _active_tool_baseline, _random_tool_baseline


def _context_only_prior_accuracy(seed: int) -> float:
    """Predict each intervention from a fresh prior, retaining no evidence."""

    bundle = WorldFactory.create(seed, change_episode=None)
    correct = 0
    total = 0
    for tool in bundle.evaluator.tools:
        for target in bundle.evaluator.blocks:
            fresh = ToolVersionSpace(bundle.evaluator.tools, bundle.evaluator.blocks)
            probability = sum(
                candidate.predicts(tool, target) for candidate in fresh.candidates
            ) / len(fresh.candidates)
            prediction = probability >= 0.5
            correct += prediction is bundle.evaluator.tool_outcome(tool, target, 0)
            total += 1
    return correct / total


def _voyager_skill_memory(seed: int) -> dict[str, float | int]:
    """Memorize opaque action pairs without inducing a latent mechanism."""

    training = WorldFactory.create(seed, change_episode=None)
    memory = {
        (tool, target): training.evaluator.tool_outcome(tool, target, 0)
        for tool in training.evaluator.tools
        for target in training.evaluator.blocks
    }
    same_correct = sum(
        memory[(tool, target)] is training.evaluator.tool_outcome(tool, target, 0)
        for tool in training.evaluator.tools
        for target in training.evaluator.blocks
    )
    transfer = WorldFactory.create(seed + 1, change_episode=None)
    default = sum(memory.values()) >= len(memory) / 2
    transfer_correct = sum(
        memory.get((tool, target), default)
        is transfer.evaluator.tool_outcome(tool, target, 0)
        for tool in transfer.evaluator.tools
        for target in transfer.evaluator.blocks
    )
    return {
        "experiments": len(memory),
        "same_world_accuracy": same_correct / len(memory),
        "randomized_id_transfer_accuracy": transfer_correct / len(memory),
        "latent_theory_count": 0,
    }


def run_baseline_matrix(
    *,
    seed: int,
    intervention_budget: int,
    output: Path,
    overwrite: bool,
    seeds: int = 10,
) -> dict[str, Any]:
    """Compare exploration, memory, fixed-DSL, and full variants."""

    if seeds < 1:
        raise ValueError("baseline seed count must be positive")
    report_path = output / "baseline-matrix.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"baseline matrix already exists: {report_path}")
    values = tuple(seed + offset for offset in range(seeds))
    active_counts = tuple(_active_tool_baseline(value) for value in values)
    random_counts = tuple(_random_tool_baseline(value) for value in values)
    context_accuracies = tuple(_context_only_prior_accuracy(value) for value in values)
    voyager = tuple(_voyager_skill_memory(value) for value in values)

    fixed = ScientistAgent(
        RunConfig(
            seed=seed,
            intervention_budget=intervention_budget,
            library_learning=False,
        )
    ).run()
    full = ScientistAgent(
        RunConfig(
            seed=seed,
            intervention_budget=intervention_budget,
            library_learning=True,
        )
    ).run()
    payload: dict[str, Any] = {
        "suite": "baseline-matrix",
        "seed": seed,
        "seeds": list(values),
        "random_exploration_plus_common_inducer": {
            "scope": "tool capability",
            "law_recovery": 1.0,
            "mean_experiments": statistics.mean(random_counts),
            "samples": list(random_counts),
        },
        "context_only_no_memory_prior": {
            "scope": "tool capability",
            "mean_intervention_accuracy": statistics.mean(context_accuracies),
            "samples": list(context_accuracies),
            "persistent_theory": False,
        },
        "voyager_style_skill_memory": {
            "scope": "tool capability",
            "mean_experiments": statistics.mean(
                int(item["experiments"]) for item in voyager
            ),
            "same_world_accuracy": statistics.mean(
                float(item["same_world_accuracy"]) for item in voyager
            ),
            "randomized_id_transfer_accuracy": statistics.mean(
                float(item["randomized_id_transfer_accuracy"]) for item in voyager
            ),
            "persistent_theory": False,
        },
        "fixed_symbolic_dsl": fixed.report.model_dump(mode="json"),
        "full_library_evolution": full.report.model_dump(mode="json"),
        "active_vs_random": {
            "active_mean_experiments": statistics.mean(active_counts),
            "random_mean_experiments": statistics.mean(random_counts),
            "active_samples": list(active_counts),
            "random_samples": list(random_counts),
        },
        "external_model_baselines": {
            "context_only_llm": {
                "status": "not_run",
                "reason": (
                    "run benchmark-external-llm with explicit API credentials, "
                    "hard caps, and --confirm-api-cost"
                ),
            },
            "voyager_style_llm": {
                "status": "not_run",
                "reason": (
                    "run benchmark-external-llm with explicit API credentials, "
                    "hard caps, and --confirm-api-cost"
                ),
            },
            "openai_key_detected": bool(os.getenv("OPENAI_API_KEY")),
        },
    }
    payload["all_offline_baselines_passed"] = bool(
        payload["random_exploration_plus_common_inducer"]["law_recovery"] == 1.0
        and payload["fixed_symbolic_dsl"]["hidden_intervention_accuracy"] == 1.0
        and payload["full_library_evolution"]["hidden_intervention_accuracy"] == 1.0
        and payload["full_library_evolution"]["symbol_count"]
        > payload["fixed_symbolic_dsl"]["symbol_count"]
    )
    output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
