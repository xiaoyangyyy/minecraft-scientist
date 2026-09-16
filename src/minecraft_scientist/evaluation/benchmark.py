"""Reproducible MVP benchmark orchestration."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from minecraft_scientist.agent import ScientistAgent
from minecraft_scientist.discovery import (
    PROTOCOL_CANDIDATES,
    HeuristicHypothesisGenerator,
    ToolVersionSpace,
)
from minecraft_scientist.planning import ExperimentExecutor
from minecraft_scientist.protocols import RunConfig
from minecraft_scientist.world import WorldFactory


def _random_tool_baseline(seed: int) -> int:
    bundle = WorldFactory.create(seed, change_episode=None)
    space = ToolVersionSpace(bundle.evaluator.tools, bundle.evaluator.blocks)
    pairs = [
        (tool, target)
        for tool in bundle.evaluator.tools
        for target in bundle.evaluator.blocks
    ]
    random.Random(seed).shuffle(pairs)
    count = 0
    for tool, target in pairs:
        if space.resolved():
            break
        count += 1
        space.observe(
            tool,
            target,
            bundle.evaluator.tool_outcome(tool, target, 0),
            f"random-{count:03d}",
        )
    return count


def _active_tool_baseline(seed: int) -> int:
    bundle = WorldFactory.create(seed, change_episode=None)
    space = ToolVersionSpace(bundle.evaluator.tools, bundle.evaluator.blocks)
    count = 0
    while not space.resolved():
        pair = space.next_experiment()
        if pair is None:
            break
        count += 1
        space.observe(
            pair[0],
            pair[1],
            bundle.evaluator.tool_outcome(pair[0], pair[1], 0),
            f"active-{count:03d}",
        )
    return count


def _transfer_experiments(seed: int, *, symbol_available: bool) -> int:
    bundle = WorldFactory.create(seed + 10_000, change_episode=None)
    bundle.controller.begin_episode(0)
    groups = HeuristicHypothesisGenerator().classify(bundle.agent_view.observe())
    processor = next(iter(groups["processors"]))
    resources = tuple(groups["resources"])
    canonical = ("place", "activate", "wait:1", "interact")
    protocols = (canonical,) if symbol_available else tuple(sorted(PROTOCOL_CANDIDATES))
    executor = ExperimentExecutor(bundle.agent_view)
    count = 0
    for protocol in protocols:
        for item in resources:
            count += 1
            result = executor.run_protocol_test(
                processor,
                item,
                protocol,
                mechanism="transformation_chain",
            )
            if result.outcome["success"]:
                return count
    raise RuntimeError("transfer probe did not find the transformation")


def run_mvp_benchmark(
    *,
    seed: int,
    intervention_budget: int,
    output: Path,
    overwrite: bool,
    planner: str = "information_gain",
    planner_checkpoint: str | None = None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    full_config = RunConfig(
        seed=seed,
        intervention_budget=intervention_budget,
        library_learning=True,
        planner=planner,
        planner_checkpoint=planner_checkpoint,
    )
    ablation_config = full_config.model_copy(update={"library_learning": False})
    full = ScientistAgent(full_config).run()
    ablation = ScientistAgent(ablation_config).run()
    randomized_reports = [
        ScientistAgent(full_config.model_copy(update={"seed": current_seed}))
        .run()
        .report
        for current_seed in (seed, seed + 1, seed + 2)
    ]
    baseline_seeds = (seed, seed + 1, seed + 2)
    active_tool_samples = [_active_tool_baseline(value) for value in baseline_seeds]
    random_tool_samples = [_random_tool_baseline(value) for value in baseline_seeds]
    full_transfer = _transfer_experiments(seed, symbol_available=True)
    ablation_transfer = _transfer_experiments(seed, symbol_available=False)
    full.persist(output / "full", overwrite=overwrite)
    ablation.persist(output / "no-library", overwrite=overwrite)
    payload: dict[str, object] = {
        "suite": "mvp",
        "seed": seed,
        "full": full.report.model_dump(mode="json"),
        "no_library": ablation.report.model_dump(mode="json"),
        "ablation": {
            "symbol_count_delta": full.report.symbol_count
            - ablation.report.symbol_count,
            "symbol_reuse_delta": full.report.symbol_reuse
            - ablation.report.symbol_reuse,
            "description_length_delta": (
                full.report.description_length - ablation.report.description_length
            ),
            "full_transfer_experiments": full_transfer,
            "no_library_transfer_experiments": ablation_transfer,
            "downstream_experiment_savings": ablation_transfer - full_transfer,
        },
        "baselines": {
            "seeds": list(baseline_seeds),
            "active_tool_discovery_experiments": statistics.mean(active_tool_samples),
            "random_tool_discovery_experiments": statistics.mean(random_tool_samples),
            "active_samples": active_tool_samples,
            "random_samples": random_tool_samples,
        },
        "randomized_id_suite": {
            "seeds": [report.seed for report in randomized_reports],
            "hidden_intervention_accuracy": [
                report.hidden_intervention_accuracy for report in randomized_reports
            ],
            "all_passed": all(
                report.hidden_intervention_accuracy == 1.0
                for report in randomized_reports
            ),
        },
    }
    report_path = output / "benchmark.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"benchmark report already exists: {report_path}")
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload
