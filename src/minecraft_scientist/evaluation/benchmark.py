"""Reproducible MVP benchmark orchestration."""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from minecraft_scientist.agent import ScientistAgent
from minecraft_scientist.discovery import (
    PROTOCOL_CANDIDATES,
    HeuristicHypothesisGenerator,
    ToolVersionSpace,
)
from minecraft_scientist.planning import ExperimentExecutor
from minecraft_scientist.protocols import ExperimentResult, RunConfig, SymbolPrimitive
from minecraft_scientist.world import WorldFactory

_TRANSFER_FILES = (
    "library-transfer-events.jsonl",
    "no-library-transfer-events.jsonl",
    "transfer-evidence.json",
)


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


@dataclass(frozen=True, slots=True)
class _TransferProbe:
    experiments: tuple[ExperimentResult, ...]
    symbol: SymbolPrimitive | None
    protocols: tuple[tuple[str, ...], ...]


def _primitive_protocol(primitive: SymbolPrimitive) -> tuple[str, ...]:
    definition = primitive.definition_ir
    body = definition.get("body")
    if (
        primitive.status != "accepted"
        or not primitive.validation_examples
        or primitive.type_signature != "(processor, input) -> output"
        or definition.get("parameters") != ["processor", "input"]
        or definition.get("returns") != "output"
        or not isinstance(body, (list, tuple))
        or not 1 <= len(body) <= 32
        or any(
            token not in ("place_input", "activate", "wait", "interact")
            for token in body
        )
    ):
        raise ValueError("transfer requires a validated executable transform primitive")
    # This compiles the learned program in its actual order. It does not insert
    # a known successful recipe. WAIT's default one tick is the primitive IR's
    # current action contract, not a hidden-world timing parameter.
    return tuple(
        "place" if token == "place_input" else "wait:1" if token == "wait" else token
        for token in body
    )


def _run_transfer_probe(
    seed: int, *, primitive: SymbolPrimitive | None
) -> _TransferProbe:
    protocols = (
        (_primitive_protocol(primitive),)
        if primitive is not None
        else tuple(sorted(PROTOCOL_CANDIDATES))
    )
    bundle = WorldFactory.create(seed + 10_000, change_episode=None)
    bundle.controller.begin_episode(0)
    groups = HeuristicHypothesisGenerator().classify(bundle.agent_view.observe())
    processor = next(iter(groups["processors"]))
    resources = tuple(groups["resources"])
    executor = ExperimentExecutor(bundle.agent_view)
    experiments = []
    for protocol in protocols:
        for item in resources:
            result = executor.run_protocol_test(
                processor,
                item,
                protocol,
                mechanism="transformation_chain",
            )
            experiments.append(result)
            if (
                result.compliant
                and result.outcome["success"]
                and result.outcome["output_id"] is not None
            ):
                return _TransferProbe(tuple(experiments), primitive, protocols)
    raise RuntimeError("transfer probe did not find the transformation")


def _persist_transfer_probes(
    output: Path,
    seed: int,
    library: _TransferProbe,
    no_library: _TransferProbe,
    *,
    overwrite: bool = False,
) -> None:
    if not overwrite and any((output / name).exists() for name in _TRANSFER_FILES):
        raise FileExistsError("refusing to overwrite existing transfer evidence")
    evidence = {
        "seed": seed + 10_000,
        "scope": "learned executable program reuse versus scratch protocol search; not a fixed-skill-library superiority claim",
    }
    for name, probe in (("library", library), ("no-library", no_library)):
        path = output / f"{name}-transfer-events.jsonl"
        records = [
            record
            for experiment in probe.experiments
            for record in experiment.transitions
        ]
        path.write_text(
            "".join(
                json.dumps(
                    record.model_dump(mode="json"), sort_keys=True, ensure_ascii=False
                )
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        evidence[name] = {
            "experiment_count": len(probe.experiments),
            "protocol_candidates": [list(protocol) for protocol in probe.protocols],
            "symbol": None
            if probe.symbol is None
            else probe.symbol.model_dump(mode="json"),
            "experiment_ids": [
                experiment.experiment_id for experiment in probe.experiments
            ],
            "successful_input_id": probe.experiments[
                -1
            ].intervention.controlled_variables["input_id"],
            "output_id": probe.experiments[-1].outcome["output_id"],
        }
    (output / "transfer-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_mvp_benchmark(
    *,
    seed: int,
    intervention_budget: int,
    output: Path,
    overwrite: bool,
    planner: str = "information_gain",
    planner_checkpoint: str | None = None,
) -> dict[str, object]:
    if not overwrite and any(
        (output / name).exists() for name in (*_TRANSFER_FILES, "benchmark.json")
    ):
        raise FileExistsError("benchmark artifacts already exist")
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
    primitive = next(
        (
            symbol
            for symbol in full.symbols
            if symbol.status == "accepted"
            and symbol.type_signature == "(processor, input) -> output"
        ),
        None,
    )
    if primitive is None:
        raise RuntimeError(
            "library benchmark did not learn an executable transform primitive"
        )
    library_transfer = _run_transfer_probe(seed, primitive=primitive)
    scratch_transfer = _run_transfer_probe(seed, primitive=None)
    full_transfer, ablation_transfer = (
        len(library_transfer.experiments),
        len(scratch_transfer.experiments),
    )
    full.persist(output / "full", overwrite=overwrite)
    ablation.persist(output / "no-library", overwrite=overwrite)
    _persist_transfer_probes(
        output, seed, library_transfer, scratch_transfer, overwrite=overwrite
    )
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
