"""Expanded multi-form benchmark covering equation, transfer, and revision."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import networkx as nx

from minecraft_scientist.causal import CausalSample, LocalCausalEngine
from minecraft_scientist.experience import ExperienceDatabase
from minecraft_scientist.protocols import (
    ActionKind,
    Observation,
    PrimitiveAction,
    TheoryIR,
    TheoryStatus,
    TransitionRecord,
)
from minecraft_scientist.symbolic import (
    FallEquationVersionSpace,
    RepresentationEvidence,
    RepresentationSelector,
)
from minecraft_scientist.theory import (
    TheoryInterpreter,
    TheoryRegistry,
    numeric_theory_score,
)
from minecraft_scientist.world import FallWorldBundle, FallWorldFactory

from .benchmark import run_mvp_benchmark


@dataclass(frozen=True, slots=True)
class _FallDiscovery:
    ir: TheoryIR
    transitions: tuple[TransitionRecord, ...]
    samples: tuple[CausalSample, ...]
    tested: frozenset[tuple[str, int]]
    trial_count: int


class _FallPlanner(Protocol):
    def select_fall_experiment(
        self, space: FallEquationVersionSpace
    ) -> tuple[str, int] | None: ...


def _public_inventory(
    observation: Observation,
) -> tuple[tuple[str, ...], dict[int, str]]:
    armor_ids = tuple(
        sorted(
            item.object_id
            for item in observation.objects
            if ActionKind.SELECT in item.affordances
        )
    )
    platforms = {
        int(item.observable_features["height"]): item.object_id
        for item in observation.objects
        if ActionKind.INTERACT in item.affordances
        and "height" in item.observable_features
    }
    return armor_ids, platforms


def _trial(
    bundle: FallWorldBundle,
    armor_id: str,
    height: int,
    platform_id: str,
) -> tuple[tuple[TransitionRecord, TransitionRecord], CausalSample]:
    bundle.agent_view.begin_trial()
    selection = bundle.agent_view.step(
        PrimitiveAction(kind=ActionKind.SELECT, target_id=armor_id)
    )
    result = bundle.agent_view.step(
        PrimitiveAction(kind=ActionKind.INTERACT, target_id=platform_id)
    )
    if not selection.intervention_compliance or not result.intervention_compliance:
        raise RuntimeError("fall experiment controller failed")
    damage = float(result.event.details["damage"])
    return (
        (selection, result),
        CausalSample(
            sample_id=result.transition_id,
            context={"armor_id": armor_id, "height": height},
            intervention={"armor_id": armor_id, "height": height},
            outcome={"damage": damage},
            provenance={"backend": "structured-fall"},
        ),
    )


def _discover_fall(
    bundle: FallWorldBundle,
    *,
    budget: int,
    safe_height_candidates: tuple[int, ...] = (2, 3, 4),
    slope_candidates: tuple[float, ...] = (0.2, 0.4, 0.7, 1.0, 1.3),
    planner: _FallPlanner | None = None,
) -> _FallDiscovery:
    observation = bundle.agent_view.observe()
    armor_ids, platforms = _public_inventory(observation)
    space = FallEquationVersionSpace(
        armor_ids,
        tuple(sorted(platforms)),
        safe_height_candidates=safe_height_candidates,
        slope_candidates=slope_candidates,
    )
    transitions: list[TransitionRecord] = []
    samples: list[CausalSample] = []
    tested: set[tuple[str, int]] = set()
    while not space.resolved():
        if len(samples) >= budget:
            raise RuntimeError(f"fall discovery budget exhausted ({budget})")
        candidate = (
            planner.select_fall_experiment(space)
            if planner is not None
            else space.next_experiment()
        )
        if candidate is None:
            break
        armor_id, height = candidate
        records, sample = _trial(bundle, armor_id, height, platforms[height])
        transitions.extend(records)
        samples.append(sample)
        tested.add(candidate)
        space.observe(
            armor_id,
            height,
            float(sample.outcome["damage"]),
            sample.sample_id,
        )
    return _FallDiscovery(
        ir=space.to_ir(),
        transitions=tuple(transitions),
        samples=tuple(samples),
        tested=frozenset(tested),
        trial_count=len(samples),
    )


def _behavioral_accuracy(ir: TheoryIR, bundle: FallWorldBundle) -> float:
    correct = 0
    total = 0
    for armor_id in bundle.evaluator.armor_ids:
        for height in bundle.evaluator.heights:
            predicted = TheoryInterpreter.predict(ir, armor_id=armor_id, height=height)[
                "damage"
            ]
            observed = bundle.evaluator.damage(armor_id, height)
            correct += math.isclose(float(predicted), observed, abs_tol=1e-9)
            total += 1
    return correct / total


def _ensure_causal_contrasts(
    bundle: FallWorldBundle,
    discovery: _FallDiscovery,
) -> tuple[tuple[TransitionRecord, ...], tuple[CausalSample, ...]]:
    observation = bundle.agent_view.observe()
    armor_ids, platforms = _public_inventory(observation)
    required = {
        (armor_ids[0], min(platforms)),
        (armor_ids[0], max(platforms)),
        (armor_ids[1], max(platforms)),
    }
    transitions = list(discovery.transitions)
    samples = list(discovery.samples)
    tested = set(discovery.tested)
    for armor_id, height in sorted(required - tested):
        records, sample = _trial(bundle, armor_id, height, platforms[height])
        transitions.extend(records)
        samples.append(sample)
    return tuple(transitions), tuple(samples)


def _causal_payload(samples: tuple[CausalSample, ...]) -> dict[str, Any]:
    engine = LocalCausalEngine(samples)
    armor_ids = sorted({str(sample.context["armor_id"]) for sample in samples})
    heights = sorted({int(sample.context["height"]) for sample in samples})
    height_effect = engine.controlled_effect(
        "height", "damage", fixed_context={"armor_id": armor_ids[0]}
    )
    armor_effect = engine.controlled_effect(
        "armor_id", "damage", fixed_context={"height": heights[-1]}
    )
    graph = nx.DiGraph()
    graph.add_node("height", kind="intervention")
    graph.add_node("armor_id", kind="intervention")
    graph.add_node("damage", kind="outcome")
    graph.add_edge(
        "height",
        "damage",
        effect=height_effect.absolute_effect,
        confidence=height_effect.confidence,
    )
    graph.add_edge(
        "armor_id",
        "damage",
        effect=armor_effect.absolute_effect,
        confidence=armor_effect.confidence,
    )
    return {
        "height_effect": height_effect.model_dump(mode="json"),
        "armor_effect": armor_effect.model_dump(mode="json"),
        "graph": LocalCausalEngine.graph_payload(graph),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_full_benchmark(
    *,
    seed: int,
    intervention_budget: int,
    output: Path,
    overwrite: bool,
    planner: str = "information_gain",
    planner_checkpoint: str | None = None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "full-benchmark.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"full benchmark report already exists: {report_path}")
    neural_planner: _FallPlanner | None = None
    if planner == "learned":
        if planner_checkpoint is None:
            raise ValueError("planner_checkpoint is required for the learned planner")
        from minecraft_scientist.training.planner import NeuralExperimentPlanner

        neural_planner = NeuralExperimentPlanner(Path(planner_checkpoint))
    elif planner != "information_gain":
        raise ValueError(f"unknown planner: {planner}")
    structured = run_mvp_benchmark(
        seed=seed,
        intervention_budget=intervention_budget,
        output=output / "structured",
        overwrite=overwrite,
        planner=planner,
        planner_checkpoint=planner_checkpoint,
    )

    world_a = FallWorldFactory.create(seed, law_seed=seed)
    discovery_a = _discover_fall(
        world_a, budget=intervention_budget, planner=neural_planner
    )
    transitions_a, samples_a = _ensure_causal_contrasts(world_a, discovery_a)
    untested = [
        (armor_id, height)
        for armor_id in world_a.evaluator.armor_ids
        for height in world_a.evaluator.heights
        if (armor_id, height) not in discovery_a.tested
    ]
    validation_pair = min(untested) if untested else min(discovery_a.tested)
    _, platforms_a = _public_inventory(world_a.agent_view.observe())
    validation_records, validation_sample = _trial(
        world_a,
        validation_pair[0],
        validation_pair[1],
        platforms_a[validation_pair[1]],
    )
    predicted_validation = float(
        TheoryInterpreter.predict(
            discovery_a.ir,
            armor_id=validation_pair[0],
            height=validation_pair[1],
        )["damage"]
    )
    observed_validation = float(validation_sample.outcome["damage"])
    validation_passed = math.isclose(
        predicted_validation, observed_validation, abs_tol=1e-9
    )

    registry = TheoryRegistry()
    proposed = registry.propose(
        "fall-damage",
        discovery_a.ir,
        evidence_ids=tuple(sample.sample_id for sample in discovery_a.samples)
        + (validation_sample.sample_id,),
        validation_score=numeric_theory_score(
            discovery_a.ir,
            [predicted_validation],
            [observed_validation],
        ),
        confidence=1.0 if validation_passed else 0.0,
        created_by="equation-version-space",
    )
    if not validation_passed:
        registry.transition("fall-damage", proposed.version, TheoryStatus.REJECTED)
        raise RuntimeError("fall equation failed shadow validation")
    version_a = registry.accept_proposal("fall-damage", proposed.version)

    world_b_scratch = FallWorldFactory.create(seed + 1, law_seed=seed)
    scratch_discovery = _discover_fall(
        world_b_scratch, budget=intervention_budget, planner=neural_planner
    )
    world_b_library = FallWorldFactory.create(seed + 1, law_seed=seed)
    library_discovery = _discover_fall(
        world_b_library,
        budget=intervention_budget,
        safe_height_candidates=(int(discovery_a.ir.parameters["safe_height"]),),
        slope_candidates=tuple(
            sorted(
                float(value)
                for value in discovery_a.ir.parameters["armor_slopes"].values()
            )
        ),
        planner=neural_planner,
    )
    scratch_transfer = scratch_discovery.trial_count
    library_transfer = library_discovery.trial_count
    transfer_accuracy = _behavioral_accuracy(library_discovery.ir, world_b_library)

    world_c = FallWorldFactory.create(seed, law_seed=seed, rule_shift=True)
    _, platforms_c = _public_inventory(world_c.agent_view.observe())
    counterexample_id: str | None = None
    crisis_delay = 0
    for armor_id in world_c.evaluator.armor_ids:
        for height in world_c.evaluator.heights:
            crisis_delay += 1
            _, sample = _trial(world_c, armor_id, height, platforms_c[height])
            prediction = float(
                TheoryInterpreter.predict(
                    version_a.ir, armor_id=armor_id, height=height
                )["damage"]
            )
            if not math.isclose(
                prediction, float(sample.outcome["damage"]), abs_tol=1e-9
            ):
                counterexample_id = sample.sample_id
                break
        if counterexample_id is not None:
            break
    if counterexample_id is None:
        raise RuntimeError("changed fall law did not trigger a theory crisis")
    registry.transition(
        "fall-damage",
        version_a.version,
        TheoryStatus.CHALLENGED,
        counterexample_ids=(counterexample_id,),
    )
    revision = _discover_fall(
        world_c, budget=intervention_budget, planner=neural_planner
    )
    revision_untested = [
        (armor_id, height)
        for armor_id in world_c.evaluator.armor_ids
        for height in world_c.evaluator.heights
        if (armor_id, height) not in revision.tested
    ]
    revision_validation_pair = (
        min(revision_untested) if revision_untested else min(revision.tested)
    )
    _revision_validation_records, revision_validation_sample = _trial(
        world_c,
        revision_validation_pair[0],
        revision_validation_pair[1],
        platforms_c[revision_validation_pair[1]],
    )
    revision_prediction = float(
        TheoryInterpreter.predict(
            revision.ir,
            armor_id=revision_validation_pair[0],
            height=revision_validation_pair[1],
        )["damage"]
    )
    revision_observation = float(revision_validation_sample.outcome["damage"])
    if not math.isclose(revision_prediction, revision_observation, abs_tol=1e-9):
        raise RuntimeError("revised fall equation failed shadow validation")
    proposed_revision = registry.propose(
        "fall-damage",
        revision.ir,
        parent_version=version_a.version,
        evidence_ids=tuple(sample.sample_id for sample in revision.samples)
        + (revision_validation_sample.sample_id,),
        validation_score=numeric_theory_score(
            revision.ir, [revision_prediction], [revision_observation]
        ),
        confidence=1.0,
        created_by="theory-crisis-equation-replacement",
    )
    registry.accept_proposal("fall-damage", proposed_revision.version)
    registry.transition("fall-damage", version_a.version, TheoryStatus.SUPERSEDED)

    database_path = output / "experience.sqlite3"
    if overwrite:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{database_path}{suffix}")
            if candidate.exists():
                candidate.unlink()
    with ExperienceDatabase(database_path) as database:
        for transition in (*transitions_a, *validation_records):
            database.append("fall_damage", transition)
        stored_transitions = database.count("fall_damage")

    representation = RepresentationSelector.select(
        RepresentationEvidence(has_continuous_inputs=True)
    )
    causal = _causal_payload(samples_a)
    history = registry.history("fall-damage")
    structured_report = structured["full"]
    fall_accuracy = _behavioral_accuracy(discovery_a.ir, world_a)
    revised_accuracy = _behavioral_accuracy(revision.ir, world_c)
    law_recovery = (
        3.0 * float(structured_report["hidden_intervention_accuracy"]) + fall_accuracy
    ) / 4.0
    total_experiments = int(structured_report["experiment_count"]) + (
        stored_transitions // 2
    )
    task_success = (float(structured_report["task_success_rate"]) + fall_accuracy) / 2.0
    payload: dict[str, object] = {
        "suite": "full",
        "seed": seed,
        "planner": planner,
        "structured": structured_report,
        "continuous_law": {
            "representation": representation.value,
            "law_recovery": fall_accuracy,
            "active_discovery_experiments": discovery_a.trial_count,
            "validation_passed": validation_passed,
            "description_length": TheoryInterpreter.description_length(discovery_a.ir),
            "experience_transitions": stored_transitions,
        },
        "world_transfer": {
            "map_changed": True,
            "law_seed_preserved": True,
            "scratch_experiments": scratch_transfer,
            "library_experiments": library_transfer,
            "experiment_savings": scratch_transfer - library_transfer,
            "law_recovery": transfer_accuracy,
        },
        "theory_revision": {
            "change_detection_delay": crisis_delay,
            "revision_experiments": revision.trial_count + 1,
            "shadow_validation_passed": True,
            "new_law_recovery": revised_accuracy,
            "history": [
                {
                    "version": theory.version,
                    "parent_version": theory.parent_version,
                    "status": theory.status.value,
                    "counterexamples": list(theory.counterexample_ids),
                }
                for theory in history
            ],
        },
        "causal_discovery": causal,
        "metrics": {
            "law_recovery": law_recovery,
            "causal_intervention_accuracy": law_recovery,
            "description_length": int(structured_report["description_length"])
            + TheoryInterpreter.description_length(discovery_a.ir),
            "world_transfer": transfer_accuracy,
            "theory_revision_detection_experiments": crisis_delay,
            "theory_revision_total_experiments": crisis_delay
            + revision.trial_count
            + 1,
            "experiment_count": total_experiments,
            "experiment_efficiency": law_recovery / max(total_experiments, 1),
            "task_success": task_success,
        },
    }
    payload["all_passed"] = bool(
        structured["full"]["hidden_intervention_accuracy"] == 1.0
        and payload["continuous_law"]["law_recovery"] == 1.0
        and payload["continuous_law"]["validation_passed"]
        and payload["world_transfer"]["experiment_savings"] > 0
        and payload["world_transfer"]["law_recovery"] == 1.0
        and payload["theory_revision"]["new_law_recovery"] == 1.0
        and len(payload["causal_discovery"]["graph"]["edges"]) == 2
    )
    _write_json(
        output / "fall-theories.json",
        [theory.model_dump(mode="json") for theory in history],
    )
    _write_json(output / "causal-graph.json", causal)
    _write_json(report_path, payload)
    return payload
