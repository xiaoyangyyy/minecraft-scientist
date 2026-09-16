"""Live server-backed piecewise equation discovery and revision."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minecraft_scientist.protocols import (
    ActionKind,
    ObjectObservation,
    Observation,
    PrimitiveAction,
    TheoryStatus,
    TransitionRecord,
    WorldEvent,
)
from minecraft_scientist.theory import TheoryInterpreter, TheoryRegistry

from .adapter import MinecraftConnectionConfig, MinecraftWorldAdapter
from .scenarios import FallEquationScenario
from .server import MinecraftServerProcess, create_server_process


def _fall_safe_height(seed: int, shifted: bool) -> int:
    standard = 2 + (seed % 3)
    return {2: 3, 3: 4, 4: 2}[standard] if shifted else standard


def _fall_slope(seed: int, shifted: bool, armor_index: int) -> float:
    slopes = (0.4, 0.7, 1.0)
    rotation = math.trunc(seed / 3) % len(slopes)
    return slopes[(armor_index + rotation + int(shifted)) % len(slopes)]


class LiveFallWorld:
    """Opaque agent view over admin-controlled deterministic damage interventions."""

    __slots__ = (
        "__adapter",
        "__channel",
        "__scenario",
        "__selected",
        "__armor_index",
        "__platform_height",
        "__counter",
    )

    def __init__(
        self,
        adapter: MinecraftWorldAdapter,
        channel: MinecraftServerProcess,
        scenario: FallEquationScenario,
    ) -> None:
        self.__adapter = adapter
        self.__channel = channel
        self.__scenario = scenario
        self.__selected: str | None = None
        self.__armor_index: dict[str, int] = {}
        self.__platform_height: dict[str, int] = {}
        self.__counter = 0
        self.begin_trial()

    def begin_trial(self) -> Observation:
        self.__scenario.apply(self.__channel)
        self.__adapter.step(PrimitiveAction(kind=ActionKind.WAIT, ticks=5))
        self.__selected = None
        observation = self.__adapter.observe()
        self.__bind_controller_ids(observation)
        return self.__decorate(observation)

    def observe(self) -> Observation:
        observation = self.__adapter.observe()
        self.__bind_controller_ids(observation)
        return self.__decorate(observation)

    def available_actions(self) -> tuple[PrimitiveAction, ...]:
        observation = self.observe()
        return tuple(
            PrimitiveAction(kind=ActionKind.SELECT, target_id=item.object_id)
            for item in observation.objects
            if item.object_id in self.__armor_index
        ) + tuple(
            PrimitiveAction(kind=ActionKind.INTERACT, target_id=item.object_id)
            for item in observation.objects
            if item.object_id in self.__platform_height
        )

    def step(self, action: PrimitiveAction) -> TransitionRecord:
        before = self.observe()
        self.__counter += 1
        if action.kind is ActionKind.SELECT and action.target_id in self.__armor_index:
            raw = self.__adapter.step(action)
            self.__selected = action.target_id
            return raw.model_copy(
                update={
                    "state_before": before,
                    "state_after": self.observe(),
                    "context": {"backend": "minecraft-fabric", "kind": "fall"},
                }
            )
        if (
            action.kind is ActionKind.INTERACT
            and action.target_id in self.__platform_height
            and self.__selected is not None
        ):
            height = self.__platform_height[action.target_id]
            armor = self.__armor_index[self.__selected]
            self.__channel.command(
                f"scientist fall {self.__scenario.username} {armor} {height}"
            )
            wait = self.__adapter.step(
                PrimitiveAction(kind=ActionKind.WAIT, ticks=2)
            )
            after = self.observe()
            damage = round(20.0 - float(after.agent_state["health"]), 3)
            event = WorldEvent(
                event_type="minecraft_fall_result",
                target_id=action.target_id,
                success=True,
                details={"height": height, "damage": damage},
            )
            return TransitionRecord(
                transition_id=(
                    f"minecraft-fall-{self.__scenario.variant}-{self.__counter:05d}"
                ),
                context={
                    "backend": "minecraft-fabric",
                    "selected_armor": self.__selected,
                },
                state_before=before,
                executed_action=action,
                intervention_compliance=wait.intervention_compliance,
                state_after=after,
                observed_delta={"damage": damage, "height": height},
                event=event,
                controller_failure=wait.controller_failure,
                provenance={
                    "backend": "minecraft-fabric",
                    "variant": self.__scenario.variant,
                    "seed": self.__scenario.seed,
                },
            )
        event = WorldEvent(
            event_type="control_failure",
            target_id=action.target_id,
            success=False,
            details={"reason": "invalid_live_fall_intervention"},
        )
        return TransitionRecord(
            transition_id=f"minecraft-fall-invalid-{self.__counter:05d}",
            context={"backend": "minecraft-fabric"},
            state_before=before,
            executed_action=action,
            intervention_compliance=False,
            state_after=before,
            observed_delta={},
            event=event,
            controller_failure="invalid_live_fall_intervention",
            provenance={"backend": "minecraft-fabric"},
        )

    def __bind_controller_ids(self, observation: Observation) -> None:
        armor_names = ("leather_boots", "chainmail_boots", "iron_boots")
        self.__armor_index = {
            self.__one("inventory", name): index
            for index, name in enumerate(armor_names)
        }
        by_position = {
            (
                int(item.observable_features["x"]),
                int(item.observable_features["y"]),
                int(item.observable_features["z"]),
            ): item.object_id
            for item in observation.objects
            if all(axis in item.observable_features for axis in ("x", "y", "z"))
        }
        self.__platform_height = {
            by_position[position]: height
            for height, position in self.__scenario.platform_positions().items()
        }

    def __decorate(self, observation: Observation) -> Observation:
        objects: list[ObjectObservation] = []
        for item in observation.objects:
            if item.object_id in self.__platform_height:
                features = dict(item.observable_features)
                features["height"] = self.__platform_height[item.object_id]
                objects.append(item.model_copy(update={"observable_features": features}))
            else:
                objects.append(item)
        return observation.model_copy(
            update={"objects": tuple(objects), "selected_id": self.__selected}
        )

    def __one(self, category: str, name: str) -> str:
        matches = self.__adapter._controller_ids_for(category, name)
        if len(matches) != 1:
            raise RuntimeError(f"missing privileged live fall binding: {category}/{name}")
        return matches[0]


class LiveFallEvaluator:
    __slots__ = ("__world", "__adapter", "__seed", "__shifted")

    def __init__(
        self,
        world: LiveFallWorld,
        adapter: MinecraftWorldAdapter,
        *,
        seed: int,
        shifted: bool,
    ) -> None:
        self.__world = world
        self.__adapter = adapter
        self.__seed = seed
        self.__shifted = shifted

    @property
    def armor_ids(self) -> tuple[str, ...]:
        observation = self.__world.observe()
        return tuple(
            sorted(
                item.object_id
                for item in observation.objects
                if ActionKind.SELECT in item.affordances
            )
        )

    @property
    def heights(self) -> tuple[int, ...]:
        return tuple(range(1, 11))

    def damage(self, armor_id: str, height: int) -> float:
        semantic_order = (
            self.__one("leather_boots"),
            self.__one("chainmail_boots"),
            self.__one("iron_boots"),
        )
        armor_index = semantic_order.index(armor_id)
        safe = _fall_safe_height(self.__seed, self.__shifted)
        slope = _fall_slope(self.__seed, self.__shifted, armor_index)
        return round(slope * max(0, height - safe), 3)

    def __one(self, name: str) -> str:
        matches = self.__adapter._controller_ids_for("inventory", name)
        if len(matches) != 1:
            raise RuntimeError(f"missing privileged live fall armor binding: {name}")
        return matches[0]


@dataclass(frozen=True, slots=True)
class LiveFallBundle:
    agent_view: LiveFallWorld
    evaluator: LiveFallEvaluator


def run_live_fall_discovery(
    *,
    server_directory: Path,
    output: Path,
    seed: int,
    intervention_budget: int = 40,
    username: str = "ScientistBot",
    overwrite: bool = False,
    server_mode: str = "fabric",
    planner: str = "information_gain",
    planner_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Discover, falsify, and replace a live Fabric-backed equation theory."""

    from minecraft_scientist.evaluation.full import (
        _behavioral_accuracy,
        _discover_fall,
        _public_inventory,
        _trial,
    )

    experiment_planner = None
    if planner == "learned":
        if planner_checkpoint is None:
            raise ValueError("planner_checkpoint is required for the learned planner")
        from minecraft_scientist.training.planner import NeuralExperimentPlanner

        experiment_planner = NeuralExperimentPlanner(planner_checkpoint)
    elif planner != "information_gain":
        raise ValueError(f"unknown planner: {planner}")

    report_path = output / "report.json"
    artifact_paths = (
        report_path,
        output / "events.jsonl",
        output / "theories.jsonl",
    )
    if not overwrite:
        existing = next((path for path in artifact_paths if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"live fall artifact already exists: {existing}")
    standard_scenario = FallEquationScenario(
        username=username,
        seed=seed,
        variant="standard",
    )
    transfer_scenario = FallEquationScenario(
        username=username,
        seed=seed,
        variant="standard",
        origin_x=standard_scenario.origin_x + 24,
    )
    allowed_positions = tuple(
        dict.fromkeys(
            (
                *standard_scenario.platform_positions().values(),
                *transfer_scenario.platform_positions().values(),
            )
        )
    )
    connection = MinecraftConnectionConfig(
        username=username,
        radius=12,
        # The bridge finds nearby solid blocks before the adapter applies the
        # explicit position allow-list.  Include the local floor in that raw
        # window so none of the ten more distant fixtures is truncated first.
        max_blocks=512,
        allowed_block_positions=allowed_positions,
    )
    with (
        create_server_process(server_directory, mode=server_mode) as server,
        MinecraftWorldAdapter(connection) as adapter,
    ):
        adapter.reset(seed)
        standard_world = LiveFallWorld(adapter, server, standard_scenario)
        standard = LiveFallBundle(
            agent_view=standard_world,
            evaluator=LiveFallEvaluator(
                standard_world,
                adapter,
                seed=seed,
                shifted=False,
            ),
        )
        discovery = _discover_fall(
            standard,
            budget=intervention_budget,
            planner=experiment_planner,
        )
        accuracy = _behavioral_accuracy(discovery.ir, standard)
        _, standard_platforms = _public_inventory(standard.agent_view.observe())
        standard_untested = [
            (armor_id, height)
            for armor_id in standard.evaluator.armor_ids
            for height in standard.evaluator.heights
            if (armor_id, height) not in discovery.tested
        ]
        standard_validation_pair = (
            min(standard_untested) if standard_untested else min(discovery.tested)
        )
        standard_validation_records, standard_validation_sample = _trial(
            standard,
            standard_validation_pair[0],
            standard_validation_pair[1],
            standard_platforms[standard_validation_pair[1]],
        )
        standard_prediction = float(
            TheoryInterpreter.predict(
                discovery.ir,
                armor_id=standard_validation_pair[0],
                height=standard_validation_pair[1],
            )["damage"]
        )
        standard_validation_passed = math.isclose(
            standard_prediction,
            float(standard_validation_sample.outcome["damage"]),
            abs_tol=1e-9,
        )
        if accuracy != 1.0 or not standard_validation_passed:
            raise RuntimeError("live fall equation failed standard validation")

        registry = TheoryRegistry()
        proposed = registry.propose(
            "live-fall-damage",
            discovery.ir,
            evidence_ids=tuple(sample.sample_id for sample in discovery.samples)
            + (standard_validation_sample.sample_id,),
            validation_score=1.0 - accuracy,
            confidence=accuracy,
            created_by="live-equation-version-space",
        )
        version_one = registry.accept_proposal(proposed.theory_id, proposed.version)

        transfer_world = LiveFallWorld(adapter, server, transfer_scenario)
        transfer = LiveFallBundle(
            agent_view=transfer_world,
            evaluator=LiveFallEvaluator(
                transfer_world,
                adapter,
                seed=seed,
                shifted=False,
            ),
        )
        _, transfer_platforms = _public_inventory(transfer.agent_view.observe())
        transfer_transitions: list[TransitionRecord] = []
        transfer_correct = 0
        transfer_pairs = tuple(
            zip(transfer.evaluator.armor_ids, (5, 7, 10), strict=True)
        )
        for armor_id, height in transfer_pairs:
            records, sample = _trial(
                transfer,
                armor_id,
                height,
                transfer_platforms[height],
            )
            transfer_transitions.extend(records)
            prediction = float(
                TheoryInterpreter.predict(
                    version_one.ir,
                    armor_id=armor_id,
                    height=height,
                )["damage"]
            )
            transfer_correct += math.isclose(
                prediction,
                float(sample.outcome["damage"]),
                abs_tol=1e-9,
            )
        transfer_accuracy = transfer_correct / len(transfer_pairs)
        if transfer_accuracy != 1.0:
            raise RuntimeError("live fall theory failed moved-arena transfer")

        shifted_scenario = FallEquationScenario(
            username=username,
            seed=seed,
            variant="shifted",
        )
        shifted_world = LiveFallWorld(adapter, server, shifted_scenario)
        shifted = LiveFallBundle(
            agent_view=shifted_world,
            evaluator=LiveFallEvaluator(
                shifted_world,
                adapter,
                seed=seed,
                shifted=True,
            ),
        )
        _, platforms = _public_inventory(shifted.agent_view.observe())
        crisis_transitions: list[TransitionRecord] = []
        counterexample: str | None = None
        crisis_delay = 0
        for armor_id in shifted.evaluator.armor_ids:
            for height in shifted.evaluator.heights:
                crisis_delay += 1
                records, sample = _trial(
                    shifted,
                    armor_id,
                    height,
                    platforms[height],
                )
                crisis_transitions.extend(records)
                predicted = float(
                    TheoryInterpreter.predict(
                        version_one.ir,
                        armor_id=armor_id,
                        height=height,
                    )["damage"]
                )
                if not math.isclose(
                    predicted,
                    float(sample.outcome["damage"]),
                    abs_tol=1e-9,
                ):
                    counterexample = sample.sample_id
                    break
            if counterexample is not None:
                break
        if counterexample is None:
            raise RuntimeError("live fall rule shift did not trigger a theory crisis")
        registry.transition(
            version_one.theory_id,
            version_one.version,
            TheoryStatus.CHALLENGED,
            counterexample_ids=(counterexample,),
        )
        revision = _discover_fall(
            shifted,
            budget=intervention_budget,
            planner=experiment_planner,
        )
        revised_accuracy = _behavioral_accuracy(revision.ir, shifted)
        _, shifted_platforms = _public_inventory(shifted.agent_view.observe())
        revision_untested = [
            (armor_id, height)
            for armor_id in shifted.evaluator.armor_ids
            for height in shifted.evaluator.heights
            if (armor_id, height) not in revision.tested
        ]
        revision_validation_pair = (
            min(revision_untested) if revision_untested else min(revision.tested)
        )
        revision_validation_records, revision_validation_sample = _trial(
            shifted,
            revision_validation_pair[0],
            revision_validation_pair[1],
            shifted_platforms[revision_validation_pair[1]],
        )
        revision_prediction = float(
            TheoryInterpreter.predict(
                revision.ir,
                armor_id=revision_validation_pair[0],
                height=revision_validation_pair[1],
            )["damage"]
        )
        revision_validation_passed = math.isclose(
            revision_prediction,
            float(revision_validation_sample.outcome["damage"]),
            abs_tol=1e-9,
        )
        if revised_accuracy != 1.0 or not revision_validation_passed:
            raise RuntimeError("live fall equation failed revision validation")
        proposed_revision = registry.propose(
            version_one.theory_id,
            revision.ir,
            parent_version=version_one.version,
            evidence_ids=tuple(sample.sample_id for sample in revision.samples)
            + (revision_validation_sample.sample_id,),
            validation_score=1.0 - revised_accuracy,
            confidence=revised_accuracy,
            created_by="live-theory-crisis-replacement",
        )
        registry.accept_proposal(
            proposed_revision.theory_id,
            proposed_revision.version,
        )
        registry.transition(
            version_one.theory_id,
            version_one.version,
            TheoryStatus.SUPERSEDED,
        )

    output.mkdir(parents=True, exist_ok=True)
    theories = registry.history("live-fall-damage")
    transitions = (
        *discovery.transitions,
        *standard_validation_records,
        *transfer_transitions,
        *crisis_transitions,
        *revision.transitions,
        *revision_validation_records,
    )
    (output / "events.jsonl").write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in transitions
        ),
        encoding="utf-8",
    )
    (output / "theories.jsonl").write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in theories
        ),
        encoding="utf-8",
    )
    payload: dict[str, Any] = {
        "seed": seed,
        "backend": "minecraft-fabric",
        "planner": planner,
        "planner_checkpoint": (
            str(planner_checkpoint) if planner_checkpoint is not None else None
        ),
        "standard_law_recovery": accuracy,
        "standard_experiments": discovery.trial_count,
        "standard_shadow_validation": standard_validation_passed,
        "description_length": TheoryInterpreter.description_length(discovery.ir),
        "world_transfer": {
            "arena_changed": True,
            "validation_experiments": len(transfer_pairs),
            "law_recovery": transfer_accuracy,
        },
        "change_detection_delay": crisis_delay,
        "revision_experiments": revision.trial_count + 1,
        "revision_shadow_validation": revision_validation_passed,
        "revised_law_recovery": revised_accuracy,
        "revised_description_length": TheoryInterpreter.description_length(
            revision.ir
        ),
        "theory_history": [
            {
                "version": theory.version,
                "parent_version": theory.parent_version,
                "status": theory.status.value,
            }
            for theory in theories
        ],
        "all_passed": (
            accuracy == 1.0
            and standard_validation_passed
            and transfer_accuracy == 1.0
            and revised_accuracy == 1.0
            and revision_validation_passed
        ),
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
