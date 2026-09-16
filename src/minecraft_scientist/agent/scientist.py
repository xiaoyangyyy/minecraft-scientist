"""The deterministic fast/slow discovery loops for the MVP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from minecraft_scientist.discovery import (
    AbstractionLearner,
    AbstractionValidation,
    HeuristicHypothesisGenerator,
    MachineVersionSpace,
    OpenAIHypothesisGenerator,
    ToolVersionSpace,
    TransformationLearner,
)
from minecraft_scientist.evaluation import evaluate_run
from minecraft_scientist.planning import ExperimentExecutor
from minecraft_scientist.protocols import (
    EvaluationReport,
    ExperimentResult,
    RunConfig,
    SymbolPrimitive,
    TheoryIR,
    TheoryStatus,
    TheoryVersion,
)
from minecraft_scientist.storage import RunStore
from minecraft_scientist.theory import (
    TheoryInterpreter,
    TheoryRegistry,
    TheoryValidator,
    theory_score,
)
from minecraft_scientist.world import WorldBundle, WorldFactory


@dataclass(frozen=True, slots=True)
class ScientistRun:
    config: RunConfig
    experiments: tuple[ExperimentResult, ...]
    theories: tuple[TheoryVersion, ...]
    symbols: tuple[SymbolPrimitive, ...]
    report: EvaluationReport

    def persist(self, output: Path, *, overwrite: bool = False) -> None:
        store = RunStore(output, overwrite=overwrite)
        store.write_config(self.config)
        store.write_experiments(self.experiments)
        store.write_theories(self.theories)
        store.write_symbols(self.symbols)
        store.write_report(self.report)


class ScientistAgent:
    def __init__(self, config: RunConfig, bundle: WorldBundle | None = None) -> None:
        self.config = config
        self.bundle = bundle or WorldFactory.create(config.seed, config.change_episode)
        self.registry = TheoryRegistry()
        self.executor = ExperimentExecutor(self.bundle.agent_view)
        self.experiments: list[ExperimentResult] = []
        if config.hypothesis_generator == "heuristic":
            self.generator = HeuristicHypothesisGenerator()
        elif config.hypothesis_generator == "openai":
            self.generator = OpenAIHypothesisGenerator()
        else:
            raise ValueError(
                f"unknown hypothesis generator: {config.hypothesis_generator}"
            )
        self._classifier = HeuristicHypothesisGenerator()
        self._neural_planner = None
        if config.planner == "learned":
            if config.planner_checkpoint is None:
                raise ValueError(
                    "planner_checkpoint is required for the learned planner"
                )
            try:
                from minecraft_scientist.training.planner import NeuralExperimentPlanner
            except ImportError as exc:
                raise RuntimeError(
                    "Install training dependencies with: uv sync --extra train"
                ) from exc
            self._neural_planner = NeuralExperimentPlanner(
                Path(config.planner_checkpoint)
            )
        elif config.planner != "information_gain":
            raise ValueError(f"unknown planner: {config.planner}")

    def run(self) -> ScientistRun:
        observation = self.bundle.controller.begin_episode(0)
        self.generator.propose(observation)
        groups = self._classifier.classify(observation)

        tool_space = self._discover_tools(
            tuple(groups["tools"]), tuple(groups["targets"])
        )
        self._register_tool_theory(tool_space, parent_version=None)

        machine_space = self._discover_machine(
            str(groups["machine"]), tuple(groups["resources"])
        )
        machine_ir = machine_space.to_ir(str(groups["machine"]))
        machine_validation = self._validate_machine(str(groups["machine"]), machine_ir)
        self._register_exact_theory(
            "machine-protocol",
            machine_ir,
            tuple(row[3] for row in machine_space.observations)
            + (machine_validation.experiment_id,),
        )

        transform = self._discover_transformations(
            tuple(groups["processors"]), tuple(groups["resources"])
        )
        transform_ir = transform.to_ir()
        transform_validation = self._validate_transformations(transform, transform_ir)
        self._register_exact_theory(
            "transformation-chain",
            transform_ir,
            tuple(row[4] for row in transform.observations)
            + tuple(result.experiment_id for result in transform_validation),
        )

        delay: int | None = None
        revision_experiments = 0
        if self.config.change_episode is not None:
            delay, revision_experiments = self._revise_after_change(
                tuple(groups["tools"]), tuple(groups["targets"])
            )

        accepted_for_library = tuple(
            theory
            for theory in self.registry.all_versions()
            if theory.status is TheoryStatus.ACCEPTED
        )
        library_validation = (
            self._collect_library_validation(transform_ir)
            if self.config.library_learning
            else None
        )
        symbols = AbstractionLearner().discover(
            accepted_for_library,
            enabled=self.config.library_learning,
            validation=library_validation,
        )

        latest = {
            "tool-capability": self.registry.latest("tool-capability"),
            "machine-protocol": self.registry.latest("machine-protocol"),
            "transformation-chain": self.registry.latest("transformation-chain"),
        }
        report = evaluate_run(
            seed=self.config.seed,
            evaluator=self.bundle.evaluator,
            latest_theories=latest,
            symbols=symbols,
            experiment_count=len(self.experiments),
            change_episode=self.config.change_episode,
            change_detection_delay=delay,
            revision_experiments=revision_experiments,
        )
        return ScientistRun(
            config=self.config,
            experiments=tuple(self.experiments),
            theories=self.registry.all_versions(),
            symbols=symbols,
            report=report,
        )

    def _discover_tools(
        self, tools: tuple[str, ...], targets: tuple[str, ...]
    ) -> ToolVersionSpace:
        space = ToolVersionSpace(tools, targets)
        while not space.resolved():
            pair = self._next_tool_experiment(space)
            if pair is None:
                break
            self._require_budget()
            result = self.executor.run_tool_test(
                *pair,
                predicted_outcomes=space.predictive_outcomes(*pair),
            )
            self.experiments.append(result)
            if not result.compliant:
                space.record_control_failure(*pair)
                continue
            space.observe(*pair, bool(result.outcome["success"]), result.experiment_id)
        return space

    def _next_tool_experiment(self, space: ToolVersionSpace) -> tuple[str, str] | None:
        if self._neural_planner is not None:
            return self._neural_planner.select_tool_experiment(space)
        return space.next_experiment()

    def _discover_machine(
        self, machine_id: str, resources: tuple[str, ...]
    ) -> MachineVersionSpace:
        space = MachineVersionSpace(resources)
        while not space.resolved():
            candidate = (
                self._neural_planner.select_machine_experiment(space)
                if self._neural_planner is not None
                else space.next_experiment()
            )
            if candidate is None:
                break
            self._require_budget()
            item, protocol = candidate
            result = self.executor.run_protocol_test(
                machine_id,
                item,
                protocol,
                predicted_outcomes=space.predictive_outcomes(item, protocol),
            )
            self.experiments.append(result)
            if not result.compliant:
                space.record_control_failure(item, protocol)
                continue
            space.observe(
                item,
                protocol,
                bool(result.outcome["success"]),
                result.experiment_id,
                result.outcome.get("output_id"),
            )
        if space.resolved() and not any(row[2] for row in space.observations):
            self._require_budget()
            best = space.candidates[0]
            result = self.executor.run_protocol_test(
                machine_id,
                best.input_id,
                best.protocol,
                predicted_outcomes=space.predictive_outcomes(
                    best.input_id, best.protocol
                ),
            )
            self.experiments.append(result)
            if not result.compliant:
                raise RuntimeError("resolved machine confirmation was not compliant")
            space.observe(
                best.input_id,
                best.protocol,
                bool(result.outcome["success"]),
                result.experiment_id,
                result.outcome.get("output_id"),
            )
        return space

    def _discover_transformations(
        self, processors: tuple[str, ...], resources: tuple[str, ...]
    ) -> TransformationLearner:
        learner = TransformationLearner(processors, resources)
        protocol = ("place", "activate", "wait:1", "interact")
        while not learner.resolved():
            candidate = (
                self._neural_planner.select_transformation_experiment(learner)
                if self._neural_planner is not None
                else learner.next_experiment()
            )
            if candidate is None:
                break
            self._require_budget()
            processor, item = candidate
            result = self.executor.run_protocol_test(
                processor,
                item,
                protocol,
                mechanism="transformation_chain",
                predicted_outcomes=learner.predictive_outcomes(processor, item),
            )
            self.experiments.append(result)
            if not result.compliant:
                learner.record_control_failure(processor, item)
                continue
            learner.observe(
                processor,
                item,
                bool(result.outcome["success"]),
                result.outcome.get("output_id"),
                result.experiment_id,
            )
        if not learner.resolved():
            raise RuntimeError(
                "transformation chain was not resolved within the budget"
            )
        return learner

    def _register_tool_theory(
        self, space: ToolVersionSpace, parent_version: int | None
    ) -> TheoryVersion:
        ir = space.to_ir()
        tested = {(tool, target) for tool, target, _, _ in space.observations}
        validation_pair = next(
            (
                (tool, target)
                for tool in space.tools
                for target in space.targets
                if (tool, target) not in tested
            ),
            (space.tools[0], space.targets[0]),
        )
        self._require_budget()
        prediction = TheoryInterpreter.predict(
            ir,
            tool_id=validation_pair[0],
            target_id=validation_pair[1],
        )
        validation = self.executor.run_tool_test(
            *validation_pair,
            predicted_outcomes={"success": bool(prediction["success"])},
        )
        self.experiments.append(validation)
        if not validation.compliant:
            raise RuntimeError("tool theory validation intervention was not compliant")
        rows = [
            (
                validation_pair[0],
                validation_pair[1],
                bool(validation.outcome["success"]),
            )
        ]
        score, confidence = TheoryValidator.validate_tool_theory(ir, rows)
        theory = self.registry.propose(
            "tool-capability",
            ir,
            evidence_ids=tuple(row[3] for row in space.observations)
            + (validation.experiment_id,),
            validation_score=score,
            confidence=confidence
            if space.resolved()
            else confidence / len(space.candidates),
            parent_version=parent_version,
        )
        return self.registry.accept_proposal(theory.theory_id, theory.version)

    def _register_exact_theory(
        self,
        theory_id: str,
        ir: TheoryIR,
        evidence_ids: tuple[str, ...],
    ) -> TheoryVersion:
        score = theory_score(ir, [True], [True])
        theory = self.registry.propose(
            theory_id,
            ir,
            evidence_ids=evidence_ids,
            validation_score=score,
            confidence=1.0,
        )
        return self.registry.accept_proposal(theory_id, theory.version)

    def _validate_machine(self, machine_id: str, ir: TheoryIR) -> ExperimentResult:
        self._require_budget()
        protocol = tuple(str(token) for token in ir.parameters["protocol"])
        waits = [index for index, token in enumerate(protocol) if token.startswith("wait:")]
        if len(waits) == 1:
            index = waits[0]
            duration = int(protocol[index].split(":", 1)[1]) + 2
            heldout_protocol = (
                *protocol[:index],
                f"wait:{duration}",
                *protocol[index + 1 :],
            )
        else:
            heldout_protocol = (*protocol[:-1], "wait:3", protocol[-1])
        prediction = TheoryInterpreter.predict(
            ir,
            input_id=str(ir.parameters["input_id"]),
            protocol=heldout_protocol,
        )
        result = self.executor.run_protocol_test(
            machine_id,
            str(ir.parameters["input_id"]),
            heldout_protocol,
            predicted_outcomes=prediction,
        )
        self.experiments.append(result)
        if (
            not result.compliant
            or bool(result.outcome["success"]) is not bool(prediction["success"])
            or (
                bool(prediction["success"])
                and result.outcome["output_id"] != ir.effects["output_id"]
            )
        ):
            raise RuntimeError("machine theory failed held-out protocol validation")
        return result

    def _validate_transformations(
        self, learner: TransformationLearner, ir: TheoryIR
    ) -> tuple[ExperimentResult, ...]:
        protocol = ("place", "activate", "wait:1", "interact")
        results: list[ExperimentResult] = []
        tested = {
            (processor, item)
            for processor, item, _, _, _ in learner.observations
        }
        for processor, pair in sorted(ir.parameters["mapping"].items()):
            self._require_budget()
            result = self.executor.run_protocol_test(
                processor,
                pair[0],
                protocol,
                mechanism="transformation_chain",
                predicted_outcomes={"success": True, "output_id": pair[1]},
            )
            self.experiments.append(result)
            if (
                not result.compliant
                or not result.outcome["success"]
                or result.outcome["output_id"] != pair[1]
            ):
                raise RuntimeError(
                    "transformation theory failed independent validation"
                )
            results.append(result)
            negative_input = next(
                (
                    item
                    for item in learner.inputs
                    if item != pair[0] and (processor, item) not in tested
                ),
                None,
            )
            if negative_input is not None:
                self._require_budget()
                negative = self.executor.run_protocol_test(
                    processor,
                    negative_input,
                    protocol,
                    mechanism="transformation_chain",
                    predicted_outcomes={"success": False, "output_id": None},
                )
                self.experiments.append(negative)
                if (
                    not negative.compliant
                    or negative.outcome["success"]
                    or negative.outcome["output_id"] is not None
                ):
                    raise RuntimeError(
                        "transformation theory failed held-out negative validation"
                    )
                results.append(negative)
        return tuple(results)

    def _revise_after_change(
        self, tools: tuple[str, ...], targets: tuple[str, ...]
    ) -> tuple[int, int]:
        assert self.config.change_episode is not None
        self.bundle.controller.begin_episode(self.config.change_episode)
        old = self.registry.latest("tool-capability")
        delay = 0
        counterexample: ExperimentResult | None = None
        for tool in sorted(tools):
            for target in sorted(targets):
                self._require_budget()
                prediction = TheoryInterpreter.predict(
                    old.ir, tool_id=tool, target_id=target
                )
                result = self.executor.run_tool_test(
                    tool,
                    target,
                    predicted_outcomes={
                        "success": bool(prediction["success"])
                    },
                )
                self.experiments.append(result)
                delay += 1
                if not result.compliant:
                    continue
                predicted = bool(prediction["success"])
                if predicted is not bool(result.outcome["success"]):
                    counterexample = result
                    break
            if counterexample is not None:
                break
        if counterexample is None:
            return delay, 0

        self.registry.transition(
            old.theory_id,
            old.version,
            TheoryStatus.CHALLENGED,
            counterexample_ids=(counterexample.experiment_id,),
        )
        before = len(self.experiments)
        new_space = self._discover_tools(tools, targets)
        self._register_tool_theory(new_space, parent_version=old.version)
        self.registry.transition(
            old.theory_id,
            old.version,
            TheoryStatus.SUPERSEDED,
        )
        return delay, len(self.experiments) - before

    def _collect_library_validation(
        self, transform_ir: TheoryIR
    ) -> AbstractionValidation:
        validation_seed = self.config.seed + 1_000_003
        world_id = f"heldout-seed-{validation_seed}"
        bundle = WorldFactory.create(validation_seed, change_episode=None)
        observation = bundle.controller.begin_episode(0)
        groups = self._classifier.classify(observation)
        executor = ExperimentExecutor(bundle.agent_view)

        tool_space = ToolVersionSpace(
            tuple(groups["tools"]), tuple(groups["targets"])
        )
        tool_evidence: list[str] = []
        while not tool_space.resolved():
            pair = tool_space.next_experiment()
            if pair is None:
                break
            self._require_budget()
            result = self._record_library_experiment(
                executor.run_tool_test(
                    *pair,
                    predicted_outcomes=tool_space.predictive_outcomes(*pair),
                ),
                world_id,
            )
            if not result.compliant:
                tool_space.record_control_failure(*pair)
                continue
            tool_evidence.append(result.experiment_id)
            tool_space.observe(
                *pair,
                bool(result.outcome["success"]),
                result.experiment_id,
            )

        semantic_pattern = tuple(
            str(token)
            for token in transform_ir.parameters.get("action_pattern", ())
        )
        protocol = tuple(
            "place"
            if token == "place_input"
            else "wait:1"
            if token == "wait"
            else token
            for token in semantic_pattern
        )
        processor = tuple(groups["processors"])[0]
        transform_evidence: list[str] = []
        for item in tuple(groups["resources"]):
            self._require_budget()
            result = self._record_library_experiment(
                executor.run_protocol_test(
                    processor,
                    item,
                    protocol,
                    mechanism="library_transfer_validation",
                    predicted_outcomes={
                        "candidate_primitive_success": True,
                        "action_pattern": "|".join(semantic_pattern),
                    },
                ),
                world_id,
            )
            if result.compliant and result.outcome["success"]:
                transform_evidence.append(result.experiment_id)
                break
        if not transform_evidence:
            raise RuntimeError("candidate transform primitive failed cross-world validation")
        return AbstractionValidation(
            world_id=world_id,
            tool_ir=tool_space.to_ir(),
            tool_evidence_ids=tuple(tool_evidence),
            transform_evidence_ids=tuple(transform_evidence),
        )

    def _record_library_experiment(
        self, result: ExperimentResult, world_id: str
    ) -> ExperimentResult:
        prefix = f"{world_id}:"
        intervention = result.intervention.model_copy(
            update={
                "intervention_id": prefix + result.intervention.intervention_id
            }
        )
        transitions = tuple(
            transition.model_copy(
                update={
                    "transition_id": prefix + transition.transition_id,
                    "intended_intervention": intervention,
                    "provenance": {
                        **transition.provenance,
                        "validation_world": world_id,
                    },
                }
            )
            for transition in result.transitions
        )
        namespaced = result.model_copy(
            update={
                "experiment_id": prefix + result.experiment_id,
                "intervention": intervention,
                "transitions": transitions,
            }
        )
        self.experiments.append(namespaced)
        return namespaced

    def _require_budget(self) -> None:
        if len(self.experiments) >= self.config.intervention_budget:
            raise RuntimeError(
                f"intervention budget exhausted ({self.config.intervention_budget})"
            )
