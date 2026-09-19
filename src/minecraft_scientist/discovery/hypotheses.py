"""Finite, inspectable hypothesis spaces used by the offline MVP."""

from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import networkx as nx

from minecraft_scientist.planning.selection import select_best_experiment
from minecraft_scientist.protocols import (
    Hypothesis,
    Observation,
    TheoryIR,
    TheoryRepresentation,
)


@dataclass(frozen=True, slots=True)
class ToolHypothesis:
    tool_ranks: dict[str, int]
    target_requirements: dict[str, int]

    def predicts(self, tool_id: str, target_id: str) -> bool:
        return self.tool_ranks[tool_id] >= self.target_requirements[target_id]


class ToolVersionSpace:
    def __init__(self, tools: tuple[str, ...], targets: tuple[str, ...], *,
                 include_unreachable: bool = False) -> None:
        self.tools = tuple(sorted(tools))
        self.targets = tuple(sorted(targets))
        self._requirement_count = len(self.tools) + int(include_unreachable)
        candidates: list[ToolHypothesis] = []
        for order in itertools.permutations(self.tools):
            ranks = {tool: rank for rank, tool in enumerate(order)}
            for requirements in itertools.product(
                range(self._requirement_count), repeat=len(self.targets)
            ):
                candidates.append(
                    ToolHypothesis(
                        tool_ranks=ranks,
                        target_requirements=dict(
                            zip(self.targets, requirements, strict=True)
                        ),
                    )
                )
        self.candidates = candidates
        self.observations: list[tuple[str, str, bool, str]] = []
        self.control_failures: dict[tuple[str, str], int] = {}

    def record_control_failure(self, tool_id: str, target_id: str) -> None:
        pair = (tool_id, target_id)
        self.control_failures[pair] = self.control_failures.get(pair, 0) + 1

    def observe(
        self, tool_id: str, target_id: str, success: bool, evidence_id: str
    ) -> None:
        self.observations.append((tool_id, target_id, success, evidence_id))
        self.candidates = [
            candidate
            for candidate in self.candidates
            if candidate.predicts(tool_id, target_id) is success
        ]
        if not self.candidates:
            raise RuntimeError("tool theory version space collapsed")

    def predictive_outcomes(self, tool_id: str, target_id: str) -> dict[str, float]:
        probability = sum(
            candidate.predicts(tool_id, target_id) for candidate in self.candidates
        ) / len(self.candidates)
        return {"success": probability, "failure": 1.0 - probability}

    def next_experiment(self) -> tuple[str, str] | None:
        pairs, features = self.experiment_features()
        return select_best_experiment(pairs, features)

    def experiment_features(
        self,
    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[float, ...], ...]]:
        """Return permutation-friendly features for every possible intervention."""

        tested = {(tool, target) for tool, target, _, _ in self.observations}
        pairs: list[tuple[str, str]] = []
        features: list[tuple[float, ...]] = []
        initial_count = max(
            1,
            len(tuple(itertools.permutations(self.tools)))
            * (self._requirement_count ** len(self.targets)),
        )
        posterior_fraction = len(self.candidates) / initial_count
        for tool in self.tools:
            for target in self.targets:
                true_count = sum(c.predicts(tool, target) for c in self.candidates)
                probability = true_count / len(self.candidates)
                if probability in (0.0, 1.0):
                    entropy = 0.0
                else:
                    import numpy as np

                    entropy = float(
                        -probability * np.log2(probability)
                        - (1.0 - probability) * np.log2(1.0 - probability)
                    )
                pairs.append((tool, target))
                features.append(
                    (
                        probability,
                        entropy,
                        float((tool, target) in tested),
                        posterior_fraction,
                        1.0,
                        self.control_failures.get((tool, target), 0)
                        / (self.control_failures.get((tool, target), 0) + 1),
                    )
                )
        return tuple(pairs), tuple(features)

    def resolved(self) -> bool:
        if len(self.candidates) == 1:
            return True
        _, features = self.experiment_features()
        return all(
            tested == 1.0 or entropy == 0.0
            for _, entropy, tested, *_ in features
        )

    def best(self) -> ToolHypothesis:
        return min(
            self.candidates,
            key=lambda c: json.dumps(
                {"ranks": c.tool_ranks, "requirements": c.target_requirements},
                sort_keys=True,
            ),
        )

    def to_ir(self) -> TheoryIR:
        hypothesis = self.best()
        return TheoryIR(
            mechanism="tool_capability",
            representation=TheoryRepresentation.RULE,
            scope={"tools": list(self.tools), "targets": list(self.targets)},
            latent_variables=("capability_level", "required_level"),
            preconditions=("selected(tool)", "interactable(target)"),
            effects={"success": "capability_level(tool) >= required_level(target)"},
            parameters={
                "tool_ranks": hypothesis.tool_ranks,
                "target_requirements": hypothesis.target_requirements,
            },
        )


PROTOCOL_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("place", "activate", "wait:1", "interact"),
    ("place", "activate", "wait:2", "interact"),
    ("activate", "place", "wait:1", "interact"),
    ("place", "wait:1", "activate", "interact"),
    ("place", "activate", "interact"),
)


@dataclass(frozen=True, slots=True)
class MachineHypothesis:
    input_id: str
    protocol: tuple[str, ...]

    def predicts(self, input_id: str, protocol: tuple[str, ...]) -> bool:
        if self.input_id != input_id:
            return False
        if self.protocol == protocol:
            return True
        expected = _wait_duration(self.protocol)
        observed = _wait_duration(protocol)
        return (
            expected is not None
            and observed is not None
            and _without_wait_duration(self.protocol) == _without_wait_duration(protocol)
            and observed >= expected
        )


def _wait_duration(protocol: tuple[str, ...]) -> int | None:
    waits = [token for token in protocol if token.startswith("wait:")]
    if len(waits) != 1:
        return None
    return int(waits[0].split(":", 1)[1])


def _without_wait_duration(protocol: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("wait" if token.startswith("wait:") else token for token in protocol)


class MachineVersionSpace:
    def __init__(self, resources: tuple[str, ...]) -> None:
        self.resources = tuple(sorted(resources))
        self.candidates = [
            MachineHypothesis(input_id=item, protocol=protocol)
            for item in self.resources
            for protocol in PROTOCOL_CANDIDATES
        ]
        self._initial_candidate_count = len(self.candidates)
        self.observations: list[tuple[str, tuple[str, ...], bool, str, str | None]] = []
        self.control_failures: dict[tuple[str, tuple[str, ...]], int] = {}

    def record_control_failure(
        self, input_id: str, protocol: tuple[str, ...]
    ) -> None:
        pair = (input_id, protocol)
        self.control_failures[pair] = self.control_failures.get(pair, 0) + 1

    def observe(
        self,
        input_id: str,
        protocol: tuple[str, ...],
        success: bool,
        evidence_id: str,
        output_id: str | None,
    ) -> None:
        self.observations.append((input_id, protocol, success, evidence_id, output_id))
        self.candidates = [
            c for c in self.candidates if c.predicts(input_id, protocol) is success
        ]
        if not self.candidates:
            raise RuntimeError("machine theory version space collapsed")

    def predictive_outcomes(
        self, input_id: str, protocol: tuple[str, ...]
    ) -> dict[str, float]:
        probability = sum(
            candidate.predicts(input_id, protocol) for candidate in self.candidates
        ) / len(self.candidates)
        return {"success": probability, "failure": 1.0 - probability}

    def next_experiment(self) -> tuple[str, tuple[str, ...]] | None:
        pairs, features = self.experiment_features()
        return select_best_experiment(pairs, features)

    def experiment_features(
        self,
    ) -> tuple[
        tuple[tuple[str, tuple[str, ...]], ...],
        tuple[tuple[float, ...], ...],
    ]:
        tested = {(item, protocol) for item, protocol, _, _, _ in self.observations}
        posterior_fraction = len(self.candidates) / self._initial_candidate_count
        pairs: list[tuple[str, tuple[str, ...]]] = []
        features: list[tuple[float, ...]] = []
        for item in self.resources:
            for protocol in PROTOCOL_CANDIDATES:
                success_count = sum(
                    candidate.predicts(item, protocol) for candidate in self.candidates
                )
                probability = success_count / len(self.candidates)
                entropy = (
                    0.0
                    if probability in (0.0, 1.0)
                    else -probability * math.log2(probability)
                    - (1.0 - probability) * math.log2(1.0 - probability)
                )
                pair = (item, protocol)
                wait_cost = sum(
                    int(token.split(":", 1)[1])
                    for token in protocol
                    if token.startswith("wait:")
                )
                action_cost = len(protocol) - int(wait_cost > 0) + wait_cost
                pairs.append(pair)
                features.append(
                    (
                        max(probability, 1.0 - probability),
                        entropy,
                        float(pair in tested),
                        posterior_fraction,
                        action_cost / 5.0,
                        self.control_failures.get(pair, 0)
                        / (self.control_failures.get(pair, 0) + 1),
                    )
                )
        return tuple(pairs), tuple(features)

    def resolved(self) -> bool:
        _, features = self.experiment_features()
        return all(
            tested == 1.0 or entropy == 0.0
            for _, entropy, tested, *_ in features
        )

    def best(self) -> MachineHypothesis:
        return min(self.candidates, key=lambda c: (c.input_id, c.protocol))

    def to_ir(self, machine_id: str) -> TheoryIR:
        best = self.best()
        output = next(
            (row[4] for row in self.observations if row[2] and row[4] is not None),
            None,
        )
        return TheoryIR(
            mechanism="machine_protocol",
            representation=TheoryRepresentation.STATE_MACHINE,
            scope={"machine_id": machine_id},
            latent_variables=("machine_state",),
            preconditions=("protocol_prefix_matches",),
            effects={"output_id": output, "terminal_success": True},
            parameters={"input_id": best.input_id, "protocol": list(best.protocol)},
        )


@dataclass(frozen=True, slots=True)
class TransformationHypothesis:
    mapping: dict[str, tuple[str, str]]

    def predict(self, processor_id: str, input_id: str) -> str | None:
        source, target = self.mapping[processor_id]
        return target if source == input_id else None


class TransformationLearner:
    """Version space over opaque processor-labelled Hamiltonian resource paths."""

    _UNSEEN_TERMINAL = "<unseen-terminal>"

    def __init__(self, processors: tuple[str, ...], resources: tuple[str, ...]) -> None:
        self.processors = tuple(sorted(processors))
        self.inputs = tuple(sorted(resources))
        self.resources = self.inputs
        self._terminal_revealed = False
        if len(self.resources) < len(self.processors) + 1:
            raise ValueError(
                "a transformation chain needs at least one more resource than processor"
            )
        self.candidates = self._candidate_universe()
        self._initial_candidate_count = len(self.candidates)
        self.mapping: dict[str, tuple[str, str]] = {}
        self.observations: list[tuple[str, str, bool, str | None, str]] = []
        self.control_failures: dict[tuple[str, str], int] = {}

    def record_control_failure(self, processor_id: str, input_id: str) -> None:
        pair = (processor_id, input_id)
        self.control_failures[pair] = self.control_failures.get(pair, 0) + 1

    def grounded_candidates(self) -> tuple[TransformationHypothesis, ...]:
        """Candidates whose outputs are already members of the public universe."""

        return tuple(
            candidate
            for candidate in self.candidates
            if all(
                target != self._UNSEEN_TERMINAL
                for _, target in candidate.mapping.values()
            )
        )

    def _candidate_universe(self) -> list[TransformationHypothesis]:
        path_length = len(self.processors) + 1
        candidates = [
            TransformationHypothesis(
                mapping={
                    processor: (resource_order[index], resource_order[index + 1])
                    for index, processor in enumerate(processor_order)
                }
            )
            for resource_order in itertools.permutations(self.resources, path_length)
            for processor_order in itertools.permutations(self.processors)
        ]
        if not self._terminal_revealed:
            # The live inventory may contain a mechanism-specific distractor while
            # the real terminal product does not exist until the final edge runs.
            # Keep that possibility explicit so an all-known path cannot resolve
            # merely because opaque ordering happened to favor it.
            for sources in itertools.permutations(self.inputs, len(self.processors)):
                resource_order = (*sources, self._UNSEEN_TERMINAL)
                candidates.extend(
                    TransformationHypothesis(
                        mapping={
                            processor: (
                                resource_order[index], resource_order[index + 1]
                            )
                            for index, processor in enumerate(processor_order)
                        }
                    )
                    for processor_order in itertools.permutations(self.processors)
                )
        return candidates

    def observe(
        self,
        processor_id: str,
        input_id: str,
        success: bool,
        output_id: str | None,
        evidence_id: str,
    ) -> None:
        self.observations.append(
            (processor_id, input_id, success, output_id, evidence_id)
        )
        if success != (output_id is not None):
            raise ValueError("successful transformations require an output")
        if output_id is not None and output_id not in self.resources:
            # Live worlds can reveal a terminal product that was absent from the
            # initial inventory. Expand the opaque universe and replay evidence;
            # never reinterpret the new ID semantically.
            self.resources = tuple(sorted((*self.resources, output_id)))
            self._terminal_revealed = True
            self.candidates = self._candidate_universe()
            self._initial_candidate_count = max(
                self._initial_candidate_count, len(self.candidates)
            )
            for observed_processor, observed_input, _, observed_output, _ in self.observations:
                self.candidates = [
                    candidate
                    for candidate in self.candidates
                    if candidate.predict(observed_processor, observed_input)
                    == observed_output
                ]
        else:
            self.candidates = [
                candidate
                for candidate in self.candidates
                if candidate.predict(processor_id, input_id) == output_id
            ]
        if not self.candidates:
            raise RuntimeError("transformation theory version space collapsed")
        self._refresh_mapping()

    def predictive_outcomes(
        self, processor_id: str, input_id: str
    ) -> dict[str, float]:
        counts: dict[str, int] = {}
        for candidate in self.candidates:
            output = candidate.predict(processor_id, input_id)
            key = output if output is not None else "<no-output>"
            counts[key] = counts.get(key, 0) + 1
        return {
            key: count / len(self.candidates) for key, count in sorted(counts.items())
        }

    def next_experiment(self) -> tuple[str, str] | None:
        pairs, features = self.experiment_features()
        selected = select_best_experiment(pairs, features)
        if selected is not None:
            return selected
        tested = {
            (processor, item) for processor, item, _, _, _ in self.observations
        }
        # If the only remaining uncertainty is the identity of a not-yet-created
        # terminal product, instantiate it with its uniquely predicted source edge.
        return next(
            (
                pair
                for pair in pairs
                if pair not in tested
                and self.candidates
                and all(
                    candidate.predict(*pair) == self._UNSEEN_TERMINAL
                    for candidate in self.candidates
                )
            ),
            None,
        )

    def experiment_features(
        self,
    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[float, ...], ...]]:
        tested = {
            (processor, item) for processor, item, _, _, _ in self.observations
        }
        entropy_scale = max(1.0, math.log2(len(self.candidates)))
        posterior_fraction = len(self.candidates) / self._initial_candidate_count
        pairs: list[tuple[str, str]] = []
        features: list[tuple[float, ...]] = []
        for processor in self.processors:
            for item in self.inputs:
                counts: dict[str | None, int] = {}
                for candidate in self.candidates:
                    outcome = candidate.predict(processor, item)
                    counts[outcome] = counts.get(outcome, 0) + 1
                entropy = -sum(
                    (count / len(self.candidates))
                    * math.log2(count / len(self.candidates))
                    for count in counts.values()
                )
                pair = (processor, item)
                pairs.append(pair)
                features.append(
                    (
                        max(counts.values()) / len(self.candidates),
                        entropy / entropy_scale,
                        float(pair in tested),
                        posterior_fraction,
                        1.0,
                        self.control_failures.get(pair, 0)
                        / (self.control_failures.get(pair, 0) + 1),
                    )
                )
        return tuple(pairs), tuple(features)

    def resolved(self) -> bool:
        if any(
            self._UNSEEN_TERMINAL in pair
            for candidate in self.candidates
            for pair in candidate.mapping.values()
        ):
            return False
        _, features = self.experiment_features()
        return all(
            tested == 1.0 or entropy == 0.0
            for _, entropy, tested, *_ in features
        )

    def best(self) -> TransformationHypothesis:
        return min(
            self.candidates,
            key=lambda candidate: json.dumps(candidate.mapping, sort_keys=True),
        )

    def _refresh_mapping(self) -> None:
        first = self.candidates[0].mapping
        self.mapping = {
            processor: pair
            for processor, pair in first.items()
            if all(candidate.mapping[processor] == pair for candidate in self.candidates)
        }

    def to_ir(self) -> TheoryIR:
        if not self.resolved():
            raise RuntimeError("transformation chain is not behaviorally resolved")
        self.mapping = dict(self.best().mapping)
        graph = nx.DiGraph()
        for processor, (source, target) in self.mapping.items():
            graph.add_edge(source, target, processor=processor)
        if not nx.is_directed_acyclic_graph(graph):
            raise RuntimeError("transformation program must be acyclic")
        ordered_resources = list(nx.topological_sort(graph))
        ordered_edges = [
            f"{source}->{target}"
            for source, target in itertools.pairwise(ordered_resources)
            if graph.has_edge(source, target)
        ]
        return TheoryIR(
            mechanism="transformation_chain",
            representation=TheoryRepresentation.PROGRAM,
            scope={"processors": list(self.processors)},
            preconditions=("input_available", "processor_available"),
            effects={"operation": "transform(input) -> output"},
            parameters={
                "mapping": {key: list(value) for key, value in self.mapping.items()},
                "action_pattern": [
                    "place_input",
                    "activate",
                    "wait",
                    "interact",
                ],
            },
            composition=tuple(ordered_edges),
        )


class HeuristicHypothesisGenerator:
    """Creates finite candidate spaces from affordances, never object names."""

    def classify(self, observation: Observation) -> dict[str, tuple[str, ...] | str]:
        selectable = tuple(
            obj.object_id
            for obj in observation.objects
            if tuple(a.value for a in obj.affordances) == ("select",)
        )
        durable_tools = tuple(
            obj.object_id
            for obj in observation.objects
            if tuple(a.value for a in obj.affordances) == ("select",)
            and obj.observable_features.get("durable") is True
        )
        tools = durable_tools or selectable
        targets = tuple(
            obj.object_id
            for obj in observation.objects
            if tuple(a.value for a in obj.affordances) == ("interact",)
        )
        station_objects = [
            obj
            for obj in observation.objects
            if {a.value for a in obj.affordances}
            == {"place_input", "activate", "interact"}
        ]
        if station_objects and all(
            "distance" in obj.observable_features for obj in station_objects
        ):
            station_objects.sort(
                key=lambda obj: (
                    float(obj.observable_features["distance"]),
                    obj.object_id,
                )
            )
        stations = tuple(obj.object_id for obj in station_objects)
        return {
            "tools": tools,
            "targets": targets,
            "machine": stations[0],
            "processors": stations[1:],
            "resources": tuple(sorted(set(observation.inventory) - set(tools))),
        }

    def propose(self, observation: Observation) -> tuple[Hypothesis, ...]:
        groups = self.classify(observation)
        return (
            Hypothesis(
                hypothesis_id="h-tool-order",
                mechanism="tool_capability",
                payload={"objects": groups["tools"], "targets": groups["targets"]},
            ),
            Hypothesis(
                hypothesis_id="h-machine-protocol",
                mechanism="machine_protocol",
                payload={"machine": groups["machine"]},
            ),
            Hypothesis(
                hypothesis_id="h-transform-chain",
                mechanism="transformation_chain",
                payload={"processors": groups["processors"]},
            ),
        )


class OpenAIHypothesisGenerator:
    """Optional proposal-only adapter; formal acceptance remains local."""

    def __init__(self, model: str = "gpt-6-astra") -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI generator")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional dependency with: uv sync --extra openai"
            ) from exc
        self._client = OpenAI()
        self._model = model

    def propose(self, observation: Observation) -> tuple[Hypothesis, ...]:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hypothesis_id": {"type": "string"},
                            "mechanism": {"type": "string"},
                            "payload": {"type": "string"},
                        },
                        "required": ["hypothesis_id", "mechanism", "payload"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["hypotheses"],
            "additionalProperties": False,
        }
        response = self._client.responses.create(
            model=self._model,
            store=False,
            instructions=(
                "Propose testable mechanism hypotheses using only opaque IDs and observations. "
                "Do not use Minecraft knowledge. Output only the requested schema."
            ),
            input=observation.model_dump_json(),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "hypotheses",
                    "schema": schema,
                }
            },
        )
        payload = json.loads(response.output_text)
        return tuple(
            Hypothesis(
                hypothesis_id=item["hypothesis_id"],
                mechanism=item["mechanism"],
                payload={"proposal": item["payload"]},
            )
            for item in payload["hypotheses"]
        )
