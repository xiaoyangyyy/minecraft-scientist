"""Local causal effects estimated only from compliant controlled interventions."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import networkx as nx

from .models import CausalSample, ControlledEffect


class LocalCausalEngine:
    def __init__(self, samples: Iterable[CausalSample] = ()) -> None:
        self.samples = list(samples)

    def add(self, sample: CausalSample) -> None:
        self.samples.append(sample)

    def controlled_effect(
        self,
        treatment: str,
        outcome: str,
        *,
        fixed_context: dict[str, str | int | float | bool] | None = None,
    ) -> ControlledEffect:
        controls = fixed_context or {}
        groups: dict[str, list[float]] = defaultdict(list)
        eligible = [
            sample
            for sample in self.samples
            if sample.compliant
            and not sample.possible_confounders
            and treatment in sample.intervention
            and outcome in sample.outcome
            and all(sample.context.get(key) == value for key, value in controls.items())
        ]
        for sample in eligible:
            raw_outcome = sample.outcome[outcome]
            if isinstance(raw_outcome, bool) or not isinstance(
                raw_outcome, (int, float)
            ):
                raise TypeError(f"outcome {outcome!r} must be numeric")
            level = json.dumps(
                sample.intervention[treatment],
                sort_keys=True,
                separators=(",", ":"),
            )
            groups[level].append(float(raw_outcome))
        if len(groups) < 2:
            raise ValueError(
                "a controlled causal effect requires at least two treatment levels"
            )
        means = {
            level: sum(values) / len(values) for level, values in sorted(groups.items())
        }
        effect = max(means.values()) - min(means.values())
        balance = min(len(values) for values in groups.values()) / max(
            len(values) for values in groups.values()
        )
        replication = min(1.0, len(eligible) / (2 * len(groups)))
        return ControlledEffect(
            treatment=treatment,
            outcome=outcome,
            fixed_context=controls,
            group_means=means,
            absolute_effect=effect,
            sample_count=len(eligible),
            confidence=balance * replication,
        )

    def discover_graph(
        self,
        *,
        treatments: Iterable[str],
        outcomes: Iterable[str],
        fixed_context: dict[str, str | int | float | bool] | None = None,
        minimum_effect: float = 1e-9,
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        for treatment in sorted(set(treatments)):
            graph.add_node(treatment, kind="intervention")
            for outcome in sorted(set(outcomes)):
                graph.add_node(outcome, kind="outcome")
                try:
                    effect = self.controlled_effect(
                        treatment,
                        outcome,
                        fixed_context=fixed_context,
                    )
                except (TypeError, ValueError):
                    continue
                if effect.absolute_effect >= minimum_effect:
                    graph.add_edge(
                        treatment,
                        outcome,
                        effect=effect.absolute_effect,
                        confidence=effect.confidence,
                        sample_count=effect.sample_count,
                    )
        return graph

    @staticmethod
    def graph_payload(graph: nx.DiGraph) -> dict[str, Any]:
        return {
            "nodes": [
                {"id": node, **attributes}
                for node, attributes in sorted(graph.nodes(data=True))
            ],
            "edges": [
                {"source": source, "target": target, **attributes}
                for source, target, attributes in sorted(graph.edges(data=True))
            ],
        }
