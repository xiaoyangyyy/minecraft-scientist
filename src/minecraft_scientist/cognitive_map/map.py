"""Spatial knowledge graph derived from observations, theories, and open questions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field


class MapNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    center: tuple[float, float, float]
    visits: int = Field(default=0, ge=0)
    first_step: int = Field(ge=0)
    last_step: int = Field(ge=0)
    block_histogram: dict[str, int] = Field(default_factory=dict)
    entity_histogram: dict[str, int] = Field(default_factory=dict)
    theory_ids: tuple[str, ...] = ()
    unresolved_hypotheses: tuple[str, ...] = ()
    available_experiments: tuple[str, ...] = ()
    uncertainty: float = Field(default=1.0, ge=0, le=1)
    risk: float = Field(default=0.0, ge=0)


class CognitiveMap:
    def __init__(self, *, cell_size: int = 8) -> None:
        if cell_size < 1:
            raise ValueError("cell_size must be positive")
        self.cell_size = cell_size
        self.graph: nx.Graph = nx.Graph()
        self.current_node_id: str | None = None

    def ingest(
        self,
        observation: dict[str, Any],
        *,
        theory_ids: tuple[str, ...] = (),
        unresolved_hypotheses: tuple[str, ...] = (),
        available_experiments: tuple[str, ...] = (),
        uncertainty: float | None = None,
    ) -> str:
        position = observation["position"]
        coordinates = (
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
        )
        node_id = self.node_id_for(coordinates)
        step = int(observation.get("step", 0))
        blocks = self._histogram(observation.get("blocks", ()), "name")
        entities = self._histogram(observation.get("entities", ()), "name")
        risk = self._risk(observation)
        if node_id in self.graph:
            previous: MapNode = self.graph.nodes[node_id]["data"]
            node = previous.model_copy(
                update={
                    "visits": previous.visits + 1,
                    "last_step": step,
                    "block_histogram": self._merge_histograms(
                        previous.block_histogram, blocks
                    ),
                    "entity_histogram": self._merge_histograms(
                        previous.entity_histogram, entities
                    ),
                    "theory_ids": tuple(
                        sorted(set(previous.theory_ids) | set(theory_ids))
                    ),
                    "unresolved_hypotheses": tuple(
                        sorted(
                            set(previous.unresolved_hypotheses)
                            | set(unresolved_hypotheses)
                        )
                    ),
                    "available_experiments": tuple(
                        sorted(
                            set(previous.available_experiments)
                            | set(available_experiments)
                        )
                    ),
                    "uncertainty": previous.uncertainty
                    if uncertainty is None
                    else uncertainty,
                    "risk": max(previous.risk, risk),
                }
            )
        else:
            node = MapNode(
                node_id=node_id,
                center=coordinates,
                visits=1,
                first_step=step,
                last_step=step,
                block_histogram=blocks,
                entity_histogram=entities,
                theory_ids=tuple(sorted(set(theory_ids))),
                unresolved_hypotheses=tuple(sorted(set(unresolved_hypotheses))),
                available_experiments=tuple(sorted(set(available_experiments))),
                uncertainty=1.0 if uncertainty is None else uncertainty,
                risk=risk,
            )
        self.graph.add_node(node_id, data=node)
        if self.current_node_id is not None and self.current_node_id != node_id:
            source: MapNode = self.graph.nodes[self.current_node_id]["data"]
            distance = math.dist(source.center, node.center)
            self.graph.add_edge(
                self.current_node_id,
                node_id,
                distance=distance,
                traversals=self.graph.get_edge_data(
                    self.current_node_id, node_id, default={}
                ).get("traversals", 0)
                + 1,
            )
        self.current_node_id = node_id
        return node_id

    def node_id_for(self, position: tuple[float, float, float]) -> str:
        cell = tuple(math.floor(value / self.cell_size) for value in position)
        return f"cell:{cell[0]}:{cell[1]}:{cell[2]}"

    def route_to_experiment(self, experiment_id: str) -> tuple[str, ...] | None:
        if self.current_node_id is None:
            return None
        candidates = [
            node_id
            for node_id, attributes in self.graph.nodes(data=True)
            if experiment_id in attributes["data"].available_experiments
        ]
        if not candidates:
            return None
        routes: list[tuple[float, tuple[str, ...]]] = []
        for candidate in candidates:
            try:
                path = nx.shortest_path(
                    self.graph,
                    self.current_node_id,
                    candidate,
                    weight="distance",
                )
            except nx.NetworkXNoPath:
                continue
            travel = nx.path_weight(self.graph, path, weight="distance")
            target: MapNode = self.graph.nodes[candidate]["data"]
            epistemic_bonus = target.uncertainty * self.cell_size
            score = travel + target.risk * self.cell_size - epistemic_bonus
            routes.append((score, tuple(path)))
        return min(routes, default=(0.0, ()))[1] or None

    def save(self, path: Path) -> None:
        payload = {
            "format_version": 1,
            "cell_size": self.cell_size,
            "current_node_id": self.current_node_id,
            "nodes": [
                attributes["data"].model_dump(mode="json")
                for _, attributes in sorted(self.graph.nodes(data=True))
            ],
            "edges": [
                {"source": source, "target": target, **attributes}
                for source, target, attributes in sorted(self.graph.edges(data=True))
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> CognitiveMap:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("unsupported cognitive map format")
        result = cls(cell_size=int(payload["cell_size"]))
        for item in payload["nodes"]:
            node = MapNode.model_validate(item)
            result.graph.add_node(node.node_id, data=node)
        for edge in payload["edges"]:
            result.graph.add_edge(
                edge["source"],
                edge["target"],
                distance=float(edge["distance"]),
                traversals=int(edge["traversals"]),
            )
        result.current_node_id = payload["current_node_id"]
        return result

    @staticmethod
    def _histogram(items: Any, key: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in items:
            name = str(item.get(key, "unknown"))
            result[name] = result.get(name, 0) + 1
        return result

    @staticmethod
    def _merge_histograms(
        first: dict[str, int], second: dict[str, int]
    ) -> dict[str, int]:
        return {
            key: max(first.get(key, 0), second.get(key, 0))
            for key in set(first) | set(second)
        }

    @staticmethod
    def _risk(observation: dict[str, Any]) -> float:
        health = max(0.0, min(20.0, float(observation.get("health", 20.0))))
        food = max(0.0, min(20.0, float(observation.get("food", 20.0))))
        nearby_entities = len(observation.get("entities", ()))
        return (
            (20.0 - health) / 20.0
            + (20.0 - food) / 40.0
            + min(nearby_entities / 10.0, 1.0)
        )
