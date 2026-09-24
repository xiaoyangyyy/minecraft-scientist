"""Coordinate-free topological memory for visually learned routes.

The spatial :class:`~minecraft_scientist.cognitive_map.CognitiveMap` is useful for
the Mineflayer backend, where position is part of the public observation.  Visual
MineStudio experiments intentionally expose no coordinates, so they need a
separate representation.  This module stores opaque, quantized appearance
signatures and functional action-role programs without metric positions or raw
motor handles.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, field_validator


class VisualLandmark(BaseModel):
    """An opaque appearance landmark observed by the Agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    signature: tuple[int, ...]
    observations: int = Field(default=1, ge=1)

    @field_validator("signature")
    @classmethod
    def signature_is_nonempty(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("visual signature must not be empty")
        return value


class VisualRoute(BaseModel):
    """A learned functional action program between two visual landmarks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route_id: str
    source_id: str
    target_id: str
    action_roles: tuple[str, ...]
    trials: int = Field(ge=1)
    successes: int = Field(ge=0)
    mean_cost: float = Field(gt=0)

    @field_validator("action_roles")
    @classmethod
    def roles_are_opaque_tokens(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item or item.strip() != item for item in value):
            raise ValueError("route must contain nonempty normalized action roles")
        return value

    @property
    def confidence(self) -> float:
        """Smoothed empirical success probability."""

        return (self.successes + 1) / (self.trials + 2)


class VisualRoutePlan(BaseModel):
    """A composed route selected from topological memory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    action_roles: tuple[str, ...]
    expected_cost: float = Field(gt=0)
    confidence: float = Field(gt=0, le=1)

    def bind(self, role_handles: Mapping[str, str]) -> tuple[str, ...]:
        """Bind functional roles to the current episode's opaque motor handles."""

        missing = sorted(set(self.action_roles) - set(role_handles))
        if missing:
            raise ValueError(f"missing bindings for action roles: {missing}")
        handles = tuple(role_handles[role] for role in self.action_roles)
        if any(not handle for handle in handles):
            raise ValueError("bound handles must be nonempty")
        return handles


class VisualTopologicalMap:
    """Persistent coordinate-free graph of visual landmarks and route programs."""

    FORMAT_VERSION = 1

    def __init__(self, *, match_tolerance: float = 0.0) -> None:
        if not math.isfinite(match_tolerance) or match_tolerance < 0:
            raise ValueError("match_tolerance must be finite and nonnegative")
        self.match_tolerance = float(match_tolerance)
        self.graph: nx.DiGraph = nx.DiGraph()

    @staticmethod
    def node_id_for(signature: Iterable[int]) -> str:
        normalized = tuple(int(value) for value in signature)
        if not normalized:
            raise ValueError("visual signature must not be empty")
        payload = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
        return "view_" + hashlib.blake2s(payload, digest_size=8).hexdigest()

    def observe(self, signature: Iterable[int]) -> str:
        """Merge a signature with its nearest compatible landmark or add one."""

        normalized = tuple(int(value) for value in signature)
        if not normalized:
            raise ValueError("visual signature must not be empty")
        matched = self.match(normalized)
        if matched is not None:
            previous: VisualLandmark = self.graph.nodes[matched]["data"]
            self.graph.nodes[matched]["data"] = previous.model_copy(
                update={"observations": previous.observations + 1}
            )
            return matched
        node_id = self.node_id_for(normalized)
        self.graph.add_node(
            node_id,
            data=VisualLandmark(node_id=node_id, signature=normalized),
        )
        return node_id

    def match(self, signature: Iterable[int]) -> str | None:
        normalized = tuple(int(value) for value in signature)
        candidates: list[tuple[float, str]] = []
        for node_id, attributes in self.graph.nodes(data=True):
            landmark: VisualLandmark = attributes["data"]
            if len(landmark.signature) != len(normalized):
                continue
            distance = math.sqrt(sum(
                (left - right) ** 2
                for left, right in zip(
                    landmark.signature, normalized, strict=True
                )
            ))
            if distance <= self.match_tolerance:
                candidates.append((distance, node_id))
        return min(candidates, default=(0.0, None))[1]

    def record_route(
        self,
        source_signature: Iterable[int],
        target_signature: Iterable[int],
        action_roles: Iterable[str],
        *,
        success: bool,
        cost: float | None = None,
    ) -> str:
        """Record one route trial without retaining coordinates or raw handles."""

        roles = tuple(str(role) for role in action_roles)
        if not roles:
            raise ValueError("route must contain at least one action role")
        route_cost = float(len(roles) if cost is None else cost)
        if not math.isfinite(route_cost) or route_cost <= 0:
            raise ValueError("route cost must be finite and positive")
        source_id = self.observe(source_signature)
        target_id = self.observe(target_signature)
        identity = json.dumps(
            [source_id, target_id, roles], separators=(",", ":")
        ).encode("utf-8")
        route_id = "route_" + hashlib.blake2s(identity, digest_size=8).hexdigest()
        existing = self.graph.get_edge_data(source_id, target_id)
        if existing is not None:
            previous: VisualRoute = existing["data"]
            if previous.route_id != route_id:
                raise ValueError(
                    "a different program already connects these landmarks"
                )
            trials = previous.trials + 1
            successes = previous.successes + int(success)
            mean_cost = (
                previous.mean_cost * previous.trials + route_cost
            ) / trials
        else:
            trials = 1
            successes = int(success)
            mean_cost = route_cost
        route = VisualRoute(
            route_id=route_id,
            source_id=source_id,
            target_id=target_id,
            action_roles=roles,
            trials=trials,
            successes=successes,
            mean_cost=mean_cost,
        )
        self.graph.add_edge(source_id, target_id, data=route)
        return route_id

    def plan(
        self,
        source_signature: Iterable[int],
        target_id: str,
        *,
        minimum_confidence: float = 0.5,
    ) -> VisualRoutePlan | None:
        """Return the safest low-cost route from a current appearance signature."""

        if not 0 < minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in (0, 1]")
        source_id = self.match(source_signature)
        if source_id is None or target_id not in self.graph:
            return None
        usable = nx.DiGraph()
        for left, right, attributes in self.graph.edges(data=True):
            route: VisualRoute = attributes["data"]
            if route.successes and route.confidence >= minimum_confidence:
                usable.add_edge(
                    left,
                    right,
                    weight=route.mean_cost / route.confidence,
                    data=route,
                )
        try:
            nodes = nx.shortest_path(usable, source_id, target_id, weight="weight")
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None
        routes = [usable[left][right]["data"] for left, right in nx.utils.pairwise(nodes)]
        if not routes:
            return None
        return VisualRoutePlan(
            node_ids=tuple(nodes),
            route_ids=tuple(route.route_id for route in routes),
            action_roles=tuple(
                role for route in routes for role in route.action_roles
            ),
            expected_cost=sum(route.mean_cost for route in routes),
            confidence=math.prod(route.confidence for route in routes),
        )

    def save(self, path: Path) -> None:
        payload = {
            "format_version": self.FORMAT_VERSION,
            "match_tolerance": self.match_tolerance,
            "nodes": [
                attributes["data"].model_dump(mode="json")
                for _, attributes in sorted(self.graph.nodes(data=True))
            ],
            "routes": [
                attributes["data"].model_dump(mode="json")
                for _, _, attributes in sorted(self.graph.edges(data=True))
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> VisualTopologicalMap:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported visual topological map format")
        if set(payload) != {
            "format_version", "match_tolerance", "nodes", "routes"
        }:
            raise ValueError("unexpected visual topological map fields")
        result = cls(match_tolerance=float(payload["match_tolerance"]))
        for item in payload["nodes"]:
            node = VisualLandmark.model_validate(item)
            if node.node_id != result.node_id_for(node.signature):
                raise ValueError("visual landmark identifier does not match signature")
            if node.node_id in result.graph:
                raise ValueError("duplicate visual landmark")
            result.graph.add_node(node.node_id, data=node)
        for item in payload["routes"]:
            route = VisualRoute.model_validate(item)
            if route.source_id not in result.graph or route.target_id not in result.graph:
                raise ValueError("visual route references an unknown landmark")
            identity = json.dumps(
                [route.source_id, route.target_id, route.action_roles],
                separators=(",", ":"),
            ).encode("utf-8")
            expected_id = "route_" + hashlib.blake2s(
                identity, digest_size=8
            ).hexdigest()
            if route.route_id != expected_id:
                raise ValueError("visual route identifier does not match program")
            if result.graph.has_edge(route.source_id, route.target_id):
                raise ValueError("duplicate visual route")
            result.graph.add_edge(route.source_id, route.target_id, data=route)
        return result
