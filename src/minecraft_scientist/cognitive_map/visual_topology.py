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
from enum import StrEnum
from pathlib import Path

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VisualRouteStatus(StrEnum):
    """Lifecycle state for a learned visual route program."""

    ACTIVE = "active"
    CHALLENGED = "challenged"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class VisualMatchCalibration(BaseModel):
    """Auditable per-landmark radius fitted from positive and negative views."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    positive_samples: int = Field(ge=2)
    negative_samples: int = Field(ge=1)
    max_positive_distance: float = Field(ge=0)
    nearest_negative_distance: float = Field(gt=0)
    margin: float = Field(ge=0)
    radius: float = Field(ge=0)

    @model_validator(mode="after")
    def boundary_is_finite_and_separated(self) -> VisualMatchCalibration:
        values = (
            self.max_positive_distance,
            self.nearest_negative_distance,
            self.margin,
            self.radius,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("visual match calibration must be finite")
        if not math.isclose(
            self.radius,
            self.max_positive_distance + self.margin,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("calibrated radius must equal positive spread plus margin")
        if self.radius >= self.nearest_negative_distance - self.margin:
            raise ValueError("positive and negative landmark views are not separated")
        return self


class VisualLandmark(BaseModel):
    """An opaque appearance landmark observed by the Agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    signature: tuple[int, ...]
    observations: int = Field(default=1, ge=1)
    calibration: VisualMatchCalibration | None = None

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
    status: VisualRouteStatus = VisualRouteStatus.ACTIVE
    failure_streak: int = Field(default=0, ge=0)
    status_reason: str | None = None
    parent_route_id: str | None = None
    superseded_by: str | None = None

    @field_validator("action_roles")
    @classmethod
    def roles_are_opaque_tokens(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item or item.strip() != item for item in value):
            raise ValueError("route must contain nonempty normalized action roles")
        return value

    @model_validator(mode="after")
    def lifecycle_is_consistent(self) -> VisualRoute:
        if self.successes > self.trials:
            raise ValueError("route successes cannot exceed trials")
        if self.failure_streak > self.trials:
            raise ValueError("route failure streak cannot exceed trials")
        if self.status_reason is not None and (
            not self.status_reason or self.status_reason.strip() != self.status_reason
        ):
            raise ValueError("route status reason must be normalized and nonempty")
        if self.status is not VisualRouteStatus.ACTIVE and self.status_reason is None:
            raise ValueError("non-active route must record a status reason")
        terminal = self.status is VisualRouteStatus.SUPERSEDED
        if terminal != (self.superseded_by is not None):
            raise ValueError("only superseded routes must name a replacement")
        if self.parent_route_id == self.route_id:
            raise ValueError("route cannot be its own parent")
        if self.superseded_by == self.route_id:
            raise ValueError("route cannot supersede itself")
        return self

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

    FORMAT_VERSION = 4

    def __init__(
        self,
        *,
        match_tolerance: float = 0.0,
        ambiguity_margin: float = 0.0,
    ) -> None:
        if not math.isfinite(match_tolerance) or match_tolerance < 0:
            raise ValueError("match_tolerance must be finite and nonnegative")
        if not math.isfinite(ambiguity_margin) or ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must be finite and nonnegative")
        self.match_tolerance = float(match_tolerance)
        self.ambiguity_margin = float(ambiguity_margin)
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

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

    @staticmethod
    def _distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        if len(left) != len(right):
            raise ValueError("visual signatures must have equal dimensions")
        return math.sqrt(sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left, right, strict=True)
        ))

    def landmarks(self) -> tuple[VisualLandmark, ...]:
        """Return an immutable deterministic view of all visual landmarks."""

        return tuple(
            attributes["data"]
            for _, attributes in sorted(self.graph.nodes(data=True))
        )

    def calibrate_landmark(
        self,
        node_id: str,
        positive_signatures: Iterable[Iterable[int]],
        negative_signatures: Iterable[Iterable[int]],
        *,
        margin: float = 1.0,
    ) -> VisualLandmark:
        """Fit a conservative match radius from controlled labeled views.

        Positive views estimate within-landmark spread. Negative views do not
        enlarge the accepted region; they prove that the proposed positive spread
        plus margin remains separated from the nearest known alternative.
        """

        if not math.isfinite(margin) or margin < 0:
            raise ValueError("calibration margin must be finite and nonnegative")
        if node_id not in self.graph:
            raise KeyError(f"unknown visual landmark: {node_id}")
        landmark: VisualLandmark = self.graph.nodes[node_id]["data"]
        positives = tuple(
            tuple(int(value) for value in signature)
            for signature in positive_signatures
        )
        negatives = tuple(
            tuple(int(value) for value in signature)
            for signature in negative_signatures
        )
        if len(positives) < 2:
            raise ValueError("landmark calibration requires at least two positives")
        if not negatives:
            raise ValueError("landmark calibration requires at least one negative")
        if any(not signature for signature in (*positives, *negatives)):
            raise ValueError("calibration signatures must not be empty")
        positive_distances = tuple(
            self._distance(landmark.signature, signature)
            for signature in positives
        )
        negative_distances = tuple(
            self._distance(landmark.signature, signature)
            for signature in negatives
        )
        maximum_positive = max(positive_distances)
        nearest_negative = min(negative_distances)
        calibration = VisualMatchCalibration(
            positive_samples=len(positives),
            negative_samples=len(negatives),
            max_positive_distance=maximum_positive,
            nearest_negative_distance=nearest_negative,
            margin=margin,
            radius=maximum_positive + margin,
        )
        updated = landmark.model_copy(update={"calibration": calibration})
        updated = VisualLandmark.model_validate(updated.model_dump())
        self.graph.nodes[node_id]["data"] = updated
        return updated

    def match(self, signature: Iterable[int]) -> str | None:
        normalized = tuple(int(value) for value in signature)
        candidates: list[tuple[float, str]] = []
        for node_id, attributes in self.graph.nodes(data=True):
            landmark: VisualLandmark = attributes["data"]
            if len(landmark.signature) != len(normalized):
                continue
            distance = self._distance(landmark.signature, normalized)
            tolerance = (
                landmark.calibration.radius
                if landmark.calibration is not None
                else self.match_tolerance
            )
            if distance <= tolerance:
                candidates.append((distance, node_id))
        candidates.sort()
        if not candidates:
            return None
        if (
            len(candidates) > 1
            and candidates[1][0] - candidates[0][0] <= self.ambiguity_margin
        ):
            return None
        return candidates[0][1]

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
        existing = self.graph.get_edge_data(source_id, target_id, key=route_id)
        if existing is not None:
            previous: VisualRoute = existing["data"]
            trials = previous.trials + 1
            successes = previous.successes + int(success)
            mean_cost = (
                previous.mean_cost * previous.trials + route_cost
            ) / trials
            lifecycle = {
                "status": previous.status,
                "failure_streak": 0 if success else previous.failure_streak + 1,
                "status_reason": previous.status_reason,
                "parent_route_id": previous.parent_route_id,
                "superseded_by": previous.superseded_by,
            }
        else:
            trials = 1
            successes = int(success)
            mean_cost = route_cost
            lifecycle = {"failure_streak": 0 if success else 1}
        route = VisualRoute(
            route_id=route_id,
            source_id=source_id,
            target_id=target_id,
            action_roles=roles,
            trials=trials,
            successes=successes,
            mean_cost=mean_cost,
            **lifecycle,
        )
        self.graph.add_edge(source_id, target_id, key=route_id, data=route)
        return route_id

    def routes(self) -> tuple[VisualRoute, ...]:
        """Return an immutable, deterministic view of all route programs."""

        return tuple(
            attributes["data"]
            for _, _, _, attributes in sorted(
                self.graph.edges(keys=True, data=True)
            )
        )

    def get_route(self, route_id: str) -> VisualRoute:
        """Return one route by stable ID, failing closed for unknown IDs."""

        matches = [route for route in self.routes() if route.route_id == route_id]
        if len(matches) != 1:
            raise KeyError(f"unknown visual route: {route_id}")
        return matches[0]

    def _replace_route(self, route: VisualRoute) -> None:
        if not self.graph.has_edge(
            route.source_id, route.target_id, key=route.route_id
        ):
            raise KeyError(f"unknown visual route: {route.route_id}")
        self.graph[route.source_id][route.target_id][route.route_id]["data"] = route

    def record_outcome(
        self,
        route_id: str,
        *,
        success: bool,
        cost: float | None = None,
        challenge_threshold: float | None = 0.75,
        failure_reason: str | None = None,
    ) -> VisualRoute:
        """Record replay evidence and challenge a confident route on failure.

        The threshold is evaluated against the confidence *before* the new
        observation.  Passing ``None`` records evidence without an automatic
        lifecycle transition.
        """

        if challenge_threshold is not None and not 0 < challenge_threshold <= 1:
            raise ValueError("challenge threshold must be in (0, 1] or None")
        if cost is not None and (not math.isfinite(cost) or cost <= 0):
            raise ValueError("route cost must be finite and positive")
        previous = self.get_route(route_id)
        prior_confidence = previous.confidence
        trials = previous.trials + 1
        successes = previous.successes + int(success)
        mean_cost = previous.mean_cost
        if cost is not None:
            mean_cost = (previous.mean_cost * previous.trials + cost) / trials
        status = previous.status
        status_reason = previous.status_reason
        if (
            not success
            and status is VisualRouteStatus.ACTIVE
            and challenge_threshold is not None
            and prior_confidence >= challenge_threshold
        ):
            status = VisualRouteStatus.CHALLENGED
            status_reason = failure_reason or "high-confidence replay failed"
        updated = previous.model_copy(
            update={
                "trials": trials,
                "successes": successes,
                "mean_cost": mean_cost,
                "failure_streak": 0 if success else previous.failure_streak + 1,
                "status": status,
                "status_reason": status_reason,
            }
        )
        updated = VisualRoute.model_validate(updated.model_dump())
        self._replace_route(updated)
        return updated

    def challenge_route(self, route_id: str, *, reason: str) -> VisualRoute:
        """Explicitly remove an active route from planning after an anomaly."""

        previous = self.get_route(route_id)
        if previous.status is not VisualRouteStatus.ACTIVE:
            raise ValueError("only an active visual route can be challenged")
        updated = previous.model_copy(
            update={"status": VisualRouteStatus.CHALLENGED, "status_reason": reason}
        )
        updated = VisualRoute.model_validate(updated.model_dump())
        self._replace_route(updated)
        return updated

    def reinstate_route(self, route_id: str, *, reason: str) -> VisualRoute:
        """Return a challenged route to service after independent validation."""

        previous = self.get_route(route_id)
        if previous.status is not VisualRouteStatus.CHALLENGED:
            raise ValueError("only a challenged visual route can be reinstated")
        updated = previous.model_copy(
            update={"status": VisualRouteStatus.ACTIVE, "status_reason": reason}
        )
        updated = VisualRoute.model_validate(updated.model_dump())
        self._replace_route(updated)
        return updated

    def reject_route(self, route_id: str, *, reason: str) -> VisualRoute:
        """Permanently reject an active or challenged route program."""

        previous = self.get_route(route_id)
        if previous.status not in {
            VisualRouteStatus.ACTIVE,
            VisualRouteStatus.CHALLENGED,
        }:
            raise ValueError("only an active or challenged visual route can be rejected")
        updated = previous.model_copy(
            update={"status": VisualRouteStatus.REJECTED, "status_reason": reason}
        )
        updated = VisualRoute.model_validate(updated.model_dump())
        self._replace_route(updated)
        return updated

    def supersede_route(
        self,
        route_id: str,
        replacement_route_id: str,
        *,
        reason: str,
    ) -> tuple[VisualRoute, VisualRoute]:
        """Replace a challenged route with a validated competing program."""

        previous = self.get_route(route_id)
        replacement = self.get_route(replacement_route_id)
        if previous.status is not VisualRouteStatus.CHALLENGED:
            raise ValueError("only a challenged visual route can be superseded")
        if replacement.status is not VisualRouteStatus.ACTIVE:
            raise ValueError("replacement visual route must be active")
        if (previous.source_id, previous.target_id) != (
            replacement.source_id,
            replacement.target_id,
        ):
            raise ValueError("replacement must connect the same visual landmarks")
        if previous.route_id == replacement.route_id:
            raise ValueError("replacement visual route must be different")
        superseded = previous.model_copy(
            update={
                "status": VisualRouteStatus.SUPERSEDED,
                "status_reason": reason,
                "superseded_by": replacement.route_id,
            }
        )
        promoted = replacement.model_copy(
            update={"parent_route_id": previous.route_id}
        )
        superseded = VisualRoute.model_validate(superseded.model_dump())
        promoted = VisualRoute.model_validate(promoted.model_dump())
        self._replace_route(superseded)
        self._replace_route(promoted)
        return superseded, promoted

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
        for left, right, route_id, attributes in self.graph.edges(
            keys=True, data=True
        ):
            route: VisualRoute = attributes["data"]
            if (
                route.status is VisualRouteStatus.ACTIVE
                and route.successes
                and route.confidence >= minimum_confidence
            ):
                weight = route.mean_cost / route.confidence
                existing = usable.get_edge_data(left, right)
                if existing is None or (weight, route_id) < (
                    existing["weight"], existing["data"].route_id
                ):
                    usable.add_edge(left, right, weight=weight, data=route)
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
            "ambiguity_margin": self.ambiguity_margin,
            "nodes": [
                attributes["data"].model_dump(mode="json")
                for _, attributes in sorted(self.graph.nodes(data=True))
            ],
            "routes": [
                attributes["data"].model_dump(mode="json")
                for _, _, _, attributes in sorted(
                    self.graph.edges(keys=True, data=True)
                )
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
        format_version = payload.get("format_version")
        if format_version not in {1, 2, 3, cls.FORMAT_VERSION}:
            raise ValueError("unsupported visual topological map format")
        expected_fields = {"format_version", "match_tolerance", "nodes", "routes"}
        if format_version in {2, 3, cls.FORMAT_VERSION}:
            expected_fields.add("ambiguity_margin")
        if set(payload) != expected_fields:
            raise ValueError("unexpected visual topological map fields")
        result = cls(
            match_tolerance=float(payload["match_tolerance"]),
            ambiguity_margin=float(payload.get("ambiguity_margin", 0.0)),
        )
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
            if result.graph.has_edge(
                route.source_id, route.target_id, key=route.route_id
            ):
                raise ValueError("duplicate visual route")
            result.graph.add_edge(
                route.source_id,
                route.target_id,
                key=route.route_id,
                data=route,
            )
        routes = {route.route_id: route for route in result.routes()}
        for route in routes.values():
            if route.parent_route_id is not None:
                parent = routes.get(route.parent_route_id)
                if parent is None or parent.superseded_by != route.route_id:
                    raise ValueError("visual route parent link is inconsistent")
                if (parent.source_id, parent.target_id) != (
                    route.source_id,
                    route.target_id,
                ):
                    raise ValueError("visual route parent endpoints are inconsistent")
            if route.superseded_by is not None:
                replacement = routes.get(route.superseded_by)
                if replacement is None or replacement.parent_route_id != route.route_id:
                    raise ValueError("visual route replacement link is inconsistent")
                if (replacement.source_id, replacement.target_id) != (
                    route.source_id,
                    route.target_id,
                ):
                    raise ValueError(
                        "visual route replacement endpoints are inconsistent"
                    )
        for route in routes.values():
            seen: set[str] = set()
            current = route
            while current.superseded_by is not None:
                if current.route_id in seen:
                    raise ValueError("visual route supersession chain contains a cycle")
                seen.add(current.route_id)
                current = routes[current.superseded_by]
        return result
