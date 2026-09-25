"""Durable, replay-auditable memory for multi-session visual exploration.

The visual topology stores validated route programs. This module covers the earlier
stage of open-ended exploration: recognizing a place from multiple nonsemantic views,
remembering unresolved action frontiers, learning traversable links, and avoiding the
same failed frontier after a restart. SQLite contains an append-only event journal and
materialized projections; every public read verifies the typed projection while
``verify_integrity`` independently checks the journal hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _signature(value: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(item) for item in value)
    if not result:
        raise ValueError("place signature must not be empty")
    return result


def _distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("place signatures must have equal dimensions")
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left, right, strict=True)
        )
    )


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrontierStatus(StrEnum):
    """Whether an unknown outgoing action remains worth attempting."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    BLOCKED = "blocked"


class PlaceCalibration(_FrozenModel):
    """Supervised boundary for a multi-exemplar place representation."""

    positive_samples: int = Field(ge=2)
    negative_samples: int = Field(ge=1)
    maximum_positive_distance: float = Field(ge=0)
    nearest_negative_distance: float = Field(gt=0)
    margin: float = Field(ge=0)
    radius: float = Field(ge=0)

    @model_validator(mode="after")
    def boundary_is_valid(self) -> PlaceCalibration:
        values = (
            self.maximum_positive_distance,
            self.nearest_negative_distance,
            self.margin,
            self.radius,
        )
        if not all(math.isfinite(item) for item in values):
            raise ValueError("place calibration values must be finite")
        if not math.isclose(
            self.radius,
            self.maximum_positive_distance + self.margin,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("place radius must equal positive coverage plus margin")
        if self.radius >= self.nearest_negative_distance - self.margin:
            raise ValueError("positive and negative place views are not separated")
        return self


class PlaceMemory(_FrozenModel):
    """Long-lived anonymous place represented by a bounded multi-view coreset."""

    place_id: str
    exemplars: tuple[tuple[int, ...], ...]
    visits: int = Field(ge=1)
    episodes: tuple[int, ...]
    first_episode: int = Field(ge=0)
    last_episode: int = Field(ge=0)
    first_step: int = Field(ge=0)
    last_step: int = Field(ge=0)
    uncertainty: float = Field(ge=0, le=1)
    risk: float = Field(ge=0)
    match_radius: float = Field(ge=0)
    calibration: PlaceCalibration | None = None

    @field_validator("exemplars")
    @classmethod
    def exemplars_are_compatible(
        cls, value: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
        if not value or not value[0]:
            raise ValueError("place memory requires a nonempty exemplar")
        dimensions = {len(item) for item in value}
        if len(dimensions) != 1 or len(set(value)) != len(value):
            raise ValueError("place exemplars must be unique and equally sized")
        return value

    @model_validator(mode="after")
    def history_is_consistent(self) -> PlaceMemory:
        if tuple(sorted(set(self.episodes))) != self.episodes:
            raise ValueError("place episodes must be unique and sorted")
        if not self.episodes:
            raise ValueError("place memory requires at least one episode")
        if self.first_episode != self.episodes[0]:
            raise ValueError("first episode does not match place episode history")
        if self.last_episode != self.episodes[-1]:
            raise ValueError("last episode does not match place episode history")
        if self.last_episode < self.first_episode:
            raise ValueError("place episode history is reversed")
        if self.last_episode == self.first_episode and self.last_step < self.first_step:
            raise ValueError("place step history is reversed")
        if self.calibration is not None and not math.isclose(
            self.match_radius,
            self.calibration.radius,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("place radius does not match its calibration")
        return self


class PerceptualPlaceMemory(_FrozenModel):
    """Bounded anonymous scene exemplars associated with one spatial place."""

    place_id: str
    exemplars: tuple[tuple[int, ...], ...]
    observations: int = Field(ge=1)
    episodes: tuple[int, ...]
    last_episode: int = Field(ge=0)
    last_step: int = Field(ge=0)

    @field_validator("exemplars")
    @classmethod
    def exemplars_are_compatible(
        cls, value: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
        if not value or not value[0]:
            raise ValueError("perceptual place memory requires an exemplar")
        if len({len(item) for item in value}) != 1 or len(set(value)) != len(value):
            raise ValueError("perceptual exemplars must be unique and equally sized")
        return value

    @field_validator("episodes")
    @classmethod
    def episodes_are_sorted(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("perceptual episodes must be nonempty, unique, and sorted")
        return value

    @model_validator(mode="after")
    def recency_is_consistent(self) -> PerceptualPlaceMemory:
        if self.last_episode != self.episodes[-1]:
            raise ValueError("perceptual recency must match its episode history")
        return self


class FrontierMemory(_FrozenModel):
    """Persistent evidence about one candidate action program from a place."""

    frontier_id: str
    source_place_id: str
    action_roles: tuple[str, ...]
    expected_cost: float = Field(gt=0)
    uncertainty: float = Field(ge=0, le=1)
    risk: float = Field(ge=0)
    attempts: int = Field(default=0, ge=0)
    successes: int = Field(default=0, ge=0)
    failure_streak: int = Field(default=0, ge=0)
    status: FrontierStatus = FrontierStatus.ACTIVE
    discovered_place_id: str | None = None
    last_episode: int | None = Field(default=None, ge=0)
    last_step: int | None = Field(default=None, ge=0)

    @field_validator("action_roles")
    @classmethod
    def action_roles_are_normalized(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item or item.strip() != item for item in value):
            raise ValueError("frontier requires normalized action roles")
        return value

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> FrontierMemory:
        if self.successes > self.attempts or self.failure_streak > self.attempts:
            raise ValueError("frontier evidence counts are inconsistent")
        resolved = self.status is FrontierStatus.RESOLVED
        if resolved != (self.discovered_place_id is not None):
            raise ValueError("only a resolved frontier may name a discovered place")
        if (self.last_episode is None) != (self.last_step is None):
            raise ValueError("frontier recency requires both episode and step")
        return self


class PlaceTransition(_FrozenModel):
    """A replayable directed connection discovered by frontier execution."""

    source_place_id: str
    target_place_id: str
    action_roles: tuple[str, ...]
    successes: int = Field(ge=1)
    failures: int = Field(ge=0)
    mean_cost: float = Field(gt=0)

    @property
    def confidence(self) -> float:
        return (self.successes + 1) / (self.successes + self.failures + 2)


class FrontierPlan(_FrozenModel):
    """Known travel prefix followed by one information-seeking action program."""

    place_path: tuple[str, ...]
    frontier_id: str
    action_roles: tuple[str, ...]
    travel_cost: float = Field(ge=0)
    frontier_cost: float = Field(gt=0)
    utility: float
    uncertainty_gain: float = Field(ge=0)
    failure_probability: float = Field(ge=0, le=1)


class ExplorationMemory:
    """SQLite-backed multi-session place, route, and frontier memory."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        path: Path,
        *,
        default_match_radius: float = 0.0,
        ambiguity_margin: float = 0.0,
        max_exemplars: int = 8,
        block_after_failures: int = 3,
    ) -> None:
        for name, value in (
            ("default_match_radius", default_match_radius),
            ("ambiguity_margin", ambiguity_margin),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if max_exemplars < 1:
            raise ValueError("max_exemplars must be positive")
        if block_after_failures < 1:
            raise ValueError("block_after_failures must be positive")
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.default_match_radius = float(default_match_radius)
        self.ambiguity_margin = float(ambiguity_margin)
        self.max_exemplars = max_exemplars
        self.block_after_failures = block_after_failures
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        self._validate_configuration()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS exploration_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    episode_id INTEGER NOT NULL CHECK (episode_id >= 0),
                    step INTEGER NOT NULL CHECK (step >= 0),
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS places (place_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)"
            )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS perceptual_places ("
                "place_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, "
                "FOREIGN KEY (place_id) REFERENCES places(place_id))"
            )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS frontiers (frontier_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS place_transitions (
                    source_place_id TEXT NOT NULL,
                    target_place_id TEXT NOT NULL,
                    action_roles_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (source_place_id, target_place_id, action_roles_json),
                    FOREIGN KEY (source_place_id) REFERENCES places(place_id),
                    FOREIGN KEY (target_place_id) REFERENCES places(place_id)
                )
                """
            )

    def _configuration(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "default_match_radius": self.default_match_radius,
            "ambiguity_margin": self.ambiguity_margin,
            "max_exemplars": self.max_exemplars,
            "block_after_failures": self.block_after_failures,
        }

    def _validate_configuration(self) -> None:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'configuration'"
        ).fetchone()
        expected = _canonical(self._configuration())
        expected_hash = hashlib.sha256(expected.encode("utf-8")).hexdigest()
        if row is None:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO metadata (key, value) VALUES ('configuration', ?)",
                    (expected,),
                )
                self._connection.execute(
                    "INSERT INTO metadata (key, value) "
                    "VALUES ('configuration_sha256', ?)",
                    (expected_hash,),
                )
        elif row[0] != expected:
            raise ValueError("exploration memory configuration does not match database")
        else:
            hash_row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'configuration_sha256'"
            ).fetchone()
            if hash_row != (expected_hash,):
                raise ValueError("exploration memory configuration hash is invalid")

    @classmethod
    def open(cls, path: Path) -> ExplorationMemory:
        """Open an existing memory using its persisted behavioral configuration."""

        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"exploration memory does not exist: {resolved}")
        connection = sqlite3.connect(resolved)
        try:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'configuration'"
            ).fetchone()
        except sqlite3.Error as error:
            raise ValueError("invalid exploration memory database") from error
        finally:
            connection.close()
        if row is None:
            raise ValueError("exploration memory is missing its configuration")
        configuration = json.loads(row[0])
        if configuration.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported exploration memory format")
        expected = {
            "format_version",
            "default_match_radius",
            "ambiguity_margin",
            "max_exemplars",
            "block_after_failures",
        }
        if set(configuration) != expected:
            raise ValueError("unexpected exploration memory configuration fields")
        return cls(
            resolved,
            default_match_radius=float(configuration["default_match_radius"]),
            ambiguity_margin=float(configuration["ambiguity_margin"]),
            max_exemplars=int(configuration["max_exemplars"]),
            block_after_failures=int(configuration["block_after_failures"]),
        )

    @staticmethod
    def place_id_for(signature: Iterable[int]) -> str:
        normalized = _signature(signature)
        return (
            "place_"
            + hashlib.blake2s(
                _canonical(normalized).encode("utf-8"), digest_size=10
            ).hexdigest()
        )

    @staticmethod
    def frontier_id_for(source_place_id: str, action_roles: Iterable[str]) -> str:
        roles = tuple(str(item) for item in action_roles)
        if not roles:
            raise ValueError("frontier requires at least one action role")
        return (
            "frontier_"
            + hashlib.blake2s(
                _canonical((source_place_id, roles)).encode("utf-8"), digest_size=10
            ).hexdigest()
        )

    def places(self) -> tuple[PlaceMemory, ...]:
        cursor = self._connection.execute(
            "SELECT payload_json FROM places ORDER BY place_id"
        )
        return tuple(PlaceMemory.model_validate_json(row[0]) for row in cursor)

    def get_place(self, place_id: str) -> PlaceMemory:
        row = self._connection.execute(
            "SELECT payload_json FROM places WHERE place_id = ?", (place_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown place: {place_id}")
        return PlaceMemory.model_validate_json(row[0])

    def perceptual_places(self) -> tuple[PerceptualPlaceMemory, ...]:
        cursor = self._connection.execute(
            "SELECT payload_json FROM perceptual_places ORDER BY place_id"
        )
        return tuple(
            PerceptualPlaceMemory.model_validate_json(row[0]) for row in cursor
        )

    def record_perceptual_view(
        self,
        place_id: str,
        signature: Iterable[int],
        *,
        episode_id: int,
        step: int,
    ) -> PerceptualPlaceMemory:
        """Bind an anonymous scene view to a known place using bounded storage."""

        self.get_place(place_id)
        normalized = _signature(signature)
        if episode_id < 0 or step < 0:
            raise ValueError("episode and step must be nonnegative")
        row = self._connection.execute(
            "SELECT payload_json FROM perceptual_places WHERE place_id = ?",
            (place_id,),
        ).fetchone()
        if row is None:
            state = PerceptualPlaceMemory(
                place_id=place_id,
                exemplars=(normalized,),
                observations=1,
                episodes=(episode_id,),
                last_episode=episode_id,
                last_step=step,
            )
        else:
            previous = PerceptualPlaceMemory.model_validate_json(row[0])
            if len(previous.exemplars[0]) != len(normalized):
                raise ValueError("perceptual signature width changed")
            state = PerceptualPlaceMemory(
                place_id=place_id,
                exemplars=self._coreset(
                    previous.exemplars[0], (*previous.exemplars, normalized)
                ),
                observations=previous.observations + 1,
                episodes=tuple(sorted(set(previous.episodes) | {episode_id})),
                last_episode=max(previous.last_episode, episode_id),
                last_step=(
                    max(previous.last_step, step)
                    if episode_id == previous.last_episode
                    else step
                    if episode_id > previous.last_episode
                    else previous.last_step
                ),
            )
        payload = state.model_dump(mode="json")
        with self._connection:
            self._append_event(
                "perceptual_view_recorded",
                episode_id,
                step,
                {"signature": normalized, "state": payload},
            )
            self._connection.execute(
                "INSERT INTO perceptual_places (place_id, payload_json) VALUES (?, ?) "
                "ON CONFLICT(place_id) DO UPDATE SET payload_json = excluded.payload_json",
                (place_id, _canonical(payload)),
            )
        return state

    def match_perceptual_place(
        self,
        signature: Iterable[int],
        *,
        tolerance: float,
        ambiguity_margin: float = 0,
    ) -> str | None:
        """Nearest-exemplar recognition with fail-closed ambiguity handling."""

        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("perceptual tolerance must be finite and nonnegative")
        if not math.isfinite(ambiguity_margin) or ambiguity_margin < 0:
            raise ValueError(
                "perceptual ambiguity margin must be finite and nonnegative"
            )
        normalized = _signature(signature)
        candidates = sorted(
            (
                self._nearest_distance(item.exemplars, normalized),
                item.place_id,
            )
            for item in self.perceptual_places()
            if len(item.exemplars[0]) == len(normalized)
        )
        if not candidates or candidates[0][0] > tolerance:
            return None
        if (
            len(candidates) > 1
            and candidates[1][0] - candidates[0][0] <= ambiguity_margin
        ):
            return None
        return candidates[0][1]

    def frontiers(self) -> tuple[FrontierMemory, ...]:
        cursor = self._connection.execute(
            "SELECT payload_json FROM frontiers ORDER BY frontier_id"
        )
        return tuple(FrontierMemory.model_validate_json(row[0]) for row in cursor)

    def get_frontier(self, frontier_id: str) -> FrontierMemory:
        row = self._connection.execute(
            "SELECT payload_json FROM frontiers WHERE frontier_id = ?", (frontier_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown frontier: {frontier_id}")
        return FrontierMemory.model_validate_json(row[0])

    def transitions(self) -> tuple[PlaceTransition, ...]:
        cursor = self._connection.execute(
            "SELECT payload_json FROM place_transitions "
            "ORDER BY source_place_id, target_place_id, action_roles_json"
        )
        return tuple(PlaceTransition.model_validate_json(row[0]) for row in cursor)

    @staticmethod
    def _nearest_distance(
        exemplars: tuple[tuple[int, ...], ...], signature: tuple[int, ...]
    ) -> float:
        compatible = [item for item in exemplars if len(item) == len(signature)]
        return min(
            (_distance(item, signature) for item in compatible), default=math.inf
        )

    def _match_place(self, normalized: tuple[int, ...]) -> tuple[str | None, bool]:
        candidates: list[tuple[float, str]] = []
        for place in self.places():
            distance = self._nearest_distance(place.exemplars, normalized)
            if distance <= place.match_radius:
                candidates.append((distance, place.place_id))
        candidates.sort()
        if not candidates:
            return None, False
        if (
            len(candidates) > 1
            and candidates[1][0] - candidates[0][0] <= self.ambiguity_margin
        ):
            return None, True
        return candidates[0][1], False

    def match_place(self, signature: Iterable[int]) -> str | None:
        normalized = _signature(signature)
        matched, _ambiguous = self._match_place(normalized)
        return matched

    def _coreset(
        self, anchor: tuple[int, ...], candidates: Iterable[tuple[int, ...]]
    ) -> tuple[tuple[int, ...], ...]:
        unique = tuple(sorted(set(candidates)))
        if any(len(item) != len(anchor) for item in unique):
            raise ValueError("place exemplars must have equal dimensions")
        remaining = set(unique)
        remaining.discard(anchor)
        selected = [anchor]
        while remaining and len(selected) < self.max_exemplars:
            choice = min(
                remaining,
                key=lambda item: (
                    -min(_distance(item, existing) for existing in selected),
                    item,
                ),
            )
            selected.append(choice)
            remaining.remove(choice)
        return tuple(selected)

    def observe_place(
        self,
        signature: Iterable[int],
        *,
        episode_id: int,
        step: int,
        uncertainty: float = 1.0,
        risk: float = 0.0,
    ) -> PlaceMemory:
        normalized = _signature(signature)
        if episode_id < 0 or step < 0:
            raise ValueError("episode and step must be nonnegative")
        if not 0 <= uncertainty <= 1 or not math.isfinite(uncertainty):
            raise ValueError("uncertainty must be finite and in [0, 1]")
        if risk < 0 or not math.isfinite(risk):
            raise ValueError("risk must be finite and nonnegative")
        matched, ambiguous = self._match_place(normalized)
        if ambiguous:
            raise ValueError("ambiguous place observation requires more evidence")
        if matched is None:
            place_id = self.place_id_for(normalized)
            existing = self._connection.execute(
                "SELECT 1 FROM places WHERE place_id = ?", (place_id,)
            ).fetchone()
            if existing is not None:
                raise ValueError(
                    "ambiguous observation collides with an existing place"
                )
            place = PlaceMemory(
                place_id=place_id,
                exemplars=(normalized,),
                visits=1,
                episodes=(episode_id,),
                first_episode=episode_id,
                last_episode=episode_id,
                first_step=step,
                last_step=step,
                uncertainty=uncertainty,
                risk=risk,
                match_radius=self.default_match_radius,
            )
        else:
            previous = self.get_place(matched)
            if episode_id < previous.last_episode:
                last_episode = previous.last_episode
                last_step = previous.last_step
            elif episode_id == previous.last_episode:
                last_episode = episode_id
                last_step = max(previous.last_step, step)
            else:
                last_episode = episode_id
                last_step = step
            place = previous.model_copy(
                update={
                    "exemplars": self._coreset(
                        previous.exemplars[0], (*previous.exemplars, normalized)
                    ),
                    "visits": previous.visits + 1,
                    "episodes": tuple(sorted(set(previous.episodes) | {episode_id})),
                    "last_episode": last_episode,
                    "last_step": last_step,
                    "uncertainty": uncertainty,
                    "risk": max(previous.risk, risk),
                }
            )
            place = PlaceMemory.model_validate(place.model_dump())
        payload = place.model_dump(mode="json")
        with self._connection:
            self._append_event(
                "place_observed",
                episode_id,
                step,
                {"place_id": place.place_id, "signature": normalized, "state": payload},
            )
            self._connection.execute(
                "INSERT INTO places (place_id, payload_json) VALUES (?, ?) "
                "ON CONFLICT(place_id) DO UPDATE SET payload_json = excluded.payload_json",
                (place.place_id, _canonical(payload)),
            )
        return place

    def calibrate_place(
        self,
        place_id: str,
        positive_signatures: Iterable[Iterable[int]],
        negative_signatures: Iterable[Iterable[int]],
        *,
        episode_id: int,
        step: int,
        margin: float = 1.0,
    ) -> PlaceMemory:
        if episode_id < 0 or step < 0:
            raise ValueError("episode and step must be nonnegative")
        if margin < 0 or not math.isfinite(margin):
            raise ValueError("calibration margin must be finite and nonnegative")
        previous = self.get_place(place_id)
        positives = tuple(_signature(item) for item in positive_signatures)
        negatives = tuple(_signature(item) for item in negative_signatures)
        if len(positives) < 2:
            raise ValueError("place calibration requires at least two positives")
        if not negatives:
            raise ValueError("place calibration requires at least one negative")
        exemplars = self._coreset(
            previous.exemplars[0], (*previous.exemplars, *positives)
        )
        positive_distance = max(
            self._nearest_distance(exemplars, item) for item in positives
        )
        negative_distance = min(
            self._nearest_distance(exemplars, item) for item in negatives
        )
        calibration = PlaceCalibration(
            positive_samples=len(positives),
            negative_samples=len(negatives),
            maximum_positive_distance=positive_distance,
            nearest_negative_distance=negative_distance,
            margin=margin,
            radius=positive_distance + margin,
        )
        updated = previous.model_copy(
            update={
                "exemplars": exemplars,
                "match_radius": calibration.radius,
                "calibration": calibration,
            }
        )
        updated = PlaceMemory.model_validate(updated.model_dump())
        payload = updated.model_dump(mode="json")
        with self._connection:
            self._append_event(
                "place_calibrated",
                episode_id,
                step,
                {
                    "place_id": place_id,
                    "positive_signatures": positives,
                    "negative_signatures": negatives,
                    "state": payload,
                },
            )
            self._connection.execute(
                "UPDATE places SET payload_json = ? WHERE place_id = ?",
                (_canonical(payload), place_id),
            )
        return updated

    def register_frontier(
        self,
        source_place_id: str,
        action_roles: Iterable[str],
        *,
        episode_id: int,
        step: int,
        expected_cost: float = 1.0,
        uncertainty: float = 1.0,
        risk: float = 0.0,
    ) -> FrontierMemory:
        self.get_place(source_place_id)
        roles = tuple(str(item) for item in action_roles)
        frontier_id = self.frontier_id_for(source_place_id, roles)
        row = self._connection.execute(
            "SELECT payload_json FROM frontiers WHERE frontier_id = ?", (frontier_id,)
        ).fetchone()
        if row is not None:
            return FrontierMemory.model_validate_json(row[0])
        frontier = FrontierMemory(
            frontier_id=frontier_id,
            source_place_id=source_place_id,
            action_roles=roles,
            expected_cost=expected_cost,
            uncertainty=uncertainty,
            risk=risk,
        )
        payload = frontier.model_dump(mode="json")
        with self._connection:
            self._append_event(
                "frontier_registered",
                episode_id,
                step,
                {"state": payload},
            )
            self._connection.execute(
                "INSERT INTO frontiers (frontier_id, payload_json) VALUES (?, ?)",
                (frontier_id, _canonical(payload)),
            )
        return frontier

    def record_frontier_outcome(
        self,
        frontier_id: str,
        *,
        episode_id: int,
        step: int,
        success: bool,
        cost: float,
        discovered_place_id: str | None = None,
    ) -> FrontierMemory:
        if cost <= 0 or not math.isfinite(cost):
            raise ValueError("frontier outcome cost must be finite and positive")
        previous = self.get_frontier(frontier_id)
        if previous.status is not FrontierStatus.ACTIVE:
            raise ValueError("only an active frontier can record an outcome")
        if success:
            if discovered_place_id is None:
                raise ValueError("successful frontier requires a discovered place")
            self.get_place(discovered_place_id)
        elif discovered_place_id is not None:
            raise ValueError("failed frontier cannot name a discovered place")
        attempts = previous.attempts + 1
        failures = attempts - (previous.successes + int(success))
        failure_streak = 0 if success else previous.failure_streak + 1
        status = (
            FrontierStatus.RESOLVED
            if success
            else FrontierStatus.BLOCKED
            if failure_streak >= self.block_after_failures
            else FrontierStatus.ACTIVE
        )
        updated = previous.model_copy(
            update={
                "attempts": attempts,
                "successes": previous.successes + int(success),
                "failure_streak": failure_streak,
                "status": status,
                "discovered_place_id": discovered_place_id,
                "last_episode": episode_id,
                "last_step": step,
            }
        )
        updated = FrontierMemory.model_validate(updated.model_dump())
        with self._connection:
            self._append_event(
                "frontier_outcome",
                episode_id,
                step,
                {
                    "success": success,
                    "cost": cost,
                    "state": updated.model_dump(mode="json"),
                },
            )
            self._connection.execute(
                "UPDATE frontiers SET payload_json = ? WHERE frontier_id = ?",
                (_canonical(updated.model_dump(mode="json")), frontier_id),
            )
            if success:
                assert discovered_place_id is not None
                self._record_transition(
                    previous.source_place_id,
                    discovered_place_id,
                    previous.action_roles,
                    success=True,
                    cost=cost,
                )
            elif failures:
                # Failed unknown actions have no target and therefore no graph edge.
                pass
        return updated

    def _record_transition(
        self,
        source_place_id: str,
        target_place_id: str,
        roles: tuple[str, ...],
        *,
        success: bool,
        cost: float,
    ) -> None:
        roles_json = _canonical(roles)
        row = self._connection.execute(
            "SELECT payload_json FROM place_transitions WHERE "
            "source_place_id = ? AND target_place_id = ? AND action_roles_json = ?",
            (source_place_id, target_place_id, roles_json),
        ).fetchone()
        if row is None:
            if not success:
                return
            transition = PlaceTransition(
                source_place_id=source_place_id,
                target_place_id=target_place_id,
                action_roles=roles,
                successes=1,
                failures=0,
                mean_cost=cost,
            )
        else:
            previous = PlaceTransition.model_validate_json(row[0])
            trials = previous.successes + previous.failures + 1
            transition = previous.model_copy(
                update={
                    "successes": previous.successes + int(success),
                    "failures": previous.failures + int(not success),
                    "mean_cost": (previous.mean_cost * (trials - 1) + cost) / trials,
                }
            )
            transition = PlaceTransition.model_validate(transition.model_dump())
        self._connection.execute(
            "INSERT INTO place_transitions "
            "(source_place_id, target_place_id, action_roles_json, payload_json) "
            "VALUES (?, ?, ?, ?) ON CONFLICT DO UPDATE SET payload_json = excluded.payload_json",
            (
                source_place_id,
                target_place_id,
                roles_json,
                _canonical(transition.model_dump(mode="json")),
            ),
        )

    def plan_next_frontier(self, current_place_id: str) -> FrontierPlan | None:
        self.get_place(current_place_id)
        graph = nx.DiGraph()
        graph.add_nodes_from(place.place_id for place in self.places())
        for transition in self.transitions():
            weight = transition.mean_cost / transition.confidence
            existing = graph.get_edge_data(
                transition.source_place_id, transition.target_place_id
            )
            if existing is None or weight < existing["weight"]:
                graph.add_edge(
                    transition.source_place_id,
                    transition.target_place_id,
                    weight=weight,
                )
        plans: list[FrontierPlan] = []
        for frontier in self.frontiers():
            if frontier.status is not FrontierStatus.ACTIVE:
                continue
            try:
                path = nx.shortest_path(
                    graph,
                    current_place_id,
                    frontier.source_place_id,
                    weight="weight",
                )
                travel = nx.path_weight(graph, path, weight="weight")
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                continue
            failures = frontier.attempts - frontier.successes
            failure_probability = (failures + 1) / (frontier.attempts + 2)
            novelty = 1 / (frontier.attempts + 1)
            utility = (
                2.0 * frontier.uncertainty
                + novelty
                - 0.25 * (travel + frontier.expected_cost)
                - frontier.risk
                - failure_probability
            )
            plans.append(
                FrontierPlan(
                    place_path=tuple(path),
                    frontier_id=frontier.frontier_id,
                    action_roles=frontier.action_roles,
                    travel_cost=travel,
                    frontier_cost=frontier.expected_cost,
                    utility=utility,
                    uncertainty_gain=frontier.uncertainty,
                    failure_probability=failure_probability,
                )
            )
        return max(
            plans, key=lambda item: (item.utility, item.frontier_id), default=None
        )

    def _append_event(
        self,
        event_type: str,
        episode_id: int,
        step: int,
        payload: dict[str, Any],
    ) -> None:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM exploration_events"
        ).fetchone()
        assert row is not None
        sequence = int(row[0])
        body = {
            "sequence": sequence,
            "episode_id": episode_id,
            "step": step,
            "event_type": event_type,
            "payload": payload,
        }
        event_hash = _digest(body)
        event_id = f"explore_{sequence:08d}_{event_hash[:16]}"
        self._connection.execute(
            "INSERT INTO exploration_events "
            "(event_id, episode_id, step, event_type, payload_json, payload_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, episode_id, step, event_type, _canonical(payload), event_hash),
        )

    def iter_events(self) -> Iterator[dict[str, Any]]:
        cursor = self._connection.execute(
            "SELECT sequence, event_id, episode_id, step, event_type, payload_json, "
            "payload_sha256 FROM exploration_events ORDER BY sequence"
        )
        for sequence, event_id, episode, step, event_type, payload, digest in cursor:
            yield {
                "sequence": int(sequence),
                "event_id": event_id,
                "episode_id": int(episode),
                "step": int(step),
                "event_type": event_type,
                "payload": json.loads(payload),
                "payload_sha256": digest,
            }

    def verify_integrity(self) -> bool:
        configuration = _canonical(self._configuration())
        metadata = dict(
            self._connection.execute("SELECT key, value FROM metadata ORDER BY key")
        )
        if metadata != {
            "configuration": configuration,
            "configuration_sha256": hashlib.sha256(
                configuration.encode("utf-8")
            ).hexdigest(),
        }:
            return False
        expected_sequence = 1
        replayed_places: dict[str, dict[str, Any]] = {}
        replayed_perceptual_places: dict[str, dict[str, Any]] = {}
        replayed_frontiers: dict[str, dict[str, Any]] = {}
        replayed_transitions: dict[tuple[str, str, str], dict[str, Any]] = {}
        for event in self.iter_events():
            if event["sequence"] != expected_sequence:
                return False
            body = {
                key: event[key]
                for key in ("sequence", "episode_id", "step", "event_type", "payload")
            }
            if _digest(body) != event["payload_sha256"]:
                return False
            expected_id = (
                f"explore_{expected_sequence:08d}_{event['payload_sha256'][:16]}"
            )
            if event["event_id"] != expected_id:
                return False
            event_type = event["event_type"]
            state = event["payload"].get("state")
            if event_type in {"place_observed", "place_calibrated"}:
                if not isinstance(state, dict):
                    return False
                try:
                    place = PlaceMemory.model_validate(state)
                except ValueError:
                    return False
                replayed_places[place.place_id] = state
            elif event_type == "perceptual_view_recorded":
                if not isinstance(state, dict):
                    return False
                try:
                    perceptual = PerceptualPlaceMemory.model_validate(state)
                except ValueError:
                    return False
                replayed_perceptual_places[perceptual.place_id] = state
            elif event_type in {"frontier_registered", "frontier_outcome"}:
                if not isinstance(state, dict):
                    return False
                try:
                    frontier = FrontierMemory.model_validate(state)
                except ValueError:
                    return False
                replayed_frontiers[frontier.frontier_id] = state
                if (
                    event_type == "frontier_outcome"
                    and event["payload"].get("success") is True
                ):
                    target = frontier.discovered_place_id
                    cost = event["payload"].get("cost")
                    if target is None or not isinstance(cost, (int, float)):
                        return False
                    transition = PlaceTransition(
                        source_place_id=frontier.source_place_id,
                        target_place_id=target,
                        action_roles=frontier.action_roles,
                        successes=1,
                        failures=0,
                        mean_cost=float(cost),
                    )
                    key = (
                        transition.source_place_id,
                        transition.target_place_id,
                        _canonical(transition.action_roles),
                    )
                    replayed_transitions[key] = transition.model_dump(mode="json")
            else:
                return False
            expected_sequence += 1
        try:
            places = self.places()
            perceptual_places = self.perceptual_places()
            frontiers = self.frontiers()
            transitions = self.transitions()
        except ValueError:
            return False
        stored_places = {item.place_id: item.model_dump(mode="json") for item in places}
        stored_perceptual_places = {
            item.place_id: item.model_dump(mode="json") for item in perceptual_places
        }
        stored_frontiers = {
            item.frontier_id: item.model_dump(mode="json") for item in frontiers
        }
        if (
            replayed_places != stored_places
            or replayed_perceptual_places != stored_perceptual_places
            or replayed_frontiers != stored_frontiers
        ):
            return False
        stored_transitions = {
            (
                item.source_place_id,
                item.target_place_id,
                _canonical(item.action_roles),
            ): item.model_dump(mode="json")
            for item in transitions
        }
        if replayed_transitions != stored_transitions:
            return False
        row = self._connection.execute("PRAGMA integrity_check").fetchone()
        return row == ("ok",)

    def summary(self) -> dict[str, Any]:
        places = self.places()
        perceptual_places = self.perceptual_places()
        frontiers = self.frontiers()
        transitions = self.transitions()
        return {
            "format_version": self.FORMAT_VERSION,
            "place_count": len(places),
            "multi_view_place_count": sum(len(item.exemplars) > 1 for item in places),
            "calibrated_place_count": sum(
                item.calibration is not None for item in places
            ),
            "perceptual_place_count": len(perceptual_places),
            "multi_view_perceptual_place_count": sum(
                len(item.exemplars) > 1 for item in perceptual_places
            ),
            "frontier_count": len(frontiers),
            "active_frontier_count": sum(
                item.status is FrontierStatus.ACTIVE for item in frontiers
            ),
            "resolved_frontier_count": sum(
                item.status is FrontierStatus.RESOLVED for item in frontiers
            ),
            "blocked_frontier_count": sum(
                item.status is FrontierStatus.BLOCKED for item in frontiers
            ),
            "transition_count": len(transitions),
            "episode_count": len(
                {episode for item in places for episode in item.episodes}
            ),
            "event_count": sum(1 for _ in self.iter_events()),
            "integrity_valid": self.verify_integrity(),
        }

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
