"""Finite MDL-oriented learner for a piecewise continuous fall-damage law."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass

from minecraft_scientist.planning.selection import select_best_experiment
from minecraft_scientist.protocols import TheoryIR, TheoryRepresentation


@dataclass(frozen=True, slots=True)
class FallEquationHypothesis:
    safe_height: int
    armor_slopes: dict[str, float]

    def predict(self, armor_id: str, height: float) -> float:
        return self.armor_slopes[armor_id] * max(0.0, height - self.safe_height)


class FallEquationVersionSpace:
    """Active version space over hinge equations and latent armor coefficients."""

    def __init__(
        self,
        armor_ids: tuple[str, ...],
        heights: tuple[int, ...],
        *,
        safe_height_candidates: tuple[int, ...] = (2, 3, 4),
        slope_candidates: tuple[float, ...] = (0.2, 0.4, 0.7, 1.0, 1.3),
    ) -> None:
        self.armor_ids = tuple(sorted(armor_ids))
        self.heights = tuple(sorted(heights))
        if len(self.armor_ids) > len(slope_candidates):
            raise ValueError("there must be at least one slope per armor ID")
        self.candidates = [
            FallEquationHypothesis(
                safe_height=safe_height,
                armor_slopes=dict(zip(self.armor_ids, slopes, strict=True)),
            )
            for safe_height in safe_height_candidates
            for slopes in itertools.permutations(slope_candidates, len(self.armor_ids))
        ]
        self._initial_candidate_count = len(self.candidates)
        self.observations: list[tuple[str, int, float, str]] = []
        self.control_failures: dict[tuple[str, int], int] = {}

    def record_control_failure(self, armor_id: str, height: int) -> None:
        pair = (armor_id, height)
        self.control_failures[pair] = self.control_failures.get(pair, 0) + 1

    def observe(
        self, armor_id: str, height: int, damage: float, evidence_id: str
    ) -> None:
        self.observations.append((armor_id, height, damage, evidence_id))
        prior = self.candidates
        self.candidates = [
            candidate
            for candidate in prior
            if math.isclose(candidate.predict(armor_id, height), damage, abs_tol=1e-9)
        ]
        if not self.candidates:
            predicted = sorted(
                {round(candidate.predict(armor_id, height), 9) for candidate in prior}
            )
            raise RuntimeError(
                "fall equation version space collapsed: "
                f"armor={armor_id}, height={height}, observed={damage}, "
                f"prior_predictions={predicted}, observations={self.observations}"
            )

    def next_experiment(self) -> tuple[str, int] | None:
        pairs, features = self.experiment_features()
        return select_best_experiment(pairs, features)

    def experiment_features(
        self,
    ) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[float, ...], ...]]:
        tested = {(armor_id, height) for armor_id, height, _, _ in self.observations}
        pairs: list[tuple[str, int]] = []
        features: list[tuple[float, ...]] = []
        entropy_scale = max(1.0, math.log2(len(self.candidates)))
        posterior_fraction = len(self.candidates) / self._initial_candidate_count
        for armor_id in self.armor_ids:
            for height in self.heights:
                counts: dict[str, int] = {}
                for candidate in self.candidates:
                    value = f"{candidate.predict(armor_id, height):.9f}"
                    counts[value] = counts.get(value, 0) + 1
                entropy = -sum(
                    (count / len(self.candidates))
                    * math.log2(count / len(self.candidates))
                    for count in counts.values()
                )
                pairs.append((armor_id, height))
                features.append(
                    (
                        max(counts.values()) / len(self.candidates),
                        entropy / entropy_scale,
                        float((armor_id, height) in tested),
                        posterior_fraction,
                        height / max(self.heights),
                        self.control_failures.get((armor_id, height), 0)
                        / (self.control_failures.get((armor_id, height), 0) + 1),
                    )
                )
        return tuple(pairs), tuple(features)

    def resolved(self) -> bool:
        _, features = self.experiment_features()
        return all(
            tested == 1.0 or entropy == 0.0
            for _, entropy, tested, *_ in features
        )

    def best(self) -> FallEquationHypothesis:
        return min(
            self.candidates,
            key=lambda candidate: json.dumps(
                {
                    "safe_height": candidate.safe_height,
                    "armor_slopes": candidate.armor_slopes,
                },
                sort_keys=True,
            ),
        )

    def to_ir(self) -> TheoryIR:
        hypothesis = self.best()
        return TheoryIR(
            mechanism="fall_damage",
            representation=TheoryRepresentation.EQUATION,
            scope={
                "armor_ids": list(self.armor_ids),
                "height_range": list(self.heights),
            },
            preconditions=("height >= 0", "armor_id in scope"),
            latent_variables=("safe_height", "armor_damage_slope"),
            effects={"damage": "armor_slope(armor) * max(0, height - safe_height)"},
            parameters={
                "safe_height": hypothesis.safe_height,
                "armor_slopes": hypothesis.armor_slopes,
            },
        )
