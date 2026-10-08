"""Semantic version spaces and bounded, public-model-only query selection.

This first H1 slice assumes known deterministic candidate programs and resets.
It is not structure synthesis, noise inference, fault diagnosis or an innovation
claim. Exact multi-step information gain is retained as a matched strong baseline.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter

from minecraft_scientist.protocols import TheoryIR
from minecraft_scientist.theory.finite_state import (
    FiniteStateProgram,
    bounded_behavior_classes,
    shortest_witness,
    validate_costs,
)

METHODS = (
    "random",
    "informative_random",
    "one_step_ig",
    "sequence_ig",
    "witness_cover",
    "optimal_reference",
)


class DiscriminationSpace:
    def __init__(
        self,
        candidates: tuple[TheoryIR, ...],
        costs: dict[str, int],
        *,
        horizon: int = 4,
        reset_cost: int = 1,
        candidate_limit: int = 12,
        equivalence_horizon: int | None = None,
    ):
        if (
            type(candidate_limit) is not int
            or not 1 <= candidate_limit <= 512
            or not 1 <= len(candidates) <= candidate_limit
        ):
            raise ValueError(
                "candidate count exceeds the declared capacity (maximum 512)"
            )
        self.programs = tuple(FiniteStateProgram.from_ir(ir) for ir in candidates)
        self.actions = tuple(sorted(self.programs[0].actions))
        validate_costs(self.actions, costs)
        if any(
            set(program.actions) != set(self.actions)
            or set(program.outputs) != set(self.programs[0].outputs)
            for program in self.programs
        ):
            raise ValueError("all candidates must have identical public alphabets")
        if type(horizon) is not int or not 1 <= horizon <= 8:
            raise ValueError("horizon must be an exact integer in [1,8]")
        if sum(len(self.actions) ** depth for depth in range(1, horizon + 1)) > 50_000:
            raise ValueError("declared query universe exceeds the computation cap")
        if type(reset_cost) is not int or not 1 <= reset_cost <= 100:
            raise ValueError("reset has an exact positive integer cost")
        self.costs, self.horizon, self.reset_cost = dict(costs), horizon, reset_cost
        self.remaining = tuple(range(len(self.programs)))
        self.tested: set[tuple[str, ...]] = set()
        self.prediction_calls = 0
        self._predictions = {}
        self._optimal_cache = {}
        self._optimal_evaluations = 0
        if equivalence_horizon is not None:
            if type(equivalence_horizon) is not int or equivalence_horizon != horizon:
                raise ValueError(
                    "equivalence horizon must equal the query-language horizon"
                )
            self.classes = bounded_behavior_classes(self.programs, equivalence_horizon)
            return
        classes: list[list[int]] = []
        for index, program in enumerate(self.programs):
            for group in classes:
                if (
                    shortest_witness(program, self.programs[group[0]], self.costs)
                    is None
                ):
                    group.append(index)
                    break
            else:
                classes.append([index])
        self.classes = tuple(tuple(group) for group in classes)

    def representatives(self, remaining=None):
        active = set(self.remaining if remaining is None else remaining)
        return tuple(
            next(index for index in group if index in active)
            for group in self.classes
            if active.intersection(group)
        )

    def cost(self, word):
        if (
            type(word) is not tuple
            or not 1 <= len(word) <= self.horizon
            or any(
                type(action) is not str or action not in self.actions for action in word
            )
        ):
            raise ValueError("query is outside the declared intervention language")
        return self.reset_cost + sum(self.costs[action] for action in word)

    def prediction(self, index, word):
        key = (index, word)
        if key not in self._predictions:
            self.prediction_calls += 1
            if len(self._predictions) >= 100_000:
                self._predictions.clear()
            self._predictions[key] = self.programs[index].execute(word)[0]
        return self._predictions[key]

    def observe(self, word, outputs):
        self.cost(word)
        if type(outputs) is not tuple or len(outputs) != len(word):
            raise ValueError(
                "exactly one public output per executed action is required"
            )
        if any(
            type(output) is not str or output not in self.programs[0].outputs
            for output in outputs
        ):
            raise ValueError("observed output is outside the public alphabet")
        self.tested.add(word)
        self.remaining = tuple(
            index for index in self.remaining if self.prediction(index, word) == outputs
        )

    def queries(self):
        for depth in range(1, self.horizon + 1):
            yield from itertools.product(self.actions, repeat=depth)

    def groups(self, word, indices):
        groups = {}
        for index in indices:
            groups.setdefault(self.prediction(index, word), []).append(index)
        return tuple(tuple(group) for group in groups.values())

    def entropy(self, word, indices):
        counts = Counter(self.prediction(index, word) for index in indices)
        return -sum(
            (count / len(indices)) * math.log2(count / len(indices))
            for count in counts.values()
        )

    def choose(self, method: str, remaining_budget: int, rng: random.Random):
        if method not in METHODS or type(remaining_budget) is not int:
            raise ValueError("unknown method or non-integer budget")
        indices = self.representatives()
        if len(indices) < 2:
            return None
        if method == "optimal_reference":
            return (
                self.optimal_policy(indices)[1]
                if (
                    self.optimal_policy(indices)[1] is not None
                    and self.cost(self.optimal_policy(indices)[1]) <= remaining_budget
                )
                else None
            )
        if method == "witness_cover":
            words = {
                witness.actions
                for left, right in itertools.combinations(indices, 2)
                if (
                    witness := shortest_witness(
                        self.programs[left],
                        self.programs[right],
                        self.costs,
                        maximum_length=self.horizon,
                    )
                )
                is not None
            }
        elif method == "one_step_ig":
            words = ((action,) for action in self.actions)
        else:
            words = self.queries()
        options = [
            word
            for word in words
            if word not in self.tested and self.cost(word) <= remaining_budget
        ]
        if method == "random":
            return rng.choice(options) if options else None
        options = [word for word in options if self.entropy(word, indices) > 0]
        if method == "informative_random":
            return rng.choice(options) if options else None
        if not options:
            return None
        if method == "witness_cover":
            # Expected eliminated ordered pairs / cost, uniform BEHAVIOR-CLASS prior.
            def score(word):
                counts = [len(group) for group in self.groups(word, indices)]
                return (len(indices) ** 2 - sum(n * n for n in counts)) / self.cost(
                    word
                )
        else:

            def score(word):
                return self.entropy(word, indices) / self.cost(word)

        return min(options, key=lambda word: (-score(word), self.cost(word), word))

    def optimal_policy(self, indices):
        """Exact expected-cost decision tree over U, without knowing true index.

        Uniform prior over remaining exact behavior classes. No finite-budget
        success maximization is claimed; an unaffordable root simply stops.
        """
        if len(indices) <= 1:
            return 0.0, None
        if indices in self._optimal_cache:
            return self._optimal_cache[indices]
        best = (math.inf, None)
        for word in self.queries():
            self._optimal_evaluations += 1
            if self._optimal_evaluations > 200_000:
                raise RuntimeError("optimal reference computation budget exhausted")
            groups = self.groups(word, indices)
            if len(groups) <= 1:
                continue
            expected = self.cost(word) + sum(
                len(group) / len(indices) * self.optimal_policy(group)[0]
                for group in groups
            )
            if expected < best[0] or (
                expected == best[0]
                and (
                    best[1] is None
                    or (self.cost(word), word) < (self.cost(best[1]), best[1])
                )
            ):
                best = expected, word
        self._optimal_cache[indices] = best
        return best
