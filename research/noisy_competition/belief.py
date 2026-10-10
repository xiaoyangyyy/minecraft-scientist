"""Exact Bayesian model/state competition; conditioning is not causal proof."""

import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction

from minecraft_scientist.protocols import TheoryIR
from research.stochastic_ir.composition import compile_joint
from research.stochastic_ir.program import StochasticProgram

MAX_RATIONAL_BITS = 12_000
MAX_OBSERVATIONS = 256
MAX_KERNEL_CALLS = 200_000
MAX_CACHE_ENTRIES = 16_384


class ModelMismatch(ValueError):
    """No supplied model assigns positive probability to the acquired evidence."""


class WorkBound(ValueError):
    """Declared computation/representation limit; no partial probability update."""


@dataclass(frozen=True)
class Candidate:
    key: str
    mechanism_id: str
    controller_id: str
    sensor_id: str
    mechanism_ir: TheoryIR
    controller_ir: TheoryIR
    sensor_ir: TheoryIR
    idle_output: str
    prior: Fraction = Fraction(1)


def rational(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _bounded(value):
    if (
        max(value.numerator.bit_length(), value.denominator.bit_length())
        > MAX_RATIONAL_BITS
    ):
        raise WorkBound("exact rational representation bound; never round a posterior")
    return value


class ModelBank:
    def __init__(self, candidates):
        if type(candidates) is not tuple or not 1 <= len(candidates) <= 32:
            raise ValueError("one to 32 declared candidates required")
        definitions, keys, snapshots, programs = {}, set(), [], []
        for item in candidates:
            if (
                type(item) is not Candidate
                or any(
                    type(v) is not str or not 1 <= len(v) <= 80
                    for v in (
                        item.key,
                        item.mechanism_id,
                        item.controller_id,
                        item.sensor_id,
                    )
                )
                or item.key in keys
            ):
                raise ValueError(
                    "candidates need unique bounded keys and factor identifiers"
                )
            if (
                type(item.prior) is not Fraction
                or item.prior <= 0
                or item.prior.denominator > 1_000_000
                or item.prior.numerator.bit_length() > 64
            ):
                raise ValueError("strict positive bounded rational priors required")
            keys.add(item.key)
            if item.idle_output != candidates[0].idle_output:
                raise ValueError("shared nondelivery output contract required")
            factors = []
            for kind, identifier, ir in (
                ("mechanism", item.mechanism_id, item.mechanism_ir),
                ("controller", item.controller_id, item.controller_ir),
                ("sensor", item.sensor_id, item.sensor_ir),
            ):
                if not isinstance(ir, TheoryIR):
                    raise TypeError("factor definitions must use shared TheoryIR")
                copied = TheoryIR.model_validate(ir.model_dump(mode="json"))
                payload = json.dumps(
                    copied.model_dump(mode="json"), sort_keys=True, allow_nan=False
                )
                if (kind, identifier) in definitions and definitions[
                    kind, identifier
                ] != payload:
                    raise ValueError("a factor identifier cannot redefine its theory")
                definitions[kind, identifier] = payload
                factors.append(copied)
            compiled = compile_joint(*factors, idle_output=item.idle_output)
            programs.append(StochasticProgram.from_ir(compiled))
            snapshots.append(
                {
                    "key": item.key,
                    "mechanism_id": item.mechanism_id,
                    "controller_id": item.controller_id,
                    "sensor_id": item.sensor_id,
                    "factor_ir": [v.model_dump(mode="json") for v in factors],
                    "idle_output": item.idle_output,
                    "prior": rational(item.prior),
                    "joint_ir": compiled.model_dump(mode="json"),
                }
            )
        self._definitions, self.programs = snapshots, tuple(programs)
        self.actions, self.outputs = self.programs[0].actions, self.programs[0].outputs
        if any(
            p.actions != self.actions or p.outputs != self.outputs
            for p in self.programs
        ):
            raise ValueError("shared declared action/output order is required")
        normalizer = sum((c.prior for c in candidates), Fraction(0))
        self._weights = tuple(_bounded(c.prior / normalizer) for c in candidates)
        self._beliefs = tuple(p.initial_belief() for p in self.programs)
        self._cache, self.kernel_calls, self.observations, self.resets = {}, 0, 0, 0

    def definitions(self):
        return deepcopy(self._definitions)

    def snapshot(self):
        return {
            "weights": [rational(p) for p in self._weights],
            "state_beliefs": [
                {s: rational(p) for s, p in sorted(v.items())} for v in self._beliefs
            ],
            "observations": self.observations,
            "resets": self.resets,
            "kernel_calls": self.kernel_calls,
            "scope": "conditional on declared finite factors/prior; no unique physical attribution or calibrated frequentist confidence",
        }

    def reset_state(self):
        # A physical reset restarts latent state, not acquired model evidence.
        self._beliefs = tuple(p.initial_belief() for p in self.programs)
        self.resets += 1

    def project(self, weights=None, *, factor="mechanism"):
        if factor not in ("mechanism", "controller", "sensor"):
            raise ValueError("unknown factor")
        values = self._weights if weights is None else weights
        if (
            type(values) is not tuple
            or len(values) != len(self.programs)
            or any(type(v) is not Fraction or v < 0 for v in values)
            or sum(values, Fraction(0)) != 1
        ):
            raise ValueError("projection requires complete unit model mass")
        output = defaultdict(Fraction)
        for item, mass in zip(self._definitions, values, strict=True):
            output[item[f"{factor}_id"]] += mass
        return dict(output)

    def predict_step(self, action):
        if type(action) is not str or action not in self.actions:
            raise ValueError("unknown action")
        joint = tuple(
            p.predict_step(action, belief=b) if weight else {}
            for p, b, weight in zip(
                self.programs, self._beliefs, self._weights, strict=True
            )
        )
        mixture = defaultdict(Fraction)
        for weight, dist in zip(self._weights, joint, strict=True):
            for output, states in dist.items():
                mixture[output] += weight * sum(states.values(), Fraction(0))
        return joint, dict(mixture)

    def advance(self, action, observed):
        if type(observed) is not str or observed not in self.outputs:
            raise ValueError("output is outside the declared observation alphabet")
        if self.observations >= MAX_OBSERVATIONS:
            raise WorkBound("observation limit; never overwrite an evidence epoch")
        joint, mixture = self.predict_step(action)
        likelihood = mixture.get(observed, Fraction(0))
        if not likelihood:
            raise ModelMismatch("all declared models have zero likelihood")
        following, weights = [], []
        for prior, distribution in zip(self._weights, joint, strict=True):
            states = distribution.get(observed, {})
            conditional = sum(states.values(), Fraction(0))
            weights.append(_bounded(prior * conditional / likelihood))
            following.append(
                {s: _bounded(mass / conditional) for s, mass in states.items()}
                if conditional
                else {}
            )
        # Commit atomically, only after all exact bounds pass.
        self._weights, self._beliefs = tuple(weights), tuple(following)
        self.observations += 1
        return likelihood

    def forecast(self, word, *, reset):
        if (
            type(reset) is not bool
            or type(word) is not tuple
            or not word
            or len(word) > 4
            or any(type(a) is not str or a not in self.actions for a in word)
        ):
            raise ValueError("forecast requires reset bool and 1..4 declared actions")
        distributions = []
        for index, (program, weight, belief) in enumerate(
            zip(self.programs, self._weights, self._beliefs, strict=True)
        ):
            if not weight:
                distributions.append({})
                continue
            selected = program.initial_belief() if reset else belief
            initial = selected == program.initial_belief()
            key = index, word
            if initial and key in self._cache:
                distribution = dict(self._cache[key])
            else:
                if self.kernel_calls >= MAX_KERNEL_CALLS:
                    raise WorkBound(
                        "prediction kernel-call limit; no candidate filtering"
                    )
                distribution = program.sequence_distribution(word, belief=selected)
                self.kernel_calls += 1
                if initial:
                    if len(self._cache) >= MAX_CACHE_ENTRIES:
                        raise WorkBound("reset prediction cache bound; no truncation")
                    self._cache[key] = tuple(distribution.items())
            distributions.append(distribution)
        mixture = defaultdict(Fraction)
        for weight, distribution in zip(self._weights, distributions, strict=True):
            for observed, probability in distribution.items():
                mixture[observed] += weight * probability
        if sum(mixture.values(), Fraction(0)) != 1:
            raise ArithmeticError("forecast lost model/output probability mass")
        return self._weights, tuple(distributions), dict(mixture)
