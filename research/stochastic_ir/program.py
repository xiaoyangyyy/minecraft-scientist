"""Exact bounded stochastic transducers and non-reset public-history filtering.

This is standard finite-state probability propagation, not a novelty claim.
Latent delivery and sensor noise are marginalized, never obtained from the world.
"""

from collections import defaultdict
from fractions import Fraction
from functools import cached_property
from math import gcd
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from minecraft_scientist.protocols import TheoryIR, TheoryRepresentation

MAX_BRANCH_OPERATIONS = 100_000
MAX_OUTPUT_PREFIXES = 4096
LANGUAGE = "rational-stochastic-mealy-v1"
SCOPE = {
    "language": LANGUAGE,
    "reset": "initial-distribution",
    "conditioning": "public-history",
}


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Mass(Frozen):
    numerator: StrictInt = Field(ge=1, le=1_000_000)
    denominator: StrictInt = Field(ge=1, le=1_000_000)

    @model_validator(mode="after")
    def canonical(self):
        if (
            self.numerator > self.denominator
            or gcd(self.numerator, self.denominator) != 1
        ):
            raise ValueError("positive mass must be <=1 and in reduced rational form")
        return self

    def fraction(self):
        return Fraction(self.numerator, self.denominator)

    @classmethod
    def of(cls, value):
        if type(value) is not Fraction:
            raise ValueError("mass requires an exact Fraction")
        return cls(numerator=value.numerator, denominator=value.denominator)


class InitialMass(Frozen):
    state: StrictStr
    probability: Mass


class Branch(Frozen):
    source: StrictStr
    action: StrictStr
    target: StrictStr
    output: StrictStr
    probability: Mass


class StochasticProgram(Frozen):
    language: Literal["rational-stochastic-mealy-v1"] = LANGUAGE
    states: tuple[StrictStr, ...] = Field(min_length=1, max_length=16)
    actions: tuple[StrictStr, ...] = Field(min_length=1, max_length=8)
    outputs: tuple[StrictStr, ...] = Field(min_length=1, max_length=8)
    initial: tuple[InitialMass, ...] = Field(min_length=1, max_length=16)
    transitions: tuple[Branch, ...] = Field(min_length=1, max_length=2048)

    @model_validator(mode="after")
    def total_contract(self):
        for values in (self.states, self.actions, self.outputs):
            if len(set(values)) != len(values) or any(
                not v or len(v) > 80 for v in values
            ):
                raise ValueError("alphabets must have bounded unique nonempty labels")
        if len({v.state for v in self.initial}) != len(self.initial) or any(
            v.state not in self.states for v in self.initial
        ):
            raise ValueError("initial distribution references duplicate/unknown states")
        if sum((v.probability.fraction() for v in self.initial), Fraction(0)) != 1:
            raise ValueError("initial distribution must sum exactly to one")
        keys, totals = set(), defaultdict(Fraction)
        for row in self.transitions:
            if (
                row.source not in self.states
                or row.target not in self.states
                or row.action not in self.actions
                or row.output not in self.outputs
            ):
                raise ValueError("branch references an unknown symbol")
            key = row.source, row.action, row.target, row.output
            if key in keys:
                raise ValueError(
                    "duplicate branches must be combined, not counted twice"
                )
            keys.add(key)
            totals[row.source, row.action] += row.probability.fraction()
        expected = {(s, a) for s in self.states for a in self.actions}
        if set(totals) != expected or any(v != 1 for v in totals.values()):
            raise ValueError("every state/action needs exact unit transition mass")
        return self

    @cached_property
    def table(self):
        rows = defaultdict(list)
        for row in self.transitions:
            rows[row.source, row.action].append(row)
        return MappingProxyType(
            {
                key: tuple(sorted(value, key=lambda b: (b.target, b.output)))
                for key, value in rows.items()
            }
        )

    def initial_belief(self):
        return {entry.state: entry.probability.fraction() for entry in self.initial}

    def belief(self, value):
        if value is None:
            return self.initial_belief()
        if (
            type(value) is not dict
            or not value
            or not set(value) <= set(self.states)
            or any(type(v) is not Fraction or v < 0 for v in value.values())
            or sum(value.values(), Fraction(0)) != 1
        ):
            raise ValueError(
                "belief requires known states and exact nonnegative unit mass"
            )
        return {s: v for s, v in value.items() if v}

    def predict_step(self, action, *, belief=None):
        if type(action) is not str or action not in self.actions:
            raise ValueError("unknown public action")
        joint = defaultdict(lambda: defaultdict(Fraction))
        for source, weight in self.belief(belief).items():
            for row in self.table[source, action]:
                joint[row.output][row.target] += weight * row.probability.fraction()
        return {o: dict(states) for o, states in sorted(joint.items())}

    def filter(self, action, observed, *, belief=None):
        if type(observed) is not str or observed not in self.outputs:
            raise ValueError("unknown public output")
        mass = self.predict_step(action, belief=belief).get(observed, {})
        likelihood = sum(mass.values(), Fraction(0))
        if likelihood == 0:
            raise ValueError(
                "zero-likelihood evidence: model mismatch, not an implicit reset"
            )
        return likelihood, {s: p / likelihood for s, p in mass.items()}

    def filter_history(self, history, *, belief=None):
        if (
            type(history) is not tuple
            or len(history) > 64
            or any(type(pair) is not tuple or len(pair) != 2 for pair in history)
        ):
            raise ValueError("history requires <=64 (action,output) tuples")
        current, likelihood = self.belief(belief), Fraction(1)
        for action, output in history:
            conditional, current = self.filter(action, output, belief=current)
            likelihood *= conditional
        return likelihood, current

    def sequence_distribution(self, sequence, *, belief=None):
        if (
            type(sequence) is not tuple
            or len(sequence) > 16
            or any(type(a) is not str or a not in self.actions for a in sequence)
        ):
            raise ValueError("sequence requires <=16 known public actions")
        prefixes, operations = {(): self.belief(belief)}, 0
        for action in sequence:
            following = defaultdict(lambda: defaultdict(Fraction))
            for prefix, states in prefixes.items():
                for source, weight in states.items():
                    for row in self.table[source, action]:
                        operations += 1
                        if operations > MAX_BRANCH_OPERATIONS:
                            raise ValueError(
                                "prediction work bound exhausted; never truncate mass"
                            )
                        word = prefix + (row.output,)
                        following[word][row.target] += (
                            weight * row.probability.fraction()
                        )
                        if len(following) > MAX_OUTPUT_PREFIXES:
                            raise ValueError(
                                "output-prefix bound exhausted; no partial distribution"
                            )
            prefixes = following
        result = {
            word: sum(states.values(), Fraction(0))
            for word, states in sorted(prefixes.items())
        }
        if sum(result.values(), Fraction(0)) != 1:
            raise ArithmeticError("lost probability mass")
        return result

    def to_ir(self):
        return TheoryIR(
            mechanism="finite_transducer",
            representation=TheoryRepresentation.STATE_MACHINE,
            scope=dict(SCOPE),
            latent_variables=("machine_state",),
            effects={"output": "transition.output"},
            parameters=self.model_dump(mode="json"),
        )

    @classmethod
    def from_ir(cls, ir):
        if (
            not isinstance(ir, TheoryIR)
            or ir.mechanism != "finite_transducer"
            or ir.representation is not TheoryRepresentation.STATE_MACHINE
            or ir.effects != {"output": "transition.output"}
            or ir.scope != SCOPE
            or ir.preconditions
            or ir.latent_variables
            not in (("machine_state",), ("mechanism_state", "controller_state"))
            or ir.composition
            not in ((), ("delivery-then-mechanism-then-memoryless-sensor",))
        ):
            raise ValueError("unregistered stochastic TheoryIR scope/contract")
        return cls.model_validate(ir.parameters)
