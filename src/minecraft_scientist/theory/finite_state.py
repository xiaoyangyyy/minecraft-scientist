"""Bounded deterministic transducers in the existing TheoryIR, without eval.

Product-automaton witnesses are established automata machinery, not a claimed
novel causal-learning algorithm. Equivalence is conditional on this language and
action contract; it does not establish unique causal structure.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from minecraft_scientist.protocols import TheoryIR, TheoryRepresentation


class StateTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: StrictStr = Field(min_length=1, max_length=80)
    action: StrictStr = Field(min_length=1, max_length=80)
    target: StrictStr = Field(min_length=1, max_length=80)
    output: StrictStr = Field(min_length=1, max_length=80)


class FiniteStateProgram(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    language: Literal["total-mealy-v1"] = "total-mealy-v1"
    states: tuple[StrictStr, ...] = Field(min_length=1, max_length=16)
    actions: tuple[StrictStr, ...] = Field(min_length=1, max_length=8)
    outputs: tuple[StrictStr, ...] = Field(min_length=1, max_length=8)
    initial_state: StrictStr
    transitions: tuple[StateTransition, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def total_contract(self):
        for values in (self.states, self.actions, self.outputs):
            if len(set(values)) != len(values) or any(
                not value or len(value) > 80 for value in values
            ):
                raise ValueError("alphabets require bounded unique nonempty labels")
        if self.initial_state not in self.states:
            raise ValueError("initial state is outside the program")
        pairs = [(row.source, row.action) for row in self.transitions]
        expected = {(state, action) for state in self.states for action in self.actions}
        if len(set(pairs)) != len(pairs) or set(pairs) != expected:
            raise ValueError("exactly one transition per state/action is required")
        if any(
            row.target not in self.states or row.output not in self.outputs
            for row in self.transitions
        ):
            raise ValueError("transition references unknown state/output")
        return self

    @cached_property
    def table(self):
        # An internal immutable-key lookup; callers do not receive this object.
        return MappingProxyType(
            {(row.source, row.action): row for row in self.transitions}
        )

    @classmethod
    def from_ir(cls, ir: TheoryIR):
        if (
            ir.mechanism != "finite_transducer"
            or ir.representation is not TheoryRepresentation.STATE_MACHINE
            or ir.effects != {"output": "transition.output"}
        ):
            raise ValueError("unregistered finite-state IR contract")
        return cls.model_validate(ir.parameters)

    def to_ir(self) -> TheoryIR:
        return TheoryIR(
            mechanism="finite_transducer",
            representation=TheoryRepresentation.STATE_MACHINE,
            scope={"language": self.language, "reset": "initial-state"},
            latent_variables=("machine_state",),
            effects={"output": "transition.output"},
            parameters=self.model_dump(mode="json"),
        )

    def execute(self, sequence: tuple[str, ...], *, state: str | None = None):
        if type(sequence) is not tuple or len(sequence) > 256:
            raise ValueError("execution requires a tuple of at most 256 actions")
        current = self.initial_state if state is None else state
        if current not in self.states:
            raise ValueError("unknown starting state")
        emitted = []
        for action in sequence:
            if type(action) is not str or action not in self.actions:
                raise ValueError("unknown action")
            row = self.table[current, action]
            current = row.target
            emitted.append(row.output)
        return tuple(emitted), current


@dataclass(frozen=True)
class DistinguishingWitness:
    actions: tuple[str, ...]
    action_cost: int
    left_outputs: tuple[str, ...]
    right_outputs: tuple[str, ...]


def validate_costs(actions: tuple[str, ...], costs: dict[str, int]) -> None:
    if (
        type(costs) is not dict
        or set(costs) != set(actions)
        or any(type(cost) is not int or not 1 <= cost <= 100 for cost in costs.values())
    ):
        raise ValueError("every public action requires an exact positive integer cost")


def bounded_behavior_classes(programs: tuple[FiniteStateProgram, ...], horizon: int):
    """Exact reset-query equivalence up to H, via finite-depth partition refinement.

    This does not prove unbounded equivalence. No exponential word enumeration or
    digest-collision assumption is used: signatures are interned exact tuples.
    """
    if (
        not programs
        or len(programs) > 512
        or type(horizon) is not int
        or not 0 <= horizon <= 8
    ):
        raise ValueError("bounded equivalence requires <=512 programs and H in [0,8]")
    actions = tuple(sorted(programs[0].actions))
    if any(set(program.actions) != set(actions) for program in programs):
        raise ValueError("bounded equivalence requires shared action alphabets")
    colors = {
        (index, state): 0
        for index, program in enumerate(programs)
        for state in program.states
    }
    for _ in range(horizon):
        signatures, refined = {}, {}
        for index, program in enumerate(programs):
            for state in program.states:
                signature = tuple(
                    (
                        program.table[state, action].output,
                        colors[index, program.table[state, action].target],
                    )
                    for action in actions
                )
                refined[index, state] = signatures.setdefault(
                    signature, len(signatures)
                )
        colors = refined
    groups = {}
    for index, program in enumerate(programs):
        groups.setdefault(colors[index, program.initial_state], []).append(index)
    return tuple(tuple(group) for group in groups.values())


def shortest_witness(
    left: FiniteStateProgram,
    right: FiniteStateProgram,
    costs: dict[str, int],
    *,
    maximum_length: int | None = None,
) -> DistinguishingWitness | None:
    """Minimum ACTION cost; query reset is a separate shared fixed cost.

    None with no horizon proves equivalence for these total deterministic
    programs. None with a horizon proves only absence inside that horizon.
    """
    if set(left.actions) != set(right.actions) or set(left.outputs) != set(
        right.outputs
    ):
        raise ValueError("witnesses require one shared public contract")
    validate_costs(left.actions, costs)
    if maximum_length is not None and (
        type(maximum_length) is not int or not 0 <= maximum_length <= 64
    ):
        raise ValueError("maximum length must be an exact bounded integer")
    queue = [(0, (), left.initial_state, right.initial_state, False)]
    visited = set()
    while queue:
        cost, word, ls, rs, separated = heapq.heappop(queue)
        if separated:
            return DistinguishingWitness(
                word, cost, left.execute(word)[0], right.execute(word)[0]
            )
        key = (ls, rs) if maximum_length is None else (ls, rs, len(word))
        if key in visited:
            continue
        visited.add(key)
        if maximum_length is not None and len(word) >= maximum_length:
            continue
        for action in sorted(left.actions):
            lr, rr = left.table[ls, action], right.table[rs, action]
            heapq.heappush(
                queue,
                (
                    cost + costs[action],
                    word + (action,),
                    lr.target,
                    rr.target,
                    lr.output != rr.output,
                ),
            )
    return None
