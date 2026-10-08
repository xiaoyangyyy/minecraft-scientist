"""Stationary black-box development tasks, not a formal structural holdout.

The runner alone obtains private program/state count. The learner receives only
an anonymous alphabet, action costs and a capability view. Raw table execution
uses _FiniteWorld, independently of the learned-program interpreter.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from minecraft_scientist.protocols import AgentWorldView
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.world.identification import _FiniteWorld

AUTOMATA_FAMILIES = ("dense", "sparse", "cycle", "branch")


@dataclass(frozen=True)
class AutomataTask:
    actions: tuple[str, ...]
    outputs: tuple[str, ...]
    costs: dict[str, int]
    view: AgentWorldView
    ground_truth: FiniteStateProgram  # Runner/evaluator only.


def automata_task(seed: int, family: str, states: int) -> AutomataTask:
    if (
        type(seed) is not int
        or seed < 0
        or family not in AUTOMATA_FAMILIES
        or type(states) is not int
        or not 2 <= states <= 8
    ):
        raise ValueError("invalid stationary automata development task")

    def label(prefix, role):
        # No family, state count, output semantics or true mechanism in labels.
        digest = hashlib.sha256(f"aal-labels-v1:{seed}:{role}".encode()).hexdigest()[
            :16
        ]
        return f"{prefix}-{digest}"

    actions = tuple(label("a", role) for role in range(2))
    outputs = tuple(label("o", role + 10) for role in range(2))
    names = tuple(label("q", role + 20) for role in range(states))
    rng = random.Random(f"aal-tables-v1:{seed}:{family}:{states}")
    rows = []
    for i, name in enumerate(names):
        for j, action in enumerate(actions):
            target = (i + 1) % states if j == 0 else rng.randrange(states)
            output = outputs[rng.randrange(2)]
            if family == "sparse":
                output = outputs[int(i == states - 1 and j == 1)]
            elif family == "cycle":
                target = (i + 1) % states if j == 0 else i
                output = outputs[int(i == states - 1 and j == 0)]
            elif family == "branch":
                target = (i + 1) % states if j == 0 else (i + 2) % states
                output = outputs[int(j == 1 and i in (states // 2, states - 1))]
            rows.append(
                {
                    "source": name,
                    "action": action,
                    "target": names[target],
                    "output": output,
                }
            )
    program = FiniteStateProgram(
        states=names,
        actions=actions,
        outputs=outputs,
        initial_state=names[0],
        transitions=tuple(rows),
    )
    world = _FiniteWorld(program)
    return AutomataTask(
        actions,
        outputs,
        {a: 1 for a in actions},
        AgentWorldView(
            observe=world.observe,
            available=world.available_actions,
            step=world.step,
            begin_trial=world.begin_trial,
        ),
        program,
    )
