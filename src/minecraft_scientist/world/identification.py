"""Small resettable anonymous worlds for the deterministic H1 protocol slice.

Execution deliberately does not call TheoryInterpreter/FiniteStateProgram.execute.
The capability view is a trusted-agent API boundary, not an adversarial sandbox:
Python reflection into the capability registry is not a security guarantee.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from minecraft_scientist.protocols import (
    ActionKind,
    AgentWorldView,
    ObjectObservation,
    Observation,
    PrimitiveAction,
    TheoryIR,
    TransitionRecord,
    WorldEvent,
)
from minecraft_scientist.theory.finite_state import FiniteStateProgram

FAMILIES = ("activation", "delay", "toggle", "equivalent", "model_mismatch")


@dataclass(frozen=True)
class IdentificationTask:
    candidates: tuple[TheoryIR, ...]
    calibration: tuple[str, ...]
    costs: dict[str, int]
    view: AgentWorldView
    ground_truth: FiniteStateProgram  # Evaluator-only; never passed to planner/agent.


class _FiniteWorld:
    def __init__(self, program: FiniteStateProgram):
        self._initial = program.initial_state
        self._rows = {
            (row.source, row.action): (row.target, row.output)
            for row in program.transitions
        }
        self._actions = tuple(sorted(program.actions))
        self._zero = program.outputs[0]
        self._state = self._initial
        self._episode = -1
        self._tick = 0
        self._event = None

    def begin_trial(self):
        self._state, self._tick, self._event = self._initial, 0, None
        self._episode += 1
        return self.observe()

    def observe(self):
        return Observation(
            episode_id=max(0, self._episode),
            step=self._tick,
            inventory={},
            agent_state={
                "signal": self._event.details["signal"]
                if self._event is not None
                else self._zero
            },
            objects=tuple(
                ObjectObservation(object_id=action, affordances=(ActionKind.INTERACT,))
                for action in self._actions
            ),
            last_event=self._event,
        )

    def available_actions(self):
        return tuple(
            PrimitiveAction(kind=ActionKind.INTERACT, target_id=action)
            for action in self._actions
        )

    def step(self, action):
        if self._episode < 0:
            raise ValueError("begin_trial is required before execution")
        if action not in self.available_actions():
            raise ValueError("action is outside the public contract")
        before = self.observe()
        target, signal = self._rows[self._state, action.target_id]
        self._state = target
        self._tick += 1
        self._event = WorldEvent(event_type="public_signal", details={"signal": signal})
        return TransitionRecord(
            transition_id=f"episode-{self._episode}-step-{self._tick}",
            context={},
            state_before=before,
            executed_action=action,
            intervention_compliance=True,
            state_after=self.observe(),
            observed_delta={"signal": signal},
            event=self._event,
            provenance={"schema": "finite-world-h1-v1"},
        )


def _program(actions, outputs, *, activation=True, waits=1, toggle=False):
    load, start, wait, take = actions
    empty, loaded, ready = "q-empty", "q-loaded", "q-ready"
    active = tuple(f"q-wait-{index}" for index in range(waits))
    states = (empty, loaded, *active, ready)
    rows = []
    for state in states:
        for action in actions:
            target, output = state, outputs[0]
            if action == load:
                target = loaded if activation else active[0]
            elif action == start and state != empty:
                target = (
                    loaded
                    if toggle and state != loaded
                    else (active[0] if state == loaded else state)
                )
            elif action == wait and state in active:
                index = active.index(state)
                target = active[index + 1] if index + 1 < waits else ready
            elif action == take and state == ready:
                target, output = empty, outputs[1]
            rows.append(
                {"source": state, "action": action, "target": target, "output": output}
            )
    return FiniteStateProgram(
        states=states,
        actions=actions,
        outputs=outputs,
        initial_state=empty,
        transitions=tuple(rows),
    )


def identification_task(seed: int, family: str) -> IdentificationTask:
    if type(seed) is not int or seed < 0 or family not in FAMILIES:
        raise ValueError("unknown family or invalid seed")

    def label(prefix, role):
        # Labels depend on role/seed, NOT on hidden mechanism or truth choice.
        digest = hashlib.sha256(f"h1-v1:{seed}:{role}".encode()).hexdigest()[:16]
        return f"{prefix}-{digest}"

    actions = tuple(label("a", role) for role in range(4))
    outputs = tuple(label("o", role + 10) for role in range(2))
    if family in ("activation", "equivalent"):
        programs = (
            _program(actions, outputs),
            _program(actions, outputs, activation=False),
        )
    elif family in ("delay", "model_mismatch"):
        programs = (_program(actions, outputs), _program(actions, outputs, waits=2))
    else:
        programs = (_program(actions, outputs), _program(actions, outputs, toggle=True))
    if family == "equivalent":
        original = programs[0].model_dump(mode="json")
        renaming = {
            state: f"renamed-{index}" for index, state in enumerate(original["states"])
        }
        original["states"] = [renaming[state] for state in original["states"]]
        original["initial_state"] = renaming[original["initial_state"]]
        for row in original["transitions"]:
            row["source"], row["target"] = (
                renaming[row["source"]],
                renaming[row["target"]],
            )
        programs = programs[0], FiniteStateProgram.model_validate(original)
    truth = (
        _program(actions, outputs, waits=3)
        if family == "model_mismatch"
        else (programs[seed % len(programs)])
    )
    world = _FiniteWorld(truth)
    view = AgentWorldView(
        observe=world.observe,
        available=world.available_actions,
        step=world.step,
        begin_trial=world.begin_trial,
    )
    return IdentificationTask(
        tuple(program.to_ir() for program in programs),
        (actions[0], actions[1], actions[2], actions[2], actions[3]),
        {action: 1 for action in actions},
        view,
        truth,
    )
