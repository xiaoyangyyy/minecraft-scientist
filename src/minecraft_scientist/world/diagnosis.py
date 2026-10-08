"""Hidden actuator faults and changed mechanisms with independent world execution.

No fault flag or delivered command is observable. An optional, costly alternate
actuation path is a real intervention, not a ground-truth diagnostic oracle.
This development world has deterministic resettable controller models.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from minecraft_scientist.discovery.diagnosis import NO_DELIVERY, JointHypothesis
from minecraft_scientist.protocols import AgentWorldView, TransitionRecord, WorldEvent
from minecraft_scientist.theory.finite_state import FiniteStateProgram
from minecraft_scientist.world.identification import _FiniteWorld, _program

MECHANISMS = ("unchanged", "delayed", "activation_edge_removed")
CONTROLLERS = ("healthy", "primary_drop", "alternating_drop")


@dataclass(frozen=True)
class DiagnosisTask:
    hypotheses: tuple[JointHypothesis, ...]
    calibration: tuple[str, ...]
    costs: dict[str, int]
    baseline_mechanism: str
    baseline_controller: str
    view: AgentWorldView
    private_spec: dict  # Only evaluator; never pass this task as a whole to Agent.


class _ControlledWorld(_FiniteWorld):
    def __init__(self, mechanism, controller):
        super().__init__(mechanism)
        self._controller_initial = controller.initial_state
        self._controller_state = controller.initial_state
        self._control_rows = {
            (row.source, row.action): (row.target, row.output)
            for row in controller.transitions
        }
        self._actions = tuple(sorted(controller.actions))

    def begin_trial(self):
        self._controller_state = self._controller_initial
        return super().begin_trial()

    def step(self, action):
        if self._episode < 0 or action not in self.available_actions():
            raise ValueError("reset and a valid public command are required")
        before = self.observe()
        self._controller_state, delivered = self._control_rows[
            self._controller_state, action.target_id
        ]
        signal = self._zero
        if delivered != NO_DELIVERY:
            self._state, signal = self._rows[self._state, delivered]
        self._tick += 1
        self._event = WorldEvent(event_type="public_signal", details={"signal": signal})
        return TransitionRecord(
            transition_id=f"episode-{self._episode}-step-{self._tick}",
            context={
                "intervention_compliance_scope": "issued-command",
                "treatment_verification": "unknown",
            },
            state_before=before,
            executed_action=action,
            # Only the command dispatch was checked; actuator delivery is latent.
            intervention_compliance=True,
            state_after=self.observe(),
            observed_delta={"signal": signal},
            event=self._event,
            possible_confounders=("unverified-actuator-delivery",),
            provenance={"schema": "hidden-controller-h2-development-v1"},
        )


def _controller(actions, issued, *, mode):
    states = ("c-drop", "c-pass") if mode == "alternating_drop" else ("c-fixed",)
    rows = []
    for state in states:
        for index, command in enumerate(issued):
            delivered = actions[index] if index < 4 else actions[1]
            target = state
            if index == 1 and mode != "healthy":
                if mode == "primary_drop" or state == "c-drop":
                    delivered = NO_DELIVERY
                if mode == "alternating_drop":
                    target = "c-pass" if state == "c-drop" else "c-drop"
            rows.append(
                {
                    "source": state,
                    "action": command,
                    "target": target,
                    "output": delivered,
                }
            )
    return FiniteStateProgram(
        states=states,
        actions=issued,
        outputs=(*actions, NO_DELIVERY),
        initial_state=states[0],
        transitions=tuple(rows),
    )


def diagnosis_task(
    seed=0, mechanism="unchanged", controller="primary_drop", *, bypass=True
):
    if (
        type(seed) is not int
        or seed < 0
        or type(bypass) is not bool
        or mechanism not in MECHANISMS
        or controller not in CONTROLLERS
    ):
        raise ValueError("invalid diagnosis development condition")

    def label(role):
        return "id-" + hashlib.sha256(f"h2-v1:{seed}:{role}".encode()).hexdigest()[:16]

    actions = tuple(label(f"action-{i}") for i in range(4))
    issued = (*actions, label("alternate-actuation")) if bypass else actions
    outputs = tuple(label(f"signal-{i}") for i in range(2))
    old = _program(actions, outputs)
    delayed = _program(actions, outputs, waits=2)
    broken = old.model_dump(mode="json")
    for row in broken["transitions"]:
        if row["action"] == actions[1]:
            row["target"] = row["source"]
    programs = (old, delayed, FiniteStateProgram.model_validate(broken))
    controls = tuple(_controller(actions, issued, mode=mode) for mode in CONTROLLERS)
    mids = tuple(label(f"mechanism-{i}") for i in range(3))
    cids = tuple(label(f"controller-{i}") for i in range(3))
    hypotheses = tuple(
        JointHypothesis(
            mechanism_id=mids[mi],
            controller_id=cids[ci],
            mechanism_ir=program.to_ir(),
            controller_ir=control.to_ir(),
        )
        for mi, program in enumerate(programs)
        for ci, control in enumerate(controls)
    )
    mi, ci = MECHANISMS.index(mechanism), CONTROLLERS.index(controller)
    world = _ControlledWorld(programs[mi], controls[ci])
    view = AgentWorldView(
        observe=world.observe,
        available=world.available_actions,
        step=world.step,
        begin_trial=world.begin_trial,
    )
    private = {
        "schema": "joint-diagnosis-private-v1",
        "mechanism_id": mids[mi],
        "controller_id": cids[ci],
        "mechanism": programs[mi].model_dump(mode="json"),
        "controller": controls[ci].model_dump(mode="json"),
    }
    return DiagnosisTask(
        hypotheses,
        (actions[0], actions[1], actions[2], actions[3]),
        {action: 3 if action not in actions else 1 for action in issued},
        mids[0],
        cids[0],
        view,
        private,
    )
