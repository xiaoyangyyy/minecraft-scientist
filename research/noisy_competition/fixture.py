"""Finite public prior family, independent of the privately selected condition.

Three mechanism templates, three controller templates and two sensor rates are
explicit prior knowledge, not unknown-controller grammar synthesis. Every policy
gets the same 18 factor theories; actual truth is never passed to the Agent.
"""

import hashlib
from fractions import Fraction

from minecraft_scientist.world.identification import _program
from research.stochastic_ir.composition import NO_DELIVERY
from research.stochastic_ir.program import Mass, StochasticProgram
from research.stochastic_ir.world import RawControlledWorld

from .belief import Candidate

CONDITIONS = {
    "no_change": (0, 0, 1),
    "control_only": (0, 2, 1),
    "mechanism_only": (1, 0, 1),
    "both": (1, 2, 1),
    "unmodeled_sensor": (0, 0, 2),
}


def public_family(seed=0):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("invalid family seed")

    def label(role):
        return (
            "id-"
            + hashlib.sha256(
                f"noisy-competition-v1:{seed}:{role}".encode()
            ).hexdigest()[:16]
        )

    actions = tuple(label(f"action-{i}") for i in range(4))
    outputs = tuple(label(f"output-{i}") for i in range(2))
    mechanisms = tuple(
        p.to_ir()
        for p in (
            _program(actions, outputs),
            _program(actions, outputs, waits=2),
            _program(actions, outputs, activation=False),
        )
    )
    controls = []
    for mode in range(3):
        states = ("controller-0", "controller-1") if mode == 2 else ("controller-0",)
        rows = []
        for state in states:
            for index, action in enumerate(actions):
                passes = Fraction(1)
                if index == 1 and mode:
                    passes = (
                        Fraction(1, 2)
                        if mode == 1
                        else (Fraction(3, 4) if state == states[0] else Fraction(1, 4))
                    )
                rows.append(
                    {
                        "source": state,
                        "action": action,
                        "target": states[0] if mode == 2 and index == 1 else state,
                        "output": action,
                        "probability": Mass.of(passes),
                    }
                )
                if passes < 1:
                    rows.append(
                        {
                            "source": state,
                            "action": action,
                            "target": states[-1] if mode == 2 else state,
                            "output": NO_DELIVERY,
                            "probability": Mass.of(1 - passes),
                        }
                    )
        controls.append(
            StochasticProgram(
                states=states,
                actions=actions,
                outputs=(*actions, NO_DELIVERY),
                initial=tuple(
                    {"state": s, "probability": Mass.of(Fraction(1, len(states)))}
                    for s in states
                ),
                transitions=tuple(rows),
            ).to_ir()
        )
    sensors = tuple(sensor_ir(outputs, rate) for rate in (Fraction(0), Fraction(1, 8)))
    candidates = tuple(
        Candidate(
            key=label(f"candidate-{mi}-{ci}-{si}"),
            mechanism_id=label(f"mechanism-{mi}"),
            controller_id=label(f"controller-{ci}"),
            sensor_id=label(f"sensor-{si}"),
            mechanism_ir=m,
            controller_ir=c,
            sensor_ir=s,
            idle_output=outputs[0],
        )
        for mi, m in enumerate(mechanisms)
        for ci, c in enumerate(controls)
        for si, s in enumerate(sensors)
    )
    return candidates, actions, outputs


def sensor_ir(outputs, rate):
    rows = []
    for index, output in enumerate(outputs):
        rows.append(
            {
                "source": "sensor",
                "action": output,
                "target": "sensor",
                "output": output,
                "probability": Mass.of(1 - rate),
            }
        )
        if rate:
            rows.append(
                {
                    "source": "sensor",
                    "action": output,
                    "target": "sensor",
                    "output": outputs[1 - index],
                    "probability": Mass.of(rate),
                }
            )
    return StochasticProgram(
        states=("sensor",),
        actions=outputs,
        outputs=outputs,
        initial=({"state": "sensor", "probability": Mass.of(Fraction(1))},),
        transitions=tuple(rows),
    ).to_ir()


def make_world(seed, condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown noisy development condition")
    candidates, actions, outputs = public_family(seed)
    mi, ci, si = CONDITIONS[condition]
    chosen = next(
        c
        for c in candidates
        if c.mechanism_id == candidates[mi * 6].mechanism_id
        and c.controller_id == candidates[ci * 2].controller_id
        and c.sensor_id == candidates[min(si, 1)].sensor_id
    )
    sensor = chosen.sensor_ir if si < 2 else sensor_ir(outputs, Fraction(1, 2))
    world = RawControlledWorld(
        mechanism_ir=chosen.mechanism_ir,
        controller_ir=chosen.controller_ir,
        sensor_ir=sensor,
        idle_output=chosen.idle_output,
        seed=seed,
    )
    truth = {
        "mechanism_id": chosen.mechanism_id,
        "controller_id": chosen.controller_id,
        "sensor_id": chosen.sensor_id if si < 2 else "out-of-public-sensor-family",
        "in_public_family": si < 2,
    }
    return world, candidates, {a: 1 for a in actions}, truth
