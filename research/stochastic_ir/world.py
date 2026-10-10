"""Independent raw component execution behind the unchanged capability boundary.

No joint compiler, TheoryInterpreter, belief filter or predictive table runs here.
Only the issued command and measured signal enter public receipts. Randomness
is exact rational sampling; resets do not rewind/reseed the random stream.
"""

import random
from copy import deepcopy
from math import lcm

from minecraft_scientist.protocols import (
    ActionKind,
    AgentWorldView,
    ObjectObservation,
    Observation,
    PrimitiveAction,
    TransitionRecord,
    WorldEvent,
)
from minecraft_scientist.theory.finite_state import FiniteStateProgram

from .composition import NO_DELIVERY
from .program import StochasticProgram


def _weights(entries):
    denominator = lcm(*(row["probability"]["denominator"] for row in entries))
    if denominator.bit_length() > 256:
        raise ValueError("raw sampler denominator exceeds exact work bound")
    weights = tuple(
        row["probability"]["numerator"]
        * (denominator // row["probability"]["denominator"])
        for row in entries
    )
    if sum(weights) != denominator:
        raise ValueError("raw conditional mass must sum exactly to one")
    return denominator, weights


class RawControlledWorld:
    def __init__(self, *, mechanism_ir, controller_ir, sensor_ir, idle_output, seed):
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("seed must be a bounded nonnegative exact integer")
        mechanism = FiniteStateProgram.from_ir(mechanism_ir).model_dump(mode="json")
        controller = StochasticProgram.from_ir(controller_ir).model_dump(mode="json")
        sensor = StochasticProgram.from_ir(sensor_ir).model_dump(mode="json")
        if (
            NO_DELIVERY in mechanism["actions"]
            or not set(controller["outputs"]) <= {*mechanism["actions"], NO_DELIVERY}
            or idle_output not in mechanism["outputs"]
        ):
            raise ValueError("invalid delivery/idle action contract")
        if len(sensor["states"]) != 1 or set(sensor["actions"]) != set(
            mechanism["outputs"]
        ):
            raise ValueError("memoryless sensor must cover latent output alphabet")
        self._private_spec = deepcopy(
            {
                "mechanism": mechanism,
                "controller": controller,
                "sensor": sensor,
                "idle_output": idle_output,
                "seed": seed,
            }
        )
        self._mechanism = {
            (r["source"], r["action"]): (r["target"], r["output"])
            for r in mechanism["transitions"]
        }
        self._control, self._sensor = {}, {}
        for destination, model in ((self._control, controller), (self._sensor, sensor)):
            for row in model["transitions"]:
                destination.setdefault((row["source"], row["action"]), []).append(row)
            for rows in destination.values():
                rows.sort(key=lambda r: (r["target"], r["output"]))
                _weights(rows)
        self._initial_control = sorted(controller["initial"], key=lambda r: r["state"])
        _weights(self._initial_control)
        self._initial_mechanism = mechanism["initial_state"]
        self._sensor_state = sensor["states"][0]
        self._idle, self._zero = idle_output, min(sensor["outputs"])
        self._actions = tuple(sorted(controller["actions"]))
        self._rng = random.Random(seed)
        self._episode, self._tick, self._event = -1, 0, None
        self._trace = []

    def _sample(self, rows):
        denominator, weights = _weights(rows)
        draw = self._rng.randrange(denominator)
        cursor = 0
        for row, weight in zip(rows, weights, strict=True):
            cursor += weight
            if draw < cursor:
                return row
        raise ArithmeticError("exact raw sampling failed")

    def begin_trial(self):
        self._ms = self._initial_mechanism
        self._cs = self._sample(self._initial_control)["state"]
        self._episode += 1
        self._tick, self._event = 0, None
        observation = self.observe()
        self._trace.append(
            {"kind": "reset", "observation": observation.model_dump(mode="json")}
        )
        return observation

    def observe(self):
        return Observation(
            episode_id=max(0, self._episode),
            step=self._tick,
            inventory={},
            agent_state={
                "signal": self._zero
                if self._event is None
                else self._event.details["signal"]
            },
            objects=tuple(
                ObjectObservation(object_id=a, affordances=(ActionKind.INTERACT,))
                for a in self._actions
            ),
            last_event=self._event,
        )

    def available_actions(self):
        return tuple(
            PrimitiveAction(kind=ActionKind.INTERACT, target_id=a)
            for a in self._actions
        )

    def step(self, action):
        if self._episode < 0 or action not in self.available_actions():
            raise ValueError("a reset and a valid public command are required")
        before = self.observe()
        control = self._sample(self._control[self._cs, action.target_id])
        self._cs, latent = control["target"], self._idle
        if control["output"] != NO_DELIVERY:
            self._ms, latent = self._mechanism[self._ms, control["output"]]
        measured = self._sample(self._sensor[self._sensor_state, latent])["output"]
        self._tick += 1
        self._event = WorldEvent(
            event_type="public_signal", details={"signal": measured}
        )
        record = TransitionRecord(
            transition_id=f"episode-{self._episode}-step-{self._tick}",
            context={
                "intervention_compliance_scope": "issued-command",
                "treatment_verification": "unknown",
            },
            state_before=before,
            executed_action=action,
            intervention_compliance=True,
            state_after=self.observe(),
            observed_delta={"signal": measured},
            event=self._event,
            possible_confounders=(
                "latent-actuator-delivery",
                "latent-mechanism-state",
                "measurement-noise",
            ),
            provenance={"schema": "stochastic-component-world-v1"},
        )
        self._trace.append(
            {"kind": "step", "transition": record.model_dump(mode="json")}
        )
        return record

    def agent_view(self):
        # Never retain the view key on this owner (avoids callback capability
        # cycles). Registry callbacks keep the world alive while the view lives.
        return AgentWorldView(
            observe=self.observe,
            available=self.available_actions,
            step=self.step,
            begin_trial=self.begin_trial,
        )

    def evaluator_spec(self):
        return deepcopy(self._private_spec)

    def evaluator_trace(self):
        return deepcopy(self._trace)
