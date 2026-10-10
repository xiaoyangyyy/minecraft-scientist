"""Compile separate delivery and sensor factors into the shared TheoryIR."""

from collections import defaultdict
from fractions import Fraction

from minecraft_scientist.theory.finite_state import FiniteStateProgram

from .program import Mass, StochasticProgram

NO_DELIVERY = "_no_delivery_"


def lift_deterministic(ir):
    program = FiniteStateProgram.from_ir(ir)
    return StochasticProgram(
        states=program.states,
        actions=program.actions,
        outputs=program.outputs,
        initial=(
            {"state": program.initial_state, "probability": Mass.of(Fraction(1))},
        ),
        transitions=tuple(
            {**row.model_dump(), "probability": Mass.of(Fraction(1))}
            for row in program.transitions
        ),
    ).to_ir()


def compile_joint(mechanism_ir, controller_ir, sensor_ir, *, idle_output):
    mechanism = FiniteStateProgram.from_ir(mechanism_ir)
    controller = StochasticProgram.from_ir(controller_ir)
    sensor = StochasticProgram.from_ir(sensor_ir)
    if (
        NO_DELIVERY in mechanism.actions
        or not set(controller.outputs) <= {*mechanism.actions, NO_DELIVERY}
        or idle_output not in mechanism.outputs
    ):
        raise ValueError("controller outputs must be delivered actions or nondelivery")
    if len(sensor.states) != 1 or set(sensor.actions) != set(mechanism.outputs):
        raise ValueError("this sensor must be memoryless and cover every latent output")
    if len(mechanism.states) * len(controller.states) > 16:
        raise ValueError("joint states exceed the declared shared-language bound")
    names = {
        (m, c): f"joint-{mi}-{ci}"
        for mi, m in enumerate(mechanism.states)
        for ci, c in enumerate(controller.states)
    }
    branches = []
    for (ms, cs), source in names.items():
        for action in controller.actions:
            masses = defaultdict(Fraction)
            for delivery in controller.table[cs, action]:
                target, latent = ms, idle_output
                if delivery.output != NO_DELIVERY:
                    row = mechanism.table[ms, delivery.output]
                    target, latent = row.target, row.output
                for emission in sensor.table[sensor.states[0], latent]:
                    masses[names[target, delivery.target], emission.output] += (
                        delivery.probability.fraction()
                        * emission.probability.fraction()
                    )
            branches.extend(
                {
                    "source": source,
                    "action": action,
                    "target": target,
                    "output": output,
                    "probability": Mass.of(mass),
                }
                for (target, output), mass in sorted(masses.items())
            )
    initial = tuple(
        {
            "state": names[mechanism.initial_state, entry.state],
            "probability": entry.probability,
        }
        for entry in controller.initial
    )
    return (
        StochasticProgram(
            states=tuple(names.values()),
            actions=controller.actions,
            outputs=sensor.outputs,
            initial=initial,
            transitions=tuple(branches),
        )
        .to_ir()
        .model_copy(
            update={
                "latent_variables": ("mechanism_state", "controller_state"),
                "composition": ("delivery-then-mechanism-then-memoryless-sensor",),
            }
        )
    )
