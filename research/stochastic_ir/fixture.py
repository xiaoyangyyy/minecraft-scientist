"""Anonymous known-model fixture ONLY for stochastic execution verification.

This supplies the prediction model intentionally. It is not an unknown-controller
discovery task and cannot contribute mechanism-identification/innovation scores.
"""

import hashlib
from fractions import Fraction

from minecraft_scientist.world.identification import _program

from .composition import NO_DELIVERY, compile_joint
from .program import Mass, StochasticProgram
from .world import RawControlledWorld


def fixture(seed=0, *, noise=True):
    if type(seed) is not int or not 0 <= seed < 2**64 or type(noise) is not bool:
        raise ValueError("invalid fixture seed/noise mode")

    def label(role):
        return (
            "id-"
            + hashlib.sha256(
                f"stochastic-semantics-v1:{seed}:{role}".encode()
            ).hexdigest()[:16]
        )

    actions = tuple(label(f"a-{i}") for i in range(4))
    outputs = tuple(label(f"o-{i}") for i in range(2))
    mechanism = _program(actions, outputs)
    states = ("controller-0", "controller-1")
    branches = []
    for state in states:
        for index, action in enumerate(actions):
            if index == 1:
                # Correlated latent mode: a failed primary actuation changes the
                # next actuation's distribution. Neither mode/delivery is public.
                passes = Fraction(3, 4) if state == states[0] else Fraction(1, 4)
                branches.extend(
                    (
                        {
                            "source": state,
                            "action": action,
                            "target": states[0],
                            "output": action,
                            "probability": Mass.of(passes),
                        },
                        {
                            "source": state,
                            "action": action,
                            "target": states[1],
                            "output": NO_DELIVERY,
                            "probability": Mass.of(1 - passes),
                        },
                    )
                )
            else:
                branches.append(
                    {
                        "source": state,
                        "action": action,
                        "target": state,
                        "output": action,
                        "probability": Mass.of(Fraction(1)),
                    }
                )
    controller = StochasticProgram(
        states=states,
        actions=actions,
        outputs=(*actions, NO_DELIVERY),
        initial=(
            {"state": states[0], "probability": Mass.of(Fraction(1, 2))},
            {"state": states[1], "probability": Mass.of(Fraction(1, 2))},
        ),
        transitions=tuple(branches),
    )
    sensor_rows = []
    for index, latent in enumerate(outputs):
        sensor_rows.append(
            {
                "source": "sensor",
                "action": latent,
                "target": "sensor",
                "output": latent,
                "probability": Mass.of(Fraction(7, 8) if noise else Fraction(1)),
            }
        )
        if noise:
            sensor_rows.append(
                {
                    "source": "sensor",
                    "action": latent,
                    "target": "sensor",
                    "output": outputs[1 - index],
                    "probability": Mass.of(Fraction(1, 8)),
                }
            )
    sensor = StochasticProgram(
        states=("sensor",),
        actions=outputs,
        outputs=outputs,
        initial=({"state": "sensor", "probability": Mass.of(Fraction(1))},),
        transitions=tuple(sensor_rows),
    )
    inputs = {
        "mechanism_ir": mechanism.to_ir(),
        "controller_ir": controller.to_ir(),
        "sensor_ir": sensor.to_ir(),
        "idle_output": outputs[0],
    }
    world = RawControlledWorld(**inputs, seed=seed)
    return world, compile_joint(**inputs), actions, outputs
