"""Research-only control-time profile, never weaker pixel or geometry gates.

This prototype exercises the existing waiting executor before integration into a
new live registration. It does not extend saved native trajectories or assert a
successful map. Current production/full-regression inputs remain unchanged.
"""

from dataclasses import dataclass

from minecraft_scientist.exploration.public_wait_recovery import settle_with_recovery
from minecraft_scientist.exploration.stable_keyframe_map import (
    StableJumpSafeSeedKeyframePolicy,
)


@dataclass(frozen=True, slots=True)
class WaitWindow:
    minimum_wait_ticks: int = 8
    maximum_wait_ticks: int = 24
    stable_prefixes: int = 3

    def __post_init__(self):
        if (
            type(self.minimum_wait_ticks) is not int
            or self.minimum_wait_ticks != 8
            or type(self.maximum_wait_ticks) is not int
            or self.maximum_wait_ticks != 24
            or type(self.stable_prefixes) is not int
            or self.stable_prefixes != 3
        ):
            raise ValueError("research timing profile has exact eight/24/three budgets")


def profile(policy, phase):
    if phase not in ("initial", "acquisition", "future"):
        raise ValueError("requires a public experiment phase")
    settings = StableJumpSafeSeedKeyframePolicy.model_validate(policy)
    # Validate the entire parent first: temporal execution cost is a separate
    # parameter, not a way to alter ray, patch, support or bootstrap gates.
    return settings.stability if phase == "initial" else WaitWindow()


def execute(result, phase, step, ordinal, policy):
    selected = profile(policy, phase)
    return settle_with_recovery(result, phase, step, ordinal, selected)
