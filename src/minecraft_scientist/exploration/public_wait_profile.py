"""Exact post-motor control-time budget, independent of pixel acceptance gates."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PostMotorWaitWindow:
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
            raise ValueError("post-motor timing requires exact eight/24/three budgets")
