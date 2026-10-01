"""Public, preregistrable action schedules for movement/acquisition diagnosis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MotionProbeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(default=201, ge=0, strict=True)
    observation_size: Literal[224, 448] = 448
    time_of_day: Literal["noon", "midnight"] = "noon"
    weather: Literal["clear", "thunder"] = "clear"
    movement_ticks: tuple[int, int] = (3, 6)
    settle_ticks: int = Field(default=12, ge=8, le=16, strict=True)

    @model_validator(mode="after")
    def _bounded(self):
        first, second = self.movement_ticks
        if (
            any(type(value) is not int for value in self.movement_ticks)
            or not 1 <= first < second <= 10
            or second + self.settle_ticks + 1 > 32
        ):
            raise ValueError("movement/settle windows must be strictly bounded")
        return self

    @property
    def render_size(self) -> tuple[int, int]:
        factor = self.observation_size // 224
        return (640 * factor, 360 * factor)


class ScheduledProbeAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["wait", "look", "move"]
    ticks: int = Field(ge=1, le=20, strict=True)
    motion_handle_index: int | None = Field(default=None, ge=0, le=3, strict=True)
    phase: Literal[
        "startup", "reference_turn", "reference_settle", "move", "settle", "final_wait"
    ]

    @model_validator(mode="after")
    def _handle(self):
        if (self.kind == "move") != (self.motion_handle_index is not None):
            raise ValueError("only movement has an anonymous motion-handle index")
        return self


def motion_probe_schedule(
    config: MotionProbeConfig,
) -> tuple[ScheduledProbeAction, ...]:
    actions = [ScheduledProbeAction(kind="wait", ticks=20, phase="startup")]
    for _ in range(12):
        actions.extend(
            (
                ScheduledProbeAction(kind="look", ticks=1, phase="reference_turn"),
                ScheduledProbeAction(kind="wait", ticks=2, phase="reference_settle"),
            )
        )
    for ticks in config.movement_ticks:
        for index in range(4):
            actions.extend(
                (
                    ScheduledProbeAction(
                        kind="move",
                        ticks=ticks,
                        motion_handle_index=index,
                        phase="move",
                    ),
                    ScheduledProbeAction(
                        kind="wait", ticks=config.settle_ticks, phase="settle"
                    ),
                )
            )
    actions.append(
        ScheduledProbeAction(kind="wait", ticks=config.settle_ticks, phase="final_wait")
    )
    return tuple(actions)


def motion_probe_timeline(config: MotionProbeConfig) -> list[dict]:
    slot = 0
    timeline = []
    for action in motion_probe_schedule(config):
        timeline.append(
            {
                "start_frame": slot,
                "end_frame": slot + action.ticks,
                "scheduled": action.model_dump(mode="json"),
            }
        )
        slot += action.ticks
    return timeline


def motion_probe_pairs(config: MotionProbeConfig) -> list[dict]:
    """Include all fixed trials, not the best-looking or physically correct ones."""
    timeline = motion_probe_timeline(config)
    pairs = []
    for index, item in enumerate(timeline):
        phase = item["scheduled"]["phase"]
        if phase == "move":
            number = sum(pair["group"] == "moving" for pair in pairs)
            settled = timeline[index + 1]
            pairs.extend(
                (
                    {
                        "id": f"moving-{number:02d}",
                        "group": "moving",
                        "first": item["start_frame"],
                        "second": item["end_frame"],
                    },
                    {
                        "id": f"settled-{number:02d}",
                        "group": "settled",
                        "first": item["start_frame"],
                        "second": settled["end_frame"],
                    },
                    {
                        "id": f"settle-control-{number:02d}",
                        "group": "settle_control",
                        "first": settled["end_frame"] - 4,
                        "second": settled["end_frame"],
                    },
                )
            )
    for number, item in enumerate(
        [item for item in timeline if item["scheduled"]["phase"] == "reference_turn"][
            :2
        ]
    ):
        pairs.append(
            {
                "id": f"rotation-{number:02d}",
                "group": "rotation_control",
                "first": item["start_frame"],
                "second": item["end_frame"],
            }
        )
    for item in (timeline[0], timeline[-1]):
        pairs.append(
            {
                "id": item["scheduled"]["phase"],
                "group": "wait_control",
                "first": item["end_frame"] - 4,
                "second": item["end_frame"],
            }
        )
    for slot in (timeline[24]["end_frame"], timeline[-1]["end_frame"]):
        pairs.append(
            {
                "id": f"identity-{slot:03d}",
                "group": "identity_control",
                "first": slot,
                "second": slot,
            }
        )
    return pairs
