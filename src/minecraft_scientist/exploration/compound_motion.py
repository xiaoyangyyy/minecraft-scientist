"""Fixed distinct-handle two-leg acquisition; physical curvature is evaluator-only."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from .motion_probe_schedule import (
    MotionProbeConfig,
    ScheduledProbeAction,
    motion_probe_pairs,
    motion_probe_schedule,
    motion_probe_timeline,
)


class CompoundProbeConfig(MotionProbeConfig):
    sequence_kind: Literal["paired-controls-v1"] = "paired-controls-v1"
    movement_ticks: tuple[int, int] = (4, 8)
    settle_ticks: int = Field(default=8, ge=8, le=16, strict=True)
    handle_pairs: tuple[tuple[StrictInt, StrictInt], ...] = (
        (0, 2),
        (2, 1),
        (1, 3),
        (3, 0),
    )

    @model_validator(mode="after")
    def _compound_budget(self):
        if (
            not self.handle_pairs
            or len(self.handle_pairs) > 8
            or len(set(self.handle_pairs)) != len(self.handle_pairs)
            or any(
                first == second or not 0 <= first <= 3 or not 0 <= second <= 3
                for first, second in self.handle_pairs
            )
            or 2 * max(self.movement_ticks) + self.settle_ticks + 1 > 32
            or 57
            + sum(
                len(self.handle_pairs) * (2 * ticks + self.settle_ticks)
                for ticks in self.movement_ticks
            )
            + self.settle_ticks
            > 500
        ):
            raise ValueError(
                "compound controls must be distinct, unique and within frame budgets"
            )
        return self


def compound_schedule(config: CompoundProbeConfig):
    # Keep the canonical 20-frame startup and twelve RGB reference rotations.
    actions = list(motion_probe_schedule(config)[:25])
    for ticks in config.movement_ticks:
        for first, second in config.handle_pairs:
            actions.extend(
                (
                    ScheduledProbeAction(
                        kind="move",
                        ticks=ticks,
                        motion_handle_index=first,
                        phase="move",
                    ),
                    ScheduledProbeAction(
                        kind="move",
                        ticks=ticks,
                        motion_handle_index=second,
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


def compound_timeline(config):
    slot, timeline = 0, []
    for action in compound_schedule(config):
        timeline.append(
            {
                "start_frame": slot,
                "end_frame": slot + action.ticks,
                "scheduled": action.model_dump(mode="json"),
            }
        )
        slot += action.ticks
    return timeline


def compound_pairs(config):
    timeline, pairs = compound_timeline(config), []
    for number, index in enumerate(range(25, len(timeline) - 1, 3)):
        first, second, settled = timeline[index : index + 3]
        for group, begin, end in (
            ("compound_moving", first["start_frame"], second["end_frame"]),
            ("compound_settled", first["start_frame"], settled["end_frame"]),
            ("leg_one", first["start_frame"], first["end_frame"]),
            ("leg_two", second["start_frame"], second["end_frame"]),
            ("settle_control", settled["end_frame"] - 4, settled["end_frame"]),
        ):
            pairs.append(
                {
                    "id": f"{group}-{number:02d}",
                    "group": group,
                    "first": begin,
                    "second": end,
                }
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


@contextmanager
def compound_bindings(module, *, source_function: str, wrapper: Path):
    """Restore only local function aliases; never edit the SDK or historical files.

    Canonical source paths are collected before replacement. Both the hook module
    and wrapper are then included in the raw acquisition's frozen source ledger.
    The same contract is used for public validation, without touching telemetry.
    """
    canonical = {
        "MotionProbeConfig": MotionProbeConfig,
        "motion_probe_schedule": motion_probe_schedule,
        "motion_probe_pairs": motion_probe_pairs,
        "motion_probe_timeline": motion_probe_timeline,
    }
    if source_function not in ("source_paths", "capture_sources") or any(
        getattr(module, name) is not value for name, value in canonical.items()
    ):
        raise ValueError("compound binding requires canonical unmodified aliases")
    factory = getattr(module, source_function)
    paths = factory() | {"compound_motion.py": Path(__file__), wrapper.name: wrapper}
    original = canonical | {source_function: factory}
    replacement = {
        "MotionProbeConfig": CompoundProbeConfig,
        "motion_probe_schedule": compound_schedule,
        "motion_probe_pairs": compound_pairs,
        "motion_probe_timeline": compound_timeline,
        source_function: lambda: dict(paths),
    }
    try:
        for name, value in replacement.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(module, name, value)
