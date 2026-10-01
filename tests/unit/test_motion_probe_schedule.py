from itertools import pairwise

import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration.motion_probe_schedule import (
    MotionProbeConfig,
    ScheduledProbeAction,
    motion_probe_pairs,
    motion_probe_schedule,
    motion_probe_timeline,
)


def test_fixed_schedule_has_all_interventions_and_independent_settle_controls():
    config = MotionProbeConfig()
    schedule = motion_probe_schedule(config)
    assert len(schedule) == 42
    assert [
        item.motion_handle_index for item in schedule if item.kind == "move"
    ] == list(range(4)) * 2
    timeline = motion_probe_timeline(config)
    assert timeline[0]["start_frame"] == 0
    assert all(
        left["end_frame"] == right["start_frame"]
        for left, right in pairwise(timeline)
    )
    assert timeline[-1]["end_frame"] == 200
    pairs = motion_probe_pairs(config)
    assert len(pairs) == 30
    assert len({pair["id"] for pair in pairs}) == 30
    for group, count in (
        ("moving", 8),
        ("settled", 8),
        ("settle_control", 8),
        ("rotation_control", 2),
        ("wait_control", 2),
        ("identity_control", 2),
    ):
        assert sum(pair["group"] == group for pair in pairs) == count
    for number in range(8):
        moving, settled, control = pairs[number * 3 : number * 3 + 3]
        assert moving["first"] == settled["first"]
        assert settled["second"] - moving["second"] == 12
        assert control["second"] == settled["second"]
        assert control["first"] > moving["second"]
        assert settled["second"] - settled["first"] + 1 <= 32


def test_public_schedule_is_deterministic_and_does_not_use_telemetry():
    assert motion_probe_schedule(MotionProbeConfig(seed=0)) == motion_probe_schedule(
        MotionProbeConfig(seed=999)
    )
    assert motion_probe_pairs(
        MotionProbeConfig(weather="thunder")
    ) == motion_probe_pairs(MotionProbeConfig(weather="clear"))
    assert MotionProbeConfig(observation_size=224).render_size == (640, 360)
    assert MotionProbeConfig().render_size == (1280, 720)


@pytest.mark.parametrize(
    "change",
    [
        {"seed": -1},
        {"seed": True},
        {"movement_ticks": (6, 3)},
        {"movement_ticks": (0, 6)},
        {"movement_ticks": (3, 20)},
        {"settle_ticks": 2},
        {"observation_size": 640},
        {"xpos": 0},
        {"weather": "unknown"},
    ],
)
def test_invalid_schedule_configuration_fails_before_any_execution(change):
    with pytest.raises(ValidationError):
        MotionProbeConfig.model_validate(change)


def test_action_schema_does_not_allow_motion_handle_on_wait_or_missing_on_move():
    with pytest.raises(ValidationError):
        ScheduledProbeAction(
            kind="wait", ticks=1, phase="settle", motion_handle_index=0
        )
    with pytest.raises(ValidationError):
        ScheduledProbeAction(kind="move", ticks=1, phase="move")
