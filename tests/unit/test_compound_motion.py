from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_scientist.exploration.compound_motion import (
    CompoundProbeConfig,
    compound_bindings,
    compound_pairs,
    compound_schedule,
    compound_timeline,
)
from minecraft_scientist.exploration.motion_probe_schedule import (
    MotionProbeConfig,
    motion_probe_pairs,
    motion_probe_schedule,
    motion_probe_timeline,
)


def test_fixed_compounds_retain_every_leg_and_fit_existing_budgets():
    config = CompoundProbeConfig()
    schedule, timeline, pairs = (
        compound_schedule(config),
        compound_timeline(config),
        compound_pairs(config),
    )
    assert len(schedule) == 50 and timeline[-1]["end_frame"] + 1 == 225
    assert len(pairs) == 46
    assert schedule[:25] == motion_probe_schedule(config)[:25]
    for number, index in enumerate(range(25, len(schedule) - 1, 3)):
        first, second, wait = schedule[index : index + 3]
        assert (
            first.motion_handle_index,
            second.motion_handle_index,
        ) == config.handle_pairs[number % 4]
        assert first.kind == second.kind == "move" and first.ticks == second.ticks
        assert wait.kind == "wait" and wait.ticks == 8
        matching = {
            pair["group"]: pair
            for pair in pairs
            if pair["id"].endswith(f"-{number:02d}")
        }
        assert matching["compound_moving"]["first"] == matching["leg_one"]["first"]
        assert matching["leg_one"]["second"] == matching["leg_two"]["first"]
        assert matching["compound_moving"]["second"] == matching["leg_two"]["second"]
        assert (
            matching["compound_settled"]["second"]
            - matching["compound_settled"]["first"]
            + 1
            <= 32
        )
    assert CompoundProbeConfig.model_validate(config.model_dump()) == config


@pytest.mark.parametrize(
    "change",
    [
        {"handle_pairs": ()},
        {"handle_pairs": ((0, 0),)},
        {"handle_pairs": ((0, 2), (0, 2))},
        {"handle_pairs": ((0, 4),)},
        {"handle_pairs": ((False, 2),)},
        {"settle_ticks": 7},
        {"movement_ticks": (8, 10), "settle_ticks": 16},
    ],
)
def test_invalid_or_nonbounded_compound_schedules_reject(change):
    with pytest.raises(ValueError):
        CompoundProbeConfig(**change)


def _module():
    return SimpleNamespace(
        MotionProbeConfig=MotionProbeConfig,
        motion_probe_schedule=motion_probe_schedule,
        motion_probe_pairs=motion_probe_pairs,
        motion_probe_timeline=motion_probe_timeline,
        source_paths=lambda: {"old.py": Path(__file__)},
    )


def test_compound_aliases_and_source_registry_restore_on_exception():
    module = _module()
    before = vars(module).copy()
    with (
        pytest.raises(RuntimeError, match="injected"),
        compound_bindings(
            module, source_function="source_paths", wrapper=Path(__file__)
        ),
    ):
        assert module.MotionProbeConfig is CompoundProbeConfig
        assert module.motion_probe_schedule is compound_schedule
        assert {"old.py", "compound_motion.py", Path(__file__).name} == set(
            module.source_paths()
        )
        raise RuntimeError("injected")
    assert vars(module) == before


def test_nested_or_noncanonical_aliases_are_rejected_without_overwrite():
    module = _module()
    with compound_bindings(
        module, source_function="source_paths", wrapper=Path(__file__)
    ):
        with (
            pytest.raises(ValueError, match="canonical"),
            compound_bindings(
                module, source_function="source_paths", wrapper=Path(__file__)
            ),
        ):
            pytest.fail("nested binding was accepted")
        assert module.MotionProbeConfig is CompoundProbeConfig
    assert module.MotionProbeConfig is MotionProbeConfig


def test_source_registry_copy_cannot_mutate_frozen_paths():
    module = _module()
    with compound_bindings(
        module, source_function="source_paths", wrapper=Path(__file__)
    ):
        module.source_paths().clear()
        assert len(module.source_paths()) == 3
