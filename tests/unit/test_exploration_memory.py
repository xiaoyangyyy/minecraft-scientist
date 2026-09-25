from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from minecraft_scientist.exploration import ExplorationMemory, FrontierStatus


def test_multi_view_place_calibration_generalizes_and_rejects_negative(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exploration.sqlite3"
    with ExplorationMemory(
        path,
        default_match_radius=0,
        max_exemplars=3,
    ) as memory:
        place = memory.observe_place((0, 0), episode_id=0, step=0)
        calibrated = memory.calibrate_place(
            place.place_id,
            positive_signatures=((0, 0), (4, 0), (0, 4), (4, 4)),
            negative_signatures=((12, 12),),
            episode_id=0,
            step=1,
            margin=1,
        )

        assert len(calibrated.exemplars) == 3
        assert calibrated.calibration is not None
        assert calibrated.calibration.maximum_positive_distance == 4
        assert calibrated.match_radius == 5
        assert memory.match_place((4, 3)) == place.place_id
        assert memory.match_place((9, 9)) is None
        assert memory.verify_integrity()


def test_place_matching_fails_closed_when_two_places_are_ambiguous(
    tmp_path: Path,
) -> None:
    with ExplorationMemory(
        tmp_path / "ambiguous.sqlite3",
        default_match_radius=2,
        ambiguity_margin=0,
    ) as memory:
        memory.observe_place((0,), episode_id=0, step=0)
        memory.observe_place((4,), episode_id=0, step=1)

        assert memory.match_place((2,)) is None
        with pytest.raises(ValueError, match="ambiguous place observation"):
            memory.observe_place((2,), episode_id=0, step=2)
        assert len(memory.places()) == 2


def test_frontier_failures_persist_and_event_tampering_is_detected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "durable.sqlite3"
    with ExplorationMemory(
        path,
        default_match_radius=0,
        block_after_failures=2,
    ) as memory:
        origin = memory.observe_place((0,), episode_id=0, step=0)
        frontier = memory.register_frontier(
            origin.place_id,
            ("probe_left",),
            episode_id=0,
            step=1,
        )
        memory.record_frontier_outcome(
            frontier.frontier_id,
            episode_id=0,
            step=2,
            success=False,
            cost=1,
        )

    with ExplorationMemory.open(path) as resumed:
        updated = resumed.record_frontier_outcome(
            frontier.frontier_id,
            episode_id=1,
            step=0,
            success=False,
            cost=1,
        )
        assert updated.status is FrontierStatus.BLOCKED
        assert resumed.plan_next_frontier(origin.place_id) is None
        assert resumed.summary()["episode_count"] == 1
        assert resumed.verify_integrity()

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE exploration_events SET payload_json = ? WHERE sequence = 1",
            ('{"tampered":true}',),
        )
    with ExplorationMemory.open(path) as tampered:
        assert not tampered.verify_integrity()


def test_memory_rejects_reopen_with_different_behavioral_configuration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "configured.sqlite3"
    with ExplorationMemory(path, default_match_radius=3):
        pass

    with pytest.raises(ValueError, match="configuration does not match"):
        ExplorationMemory(path, default_match_radius=4)


def test_memory_rejects_tampered_behavioral_configuration_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tampered-configuration.sqlite3"
    with ExplorationMemory(path, default_match_radius=3):
        pass
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'configuration_sha256'",
            ("0" * 64,),
        )

    with pytest.raises(ValueError, match="configuration hash is invalid"):
        ExplorationMemory.open(path)
