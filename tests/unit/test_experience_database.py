from __future__ import annotations

from pathlib import Path

from minecraft_scientist.experience import ExperienceDatabase
from minecraft_scientist.protocols import (
    ActionKind,
    Observation,
    PrimitiveAction,
    TransitionRecord,
    WorldEvent,
)


def _transition(identifier: str) -> TransitionRecord:
    observation = Observation(episode_id=0, step=0, inventory={}, objects=())
    action = PrimitiveAction(kind=ActionKind.WAIT)
    event = WorldEvent(event_type="time_passed")
    return TransitionRecord(
        transition_id=identifier,
        context={},
        state_before=observation,
        executed_action=action,
        intervention_compliance=True,
        state_after=observation.model_copy(update={"step": 1}),
        observed_delta={},
        event=event,
    )


def test_experience_database_round_trip_and_mechanism_query(tmp_path: Path) -> None:
    path = tmp_path / "experience.sqlite3"
    with ExperienceDatabase(path) as database:
        database.append("fall_damage", _transition("a"))
        database.append("tool_capability", _transition("b"))
        assert database.count() == 2
        assert database.count("fall_damage") == 1
        restored = tuple(database.iter_transitions("fall_damage"))
    assert restored == (_transition("a"),)
