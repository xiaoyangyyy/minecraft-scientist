import pytest
from pydantic import ValidationError

from minecraft_scientist.protocols import ActionKind, PrimitiveAction, RunConfig


def test_models_are_strict_and_frozen() -> None:
    action = PrimitiveAction(kind=ActionKind.WAIT, ticks=2)
    with pytest.raises(ValidationError):
        action.ticks = 3
    with pytest.raises(ValidationError):
        PrimitiveAction(kind=ActionKind.WAIT, ticks=0)
    with pytest.raises(ValidationError):
        PrimitiveAction(kind=ActionKind.WAIT, unknown=True)


def test_run_config_rejects_empty_budget() -> None:
    with pytest.raises(ValidationError):
        RunConfig(intervention_budget=0)


def test_look_action_round_trips_through_public_schema() -> None:
    action = PrimitiveAction(kind="look", target_id="obj_view")
    restored = PrimitiveAction.model_validate_json(action.model_dump_json())
    assert restored.kind is ActionKind.LOOK
    assert restored.target_id == "obj_view"
