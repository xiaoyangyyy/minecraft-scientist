import gc
import pickle

import pytest

from minecraft_scientist.protocols import ActionKind, PrimitiveAction
from minecraft_scientist.world import WorldFactory


def test_agent_view_does_not_expose_ground_truth_capability() -> None:
    bundle = WorldFactory.create(0)
    view = bundle.agent_view
    assert not hasattr(view, "evaluator")
    assert not hasattr(view, "ground_truth")
    assert not hasattr(view, "__dict__")
    assert not any(
        callable(referent) and getattr(referent, "__self__", None) is not None
        for referent in gc.get_referents(view)
    )
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(view)
    public = view.observe().model_dump()
    assert "requirements" not in str(public)
    assert "protocol" not in str(public)


def test_world_is_deterministic_for_same_seed() -> None:
    first = WorldFactory.create(7)
    second = WorldFactory.create(7)
    assert first.agent_view.observe() == second.agent_view.observe()
    actions = first.agent_view.available_actions()[:4]
    records_a = [first.agent_view.step(action) for action in actions]
    records_b = [second.agent_view.step(action) for action in actions]
    assert records_a == records_b


def test_different_seeds_randomize_opaque_ids() -> None:
    first = {
        obj.object_id for obj in WorldFactory.create(1).agent_view.observe().objects
    }
    second = {
        obj.object_id for obj in WorldFactory.create(2).agent_view.observe().objects
    }
    assert first != second


def test_invalid_action_records_control_failure() -> None:
    view = WorldFactory.create(0).agent_view
    record = view.step(PrimitiveAction(kind=ActionKind.SELECT, target_id="missing"))
    assert not record.intervention_compliance
    assert record.controller_failure == "not_selectable"
    assert record.event.event_type == "control_failure"
