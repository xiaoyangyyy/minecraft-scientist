import pytest

from minecraft_scientist.discovery.hypotheses import ToolVersionSpace
from minecraft_scientist.theory.ir import TheoryInterpreter


def scoped_ir():
    space = ToolVersionSpace(("obj_a", "obj_b"), ("obj_target",))
    space.observe("obj_a", "obj_target", False, "e1")
    space.observe("obj_b", "obj_target", True, "e2")
    ir = space.to_ir()
    return ir.model_copy(update={"scope": {
        **ir.scope, "required_context": {"mode": "survival", "attack_frames": 20}
    }})


@pytest.mark.parametrize("context", [None, {}, {"mode": "creative", "attack_frames": 20},
                                    {"mode": "survival", "attack_frames": 200}])
def test_rejects_out_of_scope_visual_theory(context):
    with pytest.raises(ValueError):
        TheoryInterpreter.predict(scoped_ir(), tool_id="obj_b", target_id="obj_target",
                                  execution_context=context)


def test_predicts_in_scope_with_additional_context():
    assert TheoryInterpreter.predict(
        scoped_ir(), tool_id="obj_b", target_id="obj_target",
        execution_context={"mode": "survival", "attack_frames": 20, "seed": 2}
    ) == {"success": True}
