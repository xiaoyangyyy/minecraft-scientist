import pytest

from minecraft_scientist.discovery.visual_tools import discover_visual_tools
from minecraft_scientist.minecraft.visual_evidence import PixelChangeEvidence
from minecraft_scientist.theory.ir import TheoryInterpreter


CONTEXT = {"attack_frames": 20, "outcome": "appearance_change"}


def test_active_visual_learning_uses_observed_labels_and_scoped_ir():
    calls = []
    def probe(tool, target, index):
        calls.append((tool, target))
        return str(index), PixelChangeEvidence(0, 50 if tool == "obj_b" else 0,
                                               True, tool == "obj_b")
    result = discover_visual_tools(("obj_a", "obj_b"), ("obj_target",), probe,
                                   budget=4, execution_context=CONTEXT)
    assert result.resolved
    assert len(calls) == 2
    assert len(set(calls)) == 2
    assert TheoryInterpreter.predict(result.theory, tool_id="obj_b",
                                     target_id="obj_target",
                                     execution_context=CONTEXT)["success"]


def test_unstable_controls_exhaust_budget_without_promoting_theory():
    def probe(tool, target, index):
        return str(index), PixelChangeEvidence(20, 100, False, None)
    result = discover_visual_tools(("obj_a", "obj_b"), ("obj_target",), probe,
                                   budget=3, execution_context=CONTEXT)
    assert not result.resolved
    assert result.theory is None
    assert len(result.experiments) == 3
    assert result.candidate_count == 6


def test_semantic_identifiers_rejected_before_probe():
    with pytest.raises(ValueError, match="opaque"):
        discover_visual_tools(("iron_pickaxe", "obj_a"), ("obj_target",), None,
                              budget=2, execution_context=CONTEXT)


def test_one_tool_prior_cannot_be_promoted_without_experiments():
    with pytest.raises(ValueError, match="at least two"):
        discover_visual_tools(("obj_a",), ("obj_target",), None,
                              budget=2, execution_context=CONTEXT)


def test_insufficient_budget_retains_unresolved_competition():
    def probe(tool, target, index):
        return str(index), PixelChangeEvidence(0, 50, True, True)
    result = discover_visual_tools(("obj_a", "obj_b"), ("obj_target",), probe,
                                   budget=1, execution_context=CONTEXT)
    assert not result.resolved
    assert result.theory is None


def test_one_failure_does_not_imply_untested_tool_success():
    def probe(tool, target, index):
        return str(index), PixelChangeEvidence(0, 0, True, False)
    partial = discover_visual_tools(("obj_a", "obj_b"), ("obj_target",), probe,
                                    budget=1, execution_context=CONTEXT)
    assert not partial.resolved
    complete = discover_visual_tools(("obj_a", "obj_b"), ("obj_target",), probe,
                                     budget=2, execution_context=CONTEXT)
    assert complete.resolved
    assert len(complete.experiments) == 2
    for tool in ("obj_a", "obj_b"):
        assert not TheoryInterpreter.predict(complete.theory, tool_id=tool,
            target_id="obj_target", execution_context=CONTEXT)["success"]
