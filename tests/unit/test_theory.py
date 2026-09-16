import pytest

from minecraft_scientist.protocols import TheoryIR, TheoryRepresentation, TheoryStatus
from minecraft_scientist.theory import TheoryInterpreter, TheoryRegistry, theory_score


def _ir() -> TheoryIR:
    return TheoryIR(
        mechanism="tool_capability",
        representation=TheoryRepresentation.RULE,
        scope={"tools": ["a"], "targets": ["b"]},
        effects={"success": "rank >= requirement"},
        parameters={"tool_ranks": {"a": 0}, "target_requirements": {"b": 0}},
    )


def test_theory_ir_executes_and_scores() -> None:
    ir = _ir()
    assert TheoryInterpreter.predict(ir, tool_id="a", target_id="b") == {
        "success": True
    }
    assert theory_score(ir, [True], [True]) < theory_score(ir, [False], [True])


def test_registry_enforces_version_status_flow() -> None:
    registry = TheoryRegistry()
    v1 = registry.propose(
        "tool-capability",
        _ir(),
        evidence_ids=("e1",),
        validation_score=0.1,
        confidence=1.0,
    )
    with pytest.raises(ValueError):
        registry.transition(v1.theory_id, v1.version, TheoryStatus.ACCEPTED)
    accepted = registry.accept_proposal(v1.theory_id, v1.version)
    assert accepted.status is TheoryStatus.ACCEPTED
    registry.transition(v1.theory_id, v1.version, TheoryStatus.CHALLENGED)
    v2 = registry.propose(
        "tool-capability",
        _ir(),
        parent_version=1,
        evidence_ids=("e2",),
        validation_score=0.1,
        confidence=1.0,
    )
    assert v2.version == 2
    registry.accept_proposal(v2.theory_id, v2.version)
    superseded = registry.transition(v1.theory_id, 1, TheoryStatus.SUPERSEDED)
    assert superseded.status is TheoryStatus.SUPERSEDED
