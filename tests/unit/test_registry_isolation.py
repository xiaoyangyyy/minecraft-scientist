"""Nested-content isolation and positive lineage indices, not only outer frozen."""

import pytest

from minecraft_scientist.protocols import TheoryIR, TheoryRepresentation, TheoryStatus
from minecraft_scientist.theory.registry import TheoryRegistry


def ir():
    return TheoryIR(
        mechanism="snapshot-fixture",
        representation=TheoryRepresentation.RULE,
        scope={"contexts": [{"label": "original"}]},
        effects={"result": {"values": [0]}},
        parameters={"nested": {"values": [0]}},
    )


def propose(registry, program, **kwargs):
    return registry.propose(
        "mechanism",
        program,
        evidence_ids=("evidence",),
        validation_score=0,
        confidence=0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "port",
    ["input", "proposal", "get", "latest", "history", "all_versions", "transition"],
)
@pytest.mark.parametrize("field", ["parameters", "effects", "scope"])
def test_all_registry_boundaries_detach_nested_ir(port, field):
    registry, program = TheoryRegistry(), ir()
    original = program.model_dump(mode="json")
    proposal = propose(registry, program)
    result = {
        "input": lambda: program,
        "proposal": lambda: proposal.ir,
        "get": lambda: registry.get("mechanism", 1).ir,
        "latest": lambda: registry.latest("mechanism").ir,
        "history": lambda: registry.history("mechanism")[0].ir,
        "all_versions": lambda: registry.all_versions()[0].ir,
        "transition": lambda: (
            registry.transition("mechanism", 1, TheoryStatus.PROVISIONAL).ir
        ),
    }[port]()
    if field == "scope":
        result.scope["contexts"][0]["label"] = "modified"
    elif field == "effects":
        result.effects["result"]["values"].append(1)
    else:
        result.parameters["nested"]["values"].append(1)
    assert registry.get("mechanism", 1).ir.model_dump(mode="json") == original


def test_reused_input_and_branch_transition_cannot_change_older_versions():
    registry, program = TheoryRegistry(), ir()
    first = propose(registry, program)
    second = propose(registry, program, parent_version=1)
    second.ir.parameters["nested"]["values"].append(2)
    registry.accept_proposal("mechanism", 1)
    registry.transition(
        "mechanism", 1, TheoryStatus.CHALLENGED, counterexample_ids=("failure",)
    )
    latest = registry.accept_proposal("mechanism", 2)
    latest.ir.effects["result"]["values"].append(3)
    registry.transition("mechanism", 1, TheoryStatus.SUPERSEDED)
    assert first.status is TheoryStatus.PROPOSED
    assert registry.get("mechanism", 1).ir == program
    assert registry.get("mechanism", 2).ir == program
    assert registry.get("mechanism", 2).parent_version == 1
    assert registry.get("mechanism", 1).counterexample_ids == ("failure",)


@pytest.mark.parametrize("number", [0, -1, True, 1.0, "1", 2])
def test_invalid_version_indices_never_alias_a_valid_version(number):
    registry = TheoryRegistry()
    propose(registry, ir())
    with pytest.raises(KeyError):
        registry.get("mechanism", number)
    with pytest.raises(KeyError):
        registry.transition("mechanism", number, TheoryStatus.PROVISIONAL)
    assert registry.get("mechanism", 1).status is TheoryStatus.PROPOSED


@pytest.mark.parametrize("parent", [0, -1, True, 1.0, "1", 2])
def test_invalid_parents_do_not_create_a_version(parent):
    registry = TheoryRegistry()
    propose(registry, ir())
    with pytest.raises(ValueError):
        propose(registry, ir(), parent_version=parent)
    assert len(registry.history("mechanism")) == 1


def test_failed_transition_cannot_publish_invalid_counterexample_metadata():
    registry = TheoryRegistry()
    propose(registry, ir())
    registry.accept_proposal("mechanism", 1)
    with pytest.raises(ValueError):
        registry.transition(
            "mechanism", 1, TheoryStatus.CHALLENGED, counterexample_ids=(object(),)
        )
    assert registry.latest("mechanism").status is TheoryStatus.ACCEPTED
