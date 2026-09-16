from __future__ import annotations

from minecraft_scientist.protocols import TheoryRepresentation
from minecraft_scientist.symbolic import (
    FallEquationVersionSpace,
    RepresentationEvidence,
    RepresentationSelector,
)
from minecraft_scientist.theory import TheoryInterpreter


def test_active_piecewise_equation_recovery() -> None:
    armor_ids = ("a", "b", "c")
    truth = {"a": 0.7, "b": 1.0, "c": 0.4}
    space = FallEquationVersionSpace(armor_ids, tuple(range(1, 11)))
    count = 0
    while not space.resolved():
        armor_id, height = space.next_experiment() or ("", 0)
        assert armor_id
        damage = truth[armor_id] * max(0, height - 3)
        count += 1
        space.observe(armor_id, height, damage, f"e-{count}")
    ir = space.to_ir()
    assert count < 30
    assert ir.representation is TheoryRepresentation.EQUATION
    assert ir.parameters["safe_height"] == 3
    for armor_id in armor_ids:
        for height in range(1, 11):
            prediction = TheoryInterpreter.predict(ir, armor_id=armor_id, height=height)
            assert prediction["damage"] == truth[armor_id] * max(0, height - 3)


def test_representation_selector_covers_all_symbolic_forms() -> None:
    selector = RepresentationSelector()
    assert selector.select(RepresentationEvidence(has_continuous_inputs=True)) is (
        TheoryRepresentation.EQUATION
    )
    assert selector.select(RepresentationEvidence(order_changes_outcome=True)) is (
        TheoryRepresentation.STATE_MACHINE
    )
    assert selector.select(RepresentationEvidence(repeated_composition=True)) is (
        TheoryRepresentation.PROGRAM
    )
    assert selector.select(RepresentationEvidence(relational_predicates=True)) is (
        TheoryRepresentation.RULE
    )
