"""Evidence-driven selection among the four supported symbolic forms."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from minecraft_scientist.protocols import TheoryRepresentation


class RepresentationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    has_continuous_inputs: bool = False
    order_changes_outcome: bool = False
    repeated_composition: bool = False
    relational_predicates: bool = False


class RepresentationSelector:
    """Choose the simplest form capable of expressing observed invariants."""

    @staticmethod
    def select(evidence: RepresentationEvidence) -> TheoryRepresentation:
        if evidence.order_changes_outcome:
            return TheoryRepresentation.STATE_MACHINE
        if evidence.repeated_composition:
            return TheoryRepresentation.PROGRAM
        if evidence.has_continuous_inputs:
            return TheoryRepresentation.EQUATION
        return TheoryRepresentation.RULE
