"""Finite edit grammar for structural hypotheses, not a supplied replacement table.

The generator has no world interface, observation selector or private truth input.
Enumeration completeness is explicit; an exhausted search never implies unique
identification. State-template insertion counts as two edits with its redirect.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass

from minecraft_scientist.protocols import TheoryIR
from minecraft_scientist.theory.finite_state import FiniteStateProgram


@dataclass(frozen=True)
class StructuralProposal:
    proposal_id: str
    ir: TheoryIR
    edits: tuple[dict, ...]

    def as_json(self):
        return {
            "proposal_id": self.proposal_id,
            "ir": self.ir.model_dump(mode="json"),
            "edits": list(self.edits),
            "edit_count": len(self.edits),
            "encoded_definition_bytes": len(
                _canonical(self.ir.model_dump(mode="json")).encode()
            ),
        }


@dataclass(frozen=True)
class StructuralSearch:
    proposals: tuple[StructuralProposal, ...]
    evaluations: int
    complete: bool
    stop_reason: str | None


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def enumerate_structural_edits(
    base: TheoryIR, *, evaluation_budget=1000, state_insertion=True
):
    if (
        type(evaluation_budget) is not int
        or not 1 <= evaluation_budget <= 100_000
        or type(state_insertion) is not bool
    ):
        raise ValueError("invalid structural search budget/grammar")
    original = FiniteStateProgram.from_ir(base)
    payload = original.model_dump(mode="json")

    def candidates():
        yield payload, ()
        for index, row in enumerate(original.transitions):
            for field, values, operation in (
                ("target", original.states, "redirect_transition"),
                ("output", original.outputs, "change_output"),
            ):
                for value in values:
                    if value == getattr(row, field):
                        continue
                    changed = original.model_dump(mode="json")
                    changed["transitions"][index][field] = value
                    yield (
                        changed,
                        (
                            {
                                "operation": operation,
                                "source": row.source,
                                "action": row.action,
                                "from": getattr(row, field),
                                "to": value,
                            },
                        ),
                    )
        if (
            state_insertion
            and len(original.states) < 16
            and (len(original.states) + 1) * len(original.actions) <= 128
        ):
            name_index = 0
            while f"revision-state-{name_index}" in original.states:
                name_index += 1
            new_state = f"revision-state-{name_index}"
            for template in original.states:
                for index, row in enumerate(original.transitions):
                    changed = original.model_dump(mode="json")
                    changed["states"].append(new_state)
                    for action in original.actions:
                        copied = original.table[template, action]
                        changed["transitions"].append(
                            {
                                "source": new_state,
                                "action": action,
                                "target": new_state
                                if copied.target == template
                                else copied.target,
                                "output": copied.output,
                            }
                        )
                    changed["transitions"][index]["target"] = new_state
                    yield (
                        changed,
                        (
                            {
                                "operation": "add_state_from_template",
                                "template": template,
                                "state": new_state,
                            },
                            {
                                "operation": "redirect_transition",
                                "source": row.source,
                                "action": row.action,
                                "from": row.target,
                                "to": new_state,
                            },
                        ),
                    )

    proposals, seen, evaluations = [], set(), 0
    for changed, edits in candidates():
        if evaluations == evaluation_budget:
            return StructuralSearch(
                tuple(proposals), evaluations, False, "search_budget_exhausted"
            )
        evaluations += 1
        program = FiniteStateProgram.model_validate(changed)
        canonical = _canonical(program.model_dump(mode="json"))
        if canonical in seen:
            continue
        seen.add(canonical)
        program_ir = program.to_ir().model_copy(
            update={
                "scope": deepcopy(base.scope),
                "preconditions": base.preconditions,
                "composition": base.composition,
                "latent_variables": base.latent_variables,
            }
        )
        proposals.append(
            StructuralProposal(
                "mechanism-"
                + hashlib.sha256(
                    _canonical(program_ir.model_dump(mode="json")).encode()
                ).hexdigest()[:20],
                program_ir,
                edits,
            )
        )
    return StructuralSearch(tuple(proposals), evaluations, True, None)
